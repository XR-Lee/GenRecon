from typing import Optional

import torch
from torch import nn


def project_points_to_patches(
    coords_hom: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    img_patch_res: int,
    global_img_tokens: int,
    eps: float = 1e-6,
):
    """Project homogeneous world points into image-patch token ids.

    Single source of truth for the projection convention shared by the dense
    (:class:`Projection`) and sparse projection paths. All inputs must be
    broadcast-aligned on their leading dims:

    Args:
        coords_hom: ``[..., 4]`` homogeneous world coordinates.
        extrinsics: ``[..., 4, 4]`` world -> camera extrinsics.
        intrinsics: ``[..., 3, 3]`` camera intrinsics (normalised uv).
        img_patch_res: number of patches per image side.
        global_img_tokens: token offset reserving the leading global tokens.

    Returns:
        cam: ``[..., 3]`` coordinates in camera frame.
        valid: ``[...]`` bool mask (positive depth and uv inside ``[0, 1)``).
        patch_ids: ``[...]`` long index into the per-view token sequence.
    """
    cam = torch.matmul(coords_hom.unsqueeze(-2), extrinsics.transpose(-1, -2)).squeeze(-2)[..., :3]

    depth = cam[..., 2]
    valid_depth = depth > eps
    safe_depth = torch.where(valid_depth, depth, torch.ones_like(depth))

    uv1 = torch.matmul((cam / safe_depth.unsqueeze(-1)).unsqueeze(-2), intrinsics.transpose(-1, -2)).squeeze(-2)
    u, v = uv1[..., 0], uv1[..., 1]
    valid = valid_depth & (u >= 0.0) & (u < 1.0) & (v >= 0.0) & (v < 1.0)

    patch_u = torch.floor(u * img_patch_res).long().clamp(0, img_patch_res - 1)
    patch_v = torch.floor(v * img_patch_res).long().clamp(0, img_patch_res - 1)
    patch_ids = patch_v * img_patch_res + patch_u + global_img_tokens

    return cam, valid, patch_ids


def points_in_bounds(points: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    bounds = bounds.to(device=points.device, dtype=points.dtype)
    if bounds.numel() == 0:
        return torch.zeros(len(points), dtype=torch.bool, device=points.device)
    lower = bounds[:, 0].unsqueeze(1)
    upper = bounds[:, 1].unsqueeze(1)
    return (((points.unsqueeze(0) >= lower) & (points.unsqueeze(0) <= upper)).all(dim=-1)).any(dim=0)


def _ownership_roi_mask(points: torch.Tensor, ownership: dict) -> torch.Tensor:
    return points_in_bounds(points, ownership["roi_bounds"])


def apply_projection_ownership(
    points: torch.Tensor,
    coords_cam: torch.Tensor,
    patch_ids: torch.Tensor,
    valid: torch.Tensor,
    ownership: Optional[dict],
    global_img_tokens: int,
) -> torch.Tensor:
    """Restrict in-frustum projection candidates with SAM/track ownership.

    Outside the registered object ROI, the original validity rule is preserved.
    Inside it, ``mask`` mode requires the projected patch to belong to the
    prompted object. ``depth`` mode additionally keeps only a track-depth
    surface band; patches without track depth can optionally remain unknown.
    """
    if ownership is None:
        return valid
    mask = ownership["mask"].to(device=valid.device)
    if mask.shape[0] != valid.shape[0]:
        raise ValueError(f"ownership has {mask.shape[0]} views, projection has {valid.shape[0]}")
    local_patch_ids = patch_ids - global_img_tokens
    if local_patch_ids.min() < 0 or local_patch_ids.max() >= mask.shape[1]:
        raise ValueError("ownership patch grid does not match the projection token grid")
    owned_patch = mask.gather(1, local_patch_ids)
    if ownership.get("global_feature_filter", False):
        inside = torch.ones((1, len(points)), dtype=torch.bool, device=valid.device)
    else:
        inside = _ownership_roi_mask(points, ownership).unsqueeze(0)
    allowed = owned_patch
    if ownership["mode"] == "depth":
        depth_valid = ownership["depth_valid"].to(device=valid.device).gather(1, local_patch_ids)
        surface_depth = ownership["surface_depth"].to(
            device=coords_cam.device, dtype=coords_cam.dtype
        ).gather(1, local_patch_ids)
        surface = depth_valid & (
            (coords_cam[..., 2] - surface_depth).abs() <= float(ownership["surface_band"])
        )
        unknown = owned_patch & ~depth_valid if ownership.get("allow_unknown", True) else torch.zeros_like(surface)
        allowed = owned_patch & (surface | unknown)
    return valid & (~inside | allowed)


def classify_projection_ownership(
    points: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    ownership: dict,
    *,
    img_patch_res: int,
    global_img_tokens: int = 0,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Classify object-ROI points as free/surface/occluded from track depth."""
    if ownership["mode"] != "depth":
        raise ValueError("free-space classification requires depth ownership mode")
    if extrinsics.shape[0] != 1 or intrinsics.shape[0] != 1:
        raise ValueError("projection ownership classification expects one scene batch")
    num_views = extrinsics.shape[1]
    ones = torch.ones((len(points), 1), device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=1).unsqueeze(0)
    coords_cam, valid, patch_ids = project_points_to_patches(
        points_h,
        extrinsics.reshape(num_views, 1, 4, 4),
        intrinsics.reshape(num_views, 1, 3, 3),
        img_patch_res,
        global_img_tokens,
        eps=eps,
    )
    mask = ownership["mask"].to(device=points.device)
    local_patch_ids = patch_ids - global_img_tokens
    owned = mask.gather(1, local_patch_ids)
    known = ownership["depth_valid"].to(device=points.device).gather(1, local_patch_ids)
    reference = ownership["surface_depth"].to(
        device=points.device, dtype=coords_cam.dtype
    ).gather(1, local_patch_ids)
    ray_valid = valid & owned & known
    delta = coords_cam[..., 2] - reference
    band = float(ownership["surface_band"])
    free = ray_valid & (delta < -band)
    surface = ray_valid & (delta.abs() <= band)
    occluded = ray_valid & (delta > band)
    inside = _ownership_roi_mask(points, ownership)
    support_bounds = ownership.get("support_bounds")
    if support_bounds is not None and support_bounds.numel():
        inside_support = points_in_bounds(points, support_bounds)
    else:
        inside_support = torch.ones(len(points), dtype=torch.bool, device=points.device)
    free_votes = free.sum(dim=0)
    surface_votes = surface.sum(dim=0)
    occluded_votes = occluded.sum(dim=0)
    depth_free = inside & (free_votes >= int(ownership["min_free_views"])) & (surface_votes == 0)
    outside_support = ~inside_support if ownership.get("hard_support", False) else torch.zeros_like(inside_support)
    hard_free = depth_free | outside_support
    return {
        "inside_roi": inside,
        "inside_support": inside_support,
        "outside_support": outside_support,
        "depth_free": depth_free,
        "free_votes": free_votes,
        "surface_votes": surface_votes,
        "occluded_votes": occluded_votes,
        "hard_free": hard_free,
    }


class Projection(nn.Module):
    def __init__(
        self,
        grid_res: int,
        img_patch_res: int,
        use_camera: bool = False,
        refinement_factor: int = 1,
        global_img_tokens: int = 5,
    ):
        super().__init__()
        self.use_camera = use_camera

        voxels = torch.meshgrid(*[torch.arange(res) for res in [grid_res * refinement_factor] * 3], indexing="ij")
        voxels = torch.stack(voxels, dim=-1).reshape(-1, 3)
        coords_world = (voxels.float() + 0.5) / (grid_res * refinement_factor) - 0.5
        self.num_voxels = coords_world.shape[0]
        ones = torch.ones(self.num_voxels, 1)
        self.coords_homg = torch.cat([coords_world, ones], dim=-1)

        self.img_patch_res = img_patch_res
        self.global_img_tokens = global_img_tokens

    @staticmethod
    def _build_voxelwise_camera_enc(
        voxel_coords: torch.Tensor,
        extrinsics: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Args:
            voxel_coords: [M, V, 3] voxel coordinates in camera frame
            extrinsics: [M, 4, 4] world -> camera extrinsics

        Returns:
            camera_cond: [M, V, 5]
                [elevation, sin(azimuth), cos(azimuth), radius, voxel_distance]
        """
        R = extrinsics[:, :3, :3]  # [M, 3, 3]
        t = extrinsics[:, :3, 3]  # [M, 3]

        # --- Camera Extrinsics ---
        cam_center = -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)  # [M, 3]

        x, y, z = cam_center.unbind(dim=-1)
        radius = torch.linalg.norm(cam_center, dim=-1).clamp_min(eps)  # [M]
        azimuth = torch.atan2(x, z)  # [M]
        elevation = torch.atan2(y, torch.sqrt(x.square() + z.square()).clamp_min(eps))  # [M]

        cam_embed = torch.stack(
            [elevation, torch.sin(azimuth), torch.cos(azimuth), radius],
            dim=-1,
        )  # [M, 4]

        cam_embed = cam_embed[:, None, :].expand(-1, voxel_coords.shape[1], -1)  # [M, V, 4]

        # --- Distance from Camera to Voxel ---
        voxel_distance = torch.linalg.norm(voxel_coords, dim=-1, keepdim=True)  # [M, V, 1]

        return torch.cat([cam_embed, voxel_distance], dim=-1)  # [M, V, 5]

    def forward(self, cond, extrinsics, intrinsics, eps=1e-6):
        B, N, T, D = cond.shape
        V = self.num_voxels
        assert (
            T == self.img_patch_res**2 + self.global_img_tokens
        ), f"expected {self.img_patch_res **2 + self.global_img_tokens} tokens but got {T}"
        assert (
            B,
            N,
            4,
            4,
        ) == extrinsics.shape, f"expected {(B, N, 4, 4)} as extrinsics shape but got {extrinsics.shape}"
        assert (
            B,
            N,
            3,
            3,
        ) == intrinsics.shape, f"expected {(B, N, 3, 3)} as intrinsics shape but got {intrinsics.shape}"

        M = B * N
        cond_f = cond.view(M, T, D)
        extr_f = extrinsics.view(M, 4, 4)
        intr_f = intrinsics.view(M, 3, 3)

        coords_homg = self.coords_homg.to(device=cond.device, dtype=extr_f.dtype).unsqueeze(0)  # [1, V, 4]
        # extr/intr carry an explicit singleton voxel dim so the shared helper
        # broadcasts the [1, V, 4] grid across the M = B*N cameras.
        coords_cam, valid, patch_ids = project_points_to_patches(
            coords_homg,
            extr_f.unsqueeze(1),  # [M, 1, 4, 4]
            intr_f.unsqueeze(1),  # [M, 1, 3, 3]
            self.img_patch_res,
            self.global_img_tokens,
            eps=eps,
        )  # coords_cam [M, V, 3], valid [M, V], patch_ids [M, V]
        gather_idx = patch_ids.unsqueeze(-1).expand(-1, -1, D)  # [M, V, D]

        projection = cond_f.gather(dim=1, index=gather_idx)  # [M, V, D]
        projection *= valid.unsqueeze(-1)

        projection = projection.view(B, N, V, D)
        valid = valid.view(B, N, V)

        if self.use_camera:
            camera_emb = self._build_voxelwise_camera_enc(coords_cam, extr_f, eps=eps)  # [M, V, 5]
            camera_emb = camera_emb.to(projection.dtype)
            camera_emb = camera_emb.view(B, N, V, 5)

            return projection, valid, camera_emb

        return projection, valid, None


def project_features_on_points(
    proj,
    points: torch.Tensor,  # [S, 3] in frame of `extrinsics`
    cond: torch.Tensor,  # [1, N, T, D]
    extrinsics: torch.Tensor,  # [1, N, 4, 4]
    intrinsics: torch.Tensor,  # [1, N, 3, 3]
    eps: float = 1e-6,
    ownership: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Project image patch features onto arbitrary voxel positions.

    Same math as :meth:`Projection.forward` but reads voxel positions from
    ``points`` instead of the module's precomputed [-0.5, 0.5]^3 cube buffer.
    ``proj`` may be the dense :class:`Projection` or the sparse projection
    module (both expose ``img_patch_res`` / ``global_img_tokens``); only the
    dense one carries ``use_camera`` / camera encoding. Used to project a
    scene-wide non-cube grid (dense) or a deduplicated union of chunk sparse
    coords.
    """
    B, N, T, D = cond.shape
    assert B == 1, "project_features_on_points operates on single-scene batches"
    S = points.shape[0]
    assert (
        T == proj.img_patch_res**2 + proj.global_img_tokens
    ), f"expected {proj.img_patch_res ** 2 + proj.global_img_tokens} tokens, got {T}"

    M = B * N
    cond_f = cond.view(M, T, D)
    extr_f = extrinsics.view(M, 4, 4)
    intr_f = intrinsics.view(M, 3, 3)

    ones = torch.ones((S, 1), dtype=points.dtype, device=points.device)
    coords_homg = torch.cat([points, ones], dim=-1).unsqueeze(0)  # [1, S, 4]

    coords_cam = torch.matmul(coords_homg, extr_f.transpose(1, 2))[..., :3]  # [M, S, 3]
    depth = coords_cam[..., 2]
    valid_depth = depth > eps
    safe_depth = torch.where(valid_depth, depth, torch.ones_like(depth))
    xyz = coords_cam / safe_depth.unsqueeze(-1)

    uv1 = torch.matmul(xyz, intr_f.transpose(1, 2))
    u, v = uv1[..., 0], uv1[..., 1]
    valid_uv = (u >= 0.0) & (u < 1.0) & (v >= 0.0) & (v < 1.0)
    valid = valid_depth & valid_uv  # [M, S]

    patch_u = torch.floor(u * proj.img_patch_res).long().clamp(0, proj.img_patch_res - 1)
    patch_v = torch.floor(v * proj.img_patch_res).long().clamp(0, proj.img_patch_res - 1)
    patch_ids = patch_v * proj.img_patch_res + patch_u + proj.global_img_tokens
    valid = apply_projection_ownership(
        points,
        coords_cam,
        patch_ids,
        valid,
        ownership,
        proj.global_img_tokens,
    )
    gather_idx = patch_ids.unsqueeze(-1).expand(-1, -1, D)

    projection = cond_f.gather(dim=1, index=gather_idx)  # [M, S, D]
    projection = projection * valid.unsqueeze(-1)

    projection = projection.view(B, N, S, D)
    valid = valid.view(B, N, S)

    camera_emb = None
    # `use_camera` lives only on the dense Projection; sparse Projection
    # (SLat stages) never sets it — structured_latent_flow asserts against
    # use_camera at add_projections time.
    if getattr(proj, "use_camera", False):
        camera_emb = proj._build_voxelwise_camera_enc(coords_cam, extr_f, eps=eps)
        camera_emb = camera_emb.to(projection.dtype).view(B, N, S, 5)
    return projection, valid, camera_emb
