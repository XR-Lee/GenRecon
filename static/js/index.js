window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


function initBeforeAfterSlider(containerId) {
  var container = document.getElementById(containerId);
  if (!container) return;

  var before = container.querySelector('.bal-before');
  var beforeInset = container.querySelector('.bal-before-inset');
  var handle = container.querySelector('.bal-handle');
  var afterMedia = container.querySelector('.bal-after video, .bal-after img');
  var beforeVideo = container.querySelector('.bal-before-inset video');
  var afterVideo = container.querySelector('.bal-after video');

  var isDragging = false;

  function setPosition(pct) {
    pct = Math.max(0, Math.min(100, pct));
    before.style.width = pct + '%';
    beforeInset.style.width = container.offsetWidth + 'px';
    handle.style.left = pct + '%';
  }

  function getClientX(e) {
    return e.touches ? e.touches[0].clientX : e.clientX;
  }

  function onMove(e) {
    if (!isDragging) return;
    var rect = container.getBoundingClientRect();
    setPosition(((getClientX(e) - rect.left) / rect.width) * 100);
  }

  function onUp() { isDragging = false; }
  function onDown(e) { isDragging = true; e.preventDefault(); }

  container.addEventListener('mousedown', onDown);
  container.addEventListener('touchstart', onDown, { passive: false });
  document.addEventListener('mousemove', onMove);
  document.addEventListener('touchmove', onMove, { passive: false });
  document.addEventListener('mouseup', onUp);
  document.addEventListener('touchend', onUp);

  window.addEventListener('resize', function() { setPosition(50); });

  function init() {
    beforeInset.style.width = container.offsetWidth + 'px';
    setPosition(50);
  }

  if (afterMedia && afterMedia.tagName === 'IMG') {
    if (afterMedia.complete) { init(); } else { afterMedia.addEventListener('load', init); }
  } else if (afterMedia) {
    if (afterMedia.readyState >= 1) { init(); } else { afterMedia.addEventListener('loadedmetadata', init); }
  }

  // Keep the two videos in this slider in sync (resync if they drift > 0.1s).
  if (beforeVideo && afterVideo) {
    afterVideo.addEventListener('timeupdate', function() {
      if (Math.abs(beforeVideo.currentTime - afterVideo.currentTime) > 0.1) {
        beforeVideo.currentTime = afterVideo.currentTime;
      }
    });
  }
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

    ['cmp-ours', 'cmp-2dgs', 'cmp-da3', 'cmp-finerecon', 'cmp-monosdf', 'cmp-murre']
      .forEach(initBeforeAfterSlider);

})
