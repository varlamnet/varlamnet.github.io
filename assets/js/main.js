// DARK MODE ADDED
let systemInitiatedDark = window.matchMedia("(prefers-color-scheme: dark)"); 
let theme = sessionStorage.getItem('theme');

if (systemInitiatedDark.matches) {
	document.getElementById("theme-toggle").className = "fa fa-sun glow";
} else {
	document.getElementById("theme-toggle").className = "far fa-moon glow";
}

function prefersColorTest(systemInitiatedDark) {
  if (systemInitiatedDark.matches) {
	  document.documentElement.setAttribute('data-theme', 'dark');
	  document.getElementById("theme-toggle").className = "fa fa-sun glow";		
   	sessionStorage.setItem('theme', '');
  } else {
	  document.documentElement.setAttribute('data-theme', 'light');
	  document.getElementById("theme-toggle").className = "far fa-moon glow";
    sessionStorage.setItem('theme', '');
  }
}
systemInitiatedDark.addListener(prefersColorTest);


function modeSwitcher() {
	let theme = sessionStorage.getItem('theme');
	if (theme === "dark") {
		document.documentElement.setAttribute('data-theme', 'light');
		sessionStorage.setItem('theme', 'light');
		document.getElementById("theme-toggle").className = "far fa-moon glow";

	}	else if (theme === "light") {
		document.documentElement.setAttribute('data-theme', 'dark');
		sessionStorage.setItem('theme', 'dark');
		document.getElementById("theme-toggle").className = "fa fa-sun glow";

	} else if (systemInitiatedDark.matches) {	
		document.documentElement.setAttribute('data-theme', 'light');
		sessionStorage.setItem('theme', 'light');
		//let theme = sessionStorage.getItem('theme');
		//console.log("this was triggered");
		document.getElementById("theme-toggle").className = "far fa-moon glow";

	} else {
		document.documentElement.setAttribute('data-theme', 'dark');
		sessionStorage.setItem('theme', 'dark');
		document.getElementById("theme-toggle").className = "fa fa-sun glow";

	}
}

if (theme === "dark") {
	document.documentElement.setAttribute('data-theme', 'dark');
	sessionStorage.setItem('theme', 'dark');
	document.getElementById("theme-toggle").className = "fa fa-sun glow";
} else if (theme === "light") {
	document.documentElement.setAttribute('data-theme', 'light');
	sessionStorage.setItem('theme', 'light');
	document.getElementById("theme-toggle").className = "far fa-moon glow";
}
//DARK MODE ADDED

var BeautifulJekyllJS = {

  bigImgEl : null,
  numImgs : null,

  init : function() {
    // Shorten the navbar after scrolling a little bit down
    $(window).scroll(function() {
        if ($(".navbar").offset().top > 50) {
            $(".navbar").addClass("top-nav-short");
        } else {
            $(".navbar").removeClass("top-nav-short");
        }
    });

    // On mobile, hide the avatar when expanding the navbar menu
    $('#main-navbar').on('show.bs.collapse', function () {
      $(".navbar").addClass("top-nav-expanded");
      document.getElementById("theme-toggle").className = "null";                   // DARK MODE ADDED 
    });
    $('#main-navbar').on('hidden.bs.collapse', function () {
      $(".navbar").removeClass("top-nav-expanded");
      if (document.documentElement.getAttribute('data-theme') === "dark") {         // DARK MODE ADDED 
        document.getElementById("theme-toggle").className = "fa fa-sun glow";       // DARK MODE ADDED 
      }	else if (document.documentElement.getAttribute('data-theme') === "light") { // DARK MODE ADDED 
        document.getElementById("theme-toggle").className = "far fa-moon glow";     // DARK MODE ADDED 
      }                                                                             // DARK MODE ADDED 
    });

    // show the big header image
    BeautifulJekyllJS.initImgs();
  },

  initImgs : function() {
    // If the page was large images to randomly select from, choose an image
    if ($("#header-big-imgs").length > 0) {
      BeautifulJekyllJS.bigImgEl = $("#header-big-imgs");
      BeautifulJekyllJS.numImgs = BeautifulJekyllJS.bigImgEl.attr("data-num-img");

      // 2fc73a3a967e97599c9763d05e564189
      // set an initial image
      var imgInfo = BeautifulJekyllJS.getImgInfo();
      var src = imgInfo.src;
      var desc = imgInfo.desc;
      BeautifulJekyllJS.setImg(src, desc);

      // For better UX, prefetch the next image so that it will already be loaded when we want to show it
      var getNextImg = function() {
        var imgInfo = BeautifulJekyllJS.getImgInfo();
        var src = imgInfo.src;
        var desc = imgInfo.desc;

        var prefetchImg = new Image();
        prefetchImg.src = src;
        // if I want to do something once the image is ready: `prefetchImg.onload = function(){}`

        setTimeout(function(){
          var img = $("<div></div>").addClass("big-img-transition").css("background-image", 'url(' + src + ')');
          $(".intro-header.big-img").prepend(img);
          setTimeout(function(){ img.css("opacity", "1"); }, 50);

          // after the animation of fading in the new image is done, prefetch the next one
          //img.one("transitioned webkitTransitionEnd oTransitionEnd MSTransitionEnd", function(){
          setTimeout(function() {
            BeautifulJekyllJS.setImg(src, desc);
            img.remove();
            getNextImg();
          }, 1000);
          //});
        }, 6000);
      };

      // If there are multiple images, cycle through them
      if (BeautifulJekyllJS.numImgs > 1) {
        getNextImg();
      }
    }
  },

  getImgInfo : function() {
    var randNum = Math.floor((Math.random() * BeautifulJekyllJS.numImgs) + 1);
    var src = BeautifulJekyllJS.bigImgEl.attr("data-img-src-" + randNum);
    var desc = BeautifulJekyllJS.bigImgEl.attr("data-img-desc-" + randNum);

    return {
      src : src,
      desc : desc
    }
  },

  setImg : function(src, desc) {
    $(".intro-header.big-img").css("background-image", 'url(' + src + ')');
    if (typeof desc !== typeof undefined && desc !== false) {
      $(".img-desc").text(desc).show();
    } else {
      $(".img-desc").hide();
    }
  }
};

// 2fc73a3a967e97599c9763d05e564189

document.addEventListener('DOMContentLoaded', BeautifulJekyllJS.init);

// Bootstrap tooltip added!
$(function () {
  $('[data-toggle="tooltip"]').tooltip()
})

// Modal
$('#myModal').on('shown.bs.modal', function () {
  $('#myInput').trigger('focus')
})