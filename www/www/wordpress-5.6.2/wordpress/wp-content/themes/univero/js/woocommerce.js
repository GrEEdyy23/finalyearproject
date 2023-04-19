(function($) {
	"use strict";
	// Ajax QuickView
	jQuery(document).ready(function(){
		jQuery('a.quickview').on('click', function (e) {
			e.preventDefault();
			var self = $(this);
			self.parent().parent().parent().addClass('loading');
		    var productslug = jQuery(this).data('productslug');
		    var url = univero_ajax.ajaxurl + '?action=univero_quickview_product&productslug=' + productslug;
		    
	    	jQuery.get(url,function(data,status){
		    	$.magnificPopup.open({
					mainClass: 'ninzio-mfp-zoom-in',
					items    : {
						src : data,
						type: 'inline'
					}
				});
				// variation
                if ( typeof wc_add_to_cart_variation_params !== 'undefined' ) {
                    $( '.variations_form' ).each( function() {
                        $( this ).wc_variation_form().find('.variations select:eq(0)').change();
                    });
                }
                var config = {
                    loop: false,
                    nav: true,
                    dots: false,
                    items: 1,
                    navText: ['<span class="univero-arrow-left"></span>', '<span class="univero-arrow-right"></span>'],
                    responsive: {
	                    0:{
	                        items: 1
	                    },
	                    320:{
	                        items: 1
	                    },
	                    768:{
	                        items: 1
	                    },
	                    980:{
	                        items: 1
	                    },
	                    1280:{
	                        items: 1
	                    }
	                }
                };
                $(".quickview-owl").owlCarousel( config );
                
				self.parent().parent().parent().removeClass('loading');
		    });
		});
	});
	
	// thumb image
	$('.thumbnails-image .thumb-link, .lite-carousel-play .thumb-link').each(function(e){
		$(this).on('click', function(event){
			event.preventDefault();
			$('.main-image-carousel').trigger("to.owl.carousel", [e, 0, true]);
			
			$('.thumbnails-image .thumb-link').removeClass('active');
			$(this).addClass('active');
			return false;
		});
	});
	$('.main-image-carousel').on('changed.owl.carousel', function(event) {
		setTimeout(function(){
			var index = 0;
			$('.main-image-carousel .owl-item').each(function(i){
				if ($(this).hasClass('active')){
					index = i;
				}
			});
			$('.thumbnails-image .thumb-link').removeClass('active');
			
			if ( $('.thumbnails-image .lite-carousel-play').length > 0 ) {
				$('.thumbnails-image li').eq(index).find('.thumb-link').addClass('active');
			} else {
				$('.thumbnails-image .owl-item').eq(index).find('.thumb-link').addClass('active');
			}
		},50);
    });
	// change thumb variants
	$( 'body' ).on( 'found_variation', function( event, variation ) {
    	if ( variation && variation.image && variation.image.src && variation.image.src.length > 1 ) {
    		$('.main-image-carousel a').each(function(e){
    			var src = $('img', $(this)).attr('src');
    			if (src === variation.image.src) {
	    			$('.main-image-carousel').trigger("to.owl.carousel", [e, 0, true]);
    			}
    		});
    	}
	});
	// review
    $('.woocommerce-review-link').on('click', function(){
        $('.woocommerce-tabs a[href=#tabs-list-reviews]').trigger('click');
        $('html, body').animate({
            scrollTop: $("#reviews").offset().top
        }, 1000);
        return false;
    });
    
})(jQuery)