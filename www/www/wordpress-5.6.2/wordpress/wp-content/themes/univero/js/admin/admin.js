!function($) {
    'use strict';
    	
	$( "body" ).on( "click", ".ninzio-checkbox", function() {
		
			jQuery('.'+this.id).toggle();
		
    });
    $('.ninzio-wpcolorpicker').each(function(){
    	$(this).wpColorPicker();
    });
}(window.jQuery);
