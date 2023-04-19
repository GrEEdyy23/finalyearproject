
jQuery(document).ready(function($) {
    'use strict';
	
	$('body').delegate(".input_datetime", 'hover', function(e){
        e.preventDefault();
        $(this).datepicker({
               defaultDate: "",
               dateFormat: "yy-mm-dd",
               numberOfMonths: 1,
               showButtonPanel: true,
        });
    });
});	