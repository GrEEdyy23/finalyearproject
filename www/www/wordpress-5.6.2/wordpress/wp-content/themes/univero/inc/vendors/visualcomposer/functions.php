<?php

if ( function_exists('ninzio_framework_add_param') ) {
	ninzio_framework_add_param();
}


function univero_custom_fontawesome($icons){
	$fontawesome_icons = array(
		'Univero' => array(
			array( 'univero-padlock' => 'univero-padlock' ),
			array( 'univero-padlock-unlock' => 'univero-padlock-unlock' ),
			array( 'u-icons-woman-with-skirt' => 'univero-padlock-unlock' ),
			array( 'u-icons-notebook' => 'univero-padlock-unlock' ),
			array( 'u-icons-chat' => 'univero-padlock-unlock' ),
			array( 'u-icons-bar-chart2' => 'univero-padlock-unlock' ),
			array( 'u-icons-avatar' => 'univero-padlock-unlock' ),
			array( 'u-icons-avatar-1' => 'univero-padlock-unlock' ),
			array( 'u-icons-book2' => 'univero-padlock-unlock' ),
			array( 'u-icons-chat2' => 'univero-padlock-unlock' ),
			array( 'u-icons-chat-1' => 'univero-padlock-unlock' ),
			array( 'u-icons-chat-2' => 'univero-padlock-unlock' ),
		),
	);
	return array_merge( $fontawesome_icons, $icons );
}
add_action( 'vc_iconpicker-type-fontawesome', 'univero_custom_fontawesome', 99 );

function univero_admin_init_scripts(){
	//load font
	wp_enqueue_style( 'font-univero', get_template_directory_uri() . '/css/font-univero.css', array(), '1.8.0' );
	wp_enqueue_style( 'font-univero-content', get_template_directory_uri() . '/css/font-univero-content.css', array(), '1.8.0' );
	
	$key = 'AIzaSyAgLtmIukM56mTfet5MEoPsng51Ws06Syc';
	wp_enqueue_script('googlemap-api', '//maps.googleapis.com/maps/api/js?sensor=false&amp;libraries=places&amp;key='.$key );
	wp_enqueue_script('jquery-geocomplete', get_template_directory_uri().'/js/admin/jquery.geocomplete.min.js');

	wp_enqueue_script('jquery-ui-datepicker');
	wp_enqueue_style('jquery-ui-css', get_template_directory_uri() . '/css/jquery-ui.css' );
	wp_enqueue_script( 'univero-admin-scripts', get_template_directory_uri() . '/js/admin/custom.js', array( 'jquery'  ), '20131022', true );
}
add_action( 'admin_enqueue_scripts', 'univero_admin_init_scripts' );

function univero_map_init_scripts() {
	$key = 'AIzaSyAgLtmIukM56mTfet5MEoPsng51Ws06Syc';
	wp_enqueue_script('googlemap-api', '//maps.google.com/maps/api/js?key='.$key);
	wp_enqueue_script('gmap3', get_template_directory_uri().'/js/gmap3.js');
}
add_action('wp_enqueue_scripts', 'univero_map_init_scripts');
