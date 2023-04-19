<?php

if ( function_exists('vc_map') && class_exists('WPBakeryShortCode') ) {
	if ( !function_exists('univero_load_event_element')) {
		function univero_load_event_element() {
			vc_map( array(
				'name'        => esc_html__( 'Ninzio Events','univero'),
				'base'        => 'ninzio_events',
				"category" => esc_html__('Ninzio Event', 'univero'),
				'description' => esc_html__( 'Create Events for one Widget', 'univero' ),
				"params"      => array(
					array(
						"type" => "dropdown",
						"heading" => esc_html__("Get event By", 'univero'),
						"param_name" => "orderby",
						'value' 	=> array(
							esc_html__('Featured Events', 'univero') => 'featured', 
							esc_html__('Lastest Events', 'univero') => 'most_recent', 
							esc_html__('Randown Events', 'univero') => 'random', 
						)
					),
					array(
						'type' => 'textfield',
						'heading' => esc_html__( 'Number', 'univero' ),
						'param_name' => 'number',
						"admin_label" => true,
						'value' => 4
					),
					array(
						"type" => "dropdown",
						"heading" => esc_html__("Layout Type", 'univero'),
						"param_name" => "layout_type",
						'value' 	=> array(
							esc_html__('Grid', 'univero') => 'grid',
							esc_html__('Carousel', 'univero') => 'carousel',
						)
					),
					array(
						"type" => "dropdown",
						"heading" => esc_html__("Columns", 'univero'),
						"param_name" => "columns",
						'value' 	=> array(1,2,3,4,6)
					),
					array(
						'type' => 'checkbox',
						'heading' => esc_html__( 'Show Navigation', 'univero' ),
						'param_name' => 'show_nav',
						'value' => array( esc_html__( 'Yes, to show navigation', 'univero' ) => 'yes' ),
						'dependency' => array(
							'element' => 'layout_type',
							'value' => array('carousel'),
						),
					),
					array(
						'type' => 'checkbox',
						'heading' => esc_html__( 'Show Pagination', 'univero' ),
						'param_name' => 'show_pagination',
						'value' => array( esc_html__( 'Yes, to show Pagination', 'univero' ) => 'yes' ),
						'dependency' => array(
							'element' => 'layout_type',
							'value' => array('carousel'),
						),
					),
					array(
						'type' => 'textfield',
						'heading' => esc_html__( 'Extra class name', 'univero' ),
						'param_name' => 'el_class',
						'description' => esc_html__( 'If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.', 'univero' )
					)
				),
			));
		}
	}
	add_action( 'vc_after_set_mode', 'univero_load_event_element', 99 );

	class WPBakeryShortCode_ninzio_events extends WPBakeryShortCode {}
}