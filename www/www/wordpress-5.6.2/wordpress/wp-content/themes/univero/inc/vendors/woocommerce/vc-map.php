<?php

if ( function_exists('vc_map') && class_exists('WPBakeryShortCode') ) {
	
	if ( !function_exists('univero_woocommerce_get_categories') ) {
	    function univero_woocommerce_get_categories() {
	        $return = array( esc_html__(' --- Choose a Category --- ', 'univero') => '' );

	        $args = array(
	            'type' => 'post',
	            'child_of' => 0,
	            'orderby' => 'name',
	            'order' => 'ASC',
	            'hide_empty' => false,
	            'hierarchical' => 1,
	            'taxonomy' => 'product_cat'
	        );

	        $categories = get_categories( $args );
	        univero_get_category_childs( $categories, 0, 0, $return );

	        return $return;
	    }
	}

	if ( !function_exists('univero_get_category_childs') ) {
	    function univero_get_category_childs( $categories, $id_parent, $level, &$dropdown ) {
	        foreach ( $categories as $key => $category ) {
	            if ( $category->category_parent == $id_parent ) {
	                $dropdown = array_merge( $dropdown, array( str_repeat( "- ", $level ) . $category->name => $category->slug ) );
	                unset($categories[$key]);
	                univero_get_category_childs( $categories, $category->term_id, $level + 1, $dropdown );
	            }
	        }
	    }
	}

	if ( !function_exists('univero_load_woocommerce_element')) {
		function univero_load_woocommerce_element() {
			$types = array(
				esc_html__('Recent Products', 'univero' ) => 'recent_product',
				esc_html__('Best Selling', 'univero' ) => 'best_selling',
				esc_html__('Featured Products', 'univero' ) => 'featured_product',
				esc_html__('Top Rate', 'univero' ) => 'top_rate',
				esc_html__('On Sale', 'univero' ) => 'on_sale',
				esc_html__('Recent Review', 'univero' ) => 'recent_review'
			);
			vc_map( array(
				'name'        => esc_html__( 'Ninzio Products','univero'),
				'base'        => 'ninzio_products',
				"category" => esc_html__('Ninzio Woocommerce', 'univero'),
				'description' => esc_html__( 'Display products in frontend', 'univero' ),
				"params"      => array(
					array(
						"type" => "dropdown",
						"heading" => esc_html__("Get Products By",'univero'),
						"param_name" => "type",
						"value" => $types,
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
						"heading" => esc_html__("Columns", 'univero'),
						"param_name" => "columns",
						'value' 	=> array(1,2,3,4,6)
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

			$categories = array();
			if ( is_admin() ) {
				$categories = univero_woocommerce_get_categories();
			}
			vc_map( array(
				'name'        => esc_html__( 'Ninzio Product Categories','univero'),
				'base'        => 'ninzio_product_categories',
				"category" => esc_html__('Ninzio Woocommerce', 'univero'),
				'description' => esc_html__( 'Display product categories in frontend', 'univero' ),
				"params"      => array(
					array(
						'type' => 'param_group',
						'heading' => esc_html__('Categories Settings', 'univero' ),
						'param_name' => 'categories',
						'params' => array(
							array(
				                "type" => "textfield",
				                "heading" => esc_html__('Title','univero'),
				                "param_name" => "title",
				            ),
							array(
								"type" => "attach_image",
								'heading'	=> esc_html__('Image', 'univero' ),
								"param_name" => "image",
							),
							array(
								"type" => "dropdown",
								"heading" => esc_html__( 'Category', 'univero' ),
								"param_name" => "category",
								"value" => $categories
							),
						),
					),
					array(
						"type" => "dropdown",
						"heading" => esc_html__("Columns", 'univero'),
						"param_name" => "columns",
						'value' 	=> array(1,2,3,4,6)
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
	add_action( 'vc_after_set_mode', 'univero_load_woocommerce_element', 99 );

	class WPBakeryShortCode_ninzio_products extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_product_categories extends WPBakeryShortCode {}
}