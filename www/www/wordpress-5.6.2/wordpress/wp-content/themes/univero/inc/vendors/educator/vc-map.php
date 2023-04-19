<?php

if ( function_exists('vc_map') && class_exists('WPBakeryShortCode') ) {
	function univero_educator_get_category_childs( $categories, $id_parent, $level, &$dropdown ) {
        foreach ( $categories as $key => $category ) {
            if ( $category->category_parent == $id_parent ) {
                $dropdown = array_merge( $dropdown, array( str_repeat( "- ", $level ) . $category->name  => $category->slug ) );
                unset($categories[$key]);
                univero_educator_get_category_childs( $categories, $category->term_id, $level + 1, $dropdown );
            }
        }
    }

    function univero_educator_get_categories() {
        $return = array( esc_html__(' --- Choose a Category --- ', 'univero') => '' );

        $args = array(
            'type' => 'post',
            'child_of' => 0,
            'orderby' => 'name',
            'order' => 'ASC',
            'hide_empty' => false,
            'hierarchical' => 1,
            'taxonomy' => 'edr_course_category'
        );

        $categories = get_categories( $args );
        univero_educator_get_category_childs( $categories, 0, 0, $return );

        return $return;
    }

    function univero_educator_get_memberships() {
        $return = array( esc_html__(' --- Choose a membership --- ', 'univero') => '' );

        $args = array(
            'post_type' => 'edr_membership',
            'posts_per_page' => -1,
            'post_status'    => 'publish',
        );

        $posts = get_posts( $args );
        if ( !empty($posts) ) {
        	foreach ($posts as $post) {
        		$return[$post->post_title] = $post->post_name;
        	}
        }

        return $return;
    }

	if ( !function_exists('univero_load_educator_element')) {
		function univero_load_educator_element() {
			vc_map( array(
				'name'        => esc_html__( 'Ninzio Courses','univero'),
				'base'        => 'ninzio_courses',
				"category" => esc_html__('Ninzio Education', 'univero'),
				'description' => esc_html__( 'Create Courses for one Widget', 'univero' ),
				"params"      => array(
					array(
		              	"type" => "textfield",
		              	"heading" => esc_html__("Title", 'univero'),
		              	"param_name" => "title",
		            ),
					array(
						"type" => "dropdown",
						"heading" => esc_html__("Get Course By", 'univero'),
						"param_name" => "course_type",
						'value' 	=> array(
							esc_html__('Lastest Courses', 'univero') => 'most_recent', 
							esc_html__('Random Courses', 'univero') => 'random', 
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
							esc_html__('List', 'univero') => 'list', 
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

			$categories = array();
	        if ( is_admin() ) {
	            $categories = univero_educator_get_categories();
	        }
			vc_map( array(
				'name'        => esc_html__( 'Ninzio Courses Categories','univero'),
				'base'        => 'ninzio_course_categories',
				"category" => esc_html__('Ninzio Education', 'univero'),
				'description' => esc_html__( 'Display course categories in frontend', 'univero' ),
				"params"      => array(
					array(
						'type' => 'param_group',
						'heading' => esc_html__('Categories', 'univero' ),
						'param_name' => 'categories',
						'description' => '',
						'value' => '',
						'params' => array(
							array(
								'type' => 'textfield',
								'heading' => esc_html__( 'Category Name', 'univero' ),
								'param_name' => 'name',
							),
							array(
								'type' => 'textfield',
								'heading' => esc_html__( 'Label', 'univero' ),
								'param_name' => 'label',
							),
							array(
								"type" => "dropdown",
								"heading" => esc_html__("Select Category", 'univero'),
								"param_name" => "category",
								'value' => $categories
							),
							array(
								"type" => "attach_image",
								"heading" => esc_html__("Background Image", 'univero'),
								"param_name" => "image",
							),
				            array(
								"type" => "textfield",
								"heading" => esc_html__("Category Icon Font", 'univero'),
								"param_name" => "icon_font",
							),
							array(
								"type" => "attach_image",
								"heading" => esc_html__("Category Icon Image", 'univero'),
								"param_name" => "icon_image",
								"description" => esc_html__('If you upload an image, icon will not show.', 'univero')
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

			vc_map( array(
				'name'        => esc_html__( 'Ninzio Courses Lecturers','univero'),
				'base'        => 'ninzio_course_lecturer',
				"category" => esc_html__('Ninzio Education', 'univero'),
				'description' => esc_html__( 'Display course lecturers in frontend', 'univero' ),
				"params"      => array(
					array(
						'type' => 'textfield',
						'heading' => esc_html__( 'Number', 'univero' ),
						'param_name' => 'number',
						"admin_label" => true,
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
			$memberships = array();
			if ( is_admin() ) {
				$memberships = univero_educator_get_memberships();
			}
			vc_map( array(
				'name'        => esc_html__( 'Ninzio Course Membership','univero'),
				'base'        => 'ninzio_course_membership',
				"category" => esc_html__('Ninzio Education', 'univero'),
				'description' => esc_html__( 'Display course membership in frontend', 'univero' ),
				"params"      => array(
					array(
						"type" => "dropdown",
						"heading" => esc_html__("Choose a membership", 'univero'),
						"param_name" => "post_name",
						'value' 	=> $memberships
					),
					array(
						'type' => 'checkbox',
						'heading' => esc_html__( 'Recommend this membership', 'univero' ),
						'param_name' => 'recommend',
						'value' => array( esc_html__( 'Yes, to show recommend', 'univero' ) => 'yes' ),
					),
					array(
						'type' => 'textfield',
						'heading' => esc_html__( 'Extra class name', 'univero' ),
						'param_name' => 'el_class',
						'description' => esc_html__( 'If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.', 'univero' )
					)
				),
			));

			vc_map( array(
				'name'        => esc_html__( 'Ninzio Courses Search Form','univero'),
				'base'        => 'ninzio_course_search',
				"category" => esc_html__('Ninzio Education', 'univero'),
				'description' => esc_html__( 'Display course search form in frontend', 'univero' ),
				"params"      => array(
					array(
						'type' => 'checkbox',
						'heading' => esc_html__( 'Show Search Title Field', 'univero' ),
						'param_name' => 'show_title',
						'value' => array( esc_html__( 'Yes, to show Search Title Field', 'univero' ) => 'yes' ),
					),
					array(
						'type' => 'checkbox',
						'heading' => esc_html__( 'Show Search Category Field', 'univero' ),
						'param_name' => 'show_category',
						'value' => array( esc_html__( 'Yes, to show Search Category Field', 'univero' ) => 'yes' ),
					),
					array(
						'type' => 'checkbox',
						'heading' => esc_html__( 'Show Search Instructor Field', 'univero' ),
						'param_name' => 'show_instructor',
						'value' => array( esc_html__( 'Yes, to show Search Instructor Field', 'univero' ) => 'yes' ),
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
	add_action( 'vc_after_set_mode', 'univero_load_educator_element', 99 );

	class WPBakeryShortCode_ninzio_courses extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_course_categories extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_course_lecturer extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_course_membership extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_course_search extends WPBakeryShortCode {}
}