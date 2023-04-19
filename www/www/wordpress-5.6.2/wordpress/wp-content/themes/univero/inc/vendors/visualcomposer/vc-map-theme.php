<?php
if ( function_exists('vc_map') && class_exists('WPBakeryShortCode') ) {
	if ( !function_exists('univero_load_load_theme_element')) {
		function univero_load_load_theme_element() {
			$columns = array(1,2,3,4,6);
			// Heading Text Block
			vc_map( array(
				'name'        => esc_html__( 'Ninzio Widget Heading','univero'),
				'base'        => 'ninzio_title_heading',
				"class"       => "",
				"category" => esc_html__('Ninzio Elements', 'univero'),
				'description' => esc_html__( 'Create title for one Widget', 'univero' ),
				"params"      => array(
					array(
						'type' => 'textfield',
						'heading' => esc_html__( 'Widget title', 'univero' ),
						'param_name' => 'title',
						'description' => esc_html__( 'Enter heading title.', 'univero' ),
						"admin_label" => true,
					),
					array(
						"type" => "textarea",
						'heading' => esc_html__( 'Description', 'univero' ),
						"param_name" => "descript",
						"value" => '',
						'description' => esc_html__( 'Enter description for title.', 'univero' )
				    ),
					array(
						"type" => "dropdown",
						"heading" => esc_html__("Style", 'univero'),
						"param_name" => "style",
						'value' 	=> array(
							esc_html__('Style Default', 'univero') => 'default', 
							esc_html__('Style 1', 'univero') => 'style1', 
							esc_html__('Style 2', 'univero') => 'style2', 
						),
						'std' => ''
					),
					array(
						'type' => 'textfield',
						'heading' => esc_html__( 'Extra class name', 'univero' ),
						'param_name' => 'el_class',
						'description' => esc_html__( 'If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.', 'univero' )
					)
				),
			));

			// calltoaction
			vc_map( array(
				'name'        => esc_html__( 'Ninzio Call To Action','univero'),
				'base'        => 'ninzio_call_action',
				"class"       => "",
				"category" => esc_html__('Ninzio Elements', 'univero'),
				'description' => esc_html__( 'Create title for one Widget', 'univero' ),
				"params"      => array(
					array(
						"type" => "attach_image",
						"heading" => esc_html__("Image Feature for style Default", 'univero'),
						"param_name" => "image"
					),
					array(
						'type' => 'textfield',
						'heading' => esc_html__( 'Title', 'univero' ),
						'param_name' => 'title',
						'value'       => esc_html__( 'Title', 'univero' ),
						'description' => esc_html__( 'Enter heading title.', 'univero' ),
						"admin_label" => true
					),
					array(
						"type" => "textarea_html",
						'heading' => esc_html__( 'Description', 'univero' ),
						"param_name" => "content",
						"value" => '',
						'description' => esc_html__( 'Enter description for title.', 'univero' )
				    ),

				    array(
						'type' => 'textfield',
						'heading' => esc_html__( 'Text Button 1', 'univero' ),
						'param_name' => 'textbutton1',
						'description' => esc_html__( 'Text Button', 'univero' ),
						"admin_label" => true
					),

					array(
						'type' => 'textfield',
						'heading' => esc_html__( ' Link Button 1', 'univero' ),
						'param_name' => 'linkbutton1',
						'description' => esc_html__( 'Link Button 1', 'univero' ),
						"admin_label" => true
					),
					array(
						"type" => "dropdown",
						"heading" => esc_html__("Button Style", 'univero'),
						"param_name" => "buttons1",
						'value' 	=> array(
							esc_html__('Default ', 'univero') => 'btn-default ', 
							esc_html__('Primary ', 'univero') => 'btn-primary ', 
							esc_html__('Success ', 'univero') => 'btn-success radius-0 ', 
							esc_html__('Info ', 'univero') => 'btn-info ', 
							esc_html__('Warning ', 'univero') => 'btn-warning ', 
							esc_html__('Theme Color ', 'univero') => 'btn-theme',
							esc_html__('Theme Gradient Color ', 'univero') => 'btn-theme btn-gradient',
							esc_html__('Second Color ', 'univero') => 'btn-theme-second',
							esc_html__('Danger ', 'univero') => 'btn-danger ', 
							esc_html__('Pink ', 'univero') => 'btn-pink ', 
							esc_html__('White ', 'univero') => 'btn-white', 
							esc_html__('Primary Outline', 'univero') => 'btn-primary btn-outline', 
							esc_html__('White Outline ', 'univero') => 'btn-white btn-outline ',
							esc_html__('Theme Outline ', 'univero') => 'btn-theme btn-outline ',
						),
						'std' => ''
					),
					array(
						"type" => "dropdown",
						"heading" => esc_html__("Style", 'univero'),
						"param_name" => "style",
						'value' 	=> array(
							esc_html__('Default', 'univero') => 'styledefault',							
							esc_html__('Center White', 'univero') => 'center-white',							
						),
						'std' => ''
					),

					array(
						'type' => 'textfield',
						'heading' => esc_html__( 'Extra class name', 'univero' ),
						'param_name' => 'el_class',
						'description' => esc_html__( 'If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.', 'univero' )
					)
				),
			));
			
			// Ninzio Counter
			vc_map( array(
			    "name" => esc_html__("Ninzio Counter",'univero'),
			    "base" => "ninzio_counter",
			    "description"=> esc_html__('Counting number with your term', 'univero'),
			    "category" => esc_html__('Ninzio Elements', 'univero'),
			    "params" => array(
			    	array(
						'type' => 'param_group',
						'heading' => esc_html__('Counter Settings', 'univero' ),
						'param_name' => 'counters',
						'params' => array(
							array(
				                "type" => "textfield",
				                "heading" => esc_html__('Icon Font', 'univero'),
				                "param_name" => "icon_font",
				            ),
							array(
				                "type" => "textfield",
				                "heading" => esc_html__('Title', 'univero'),
				                "param_name" => "title",
				            ),
				            array(
								"type" => "textfield",
								"heading" => esc_html__("Number", 'univero'),
								"param_name" => "number",
							),
							array(
				                "type" => "textfield",
				                "heading" => esc_html__('Suffix Text', 'univero'),
				                "param_name" => "suffix",
				            ),
				            array(
								"type" => "dropdown",
								"heading" => esc_html__("Style", 'univero'),
								"param_name" => "style",
								'value' 	=> array(									
									esc_html__('Style 1', 'univero') => 'style1', 
									esc_html__('Style 2', 'univero') => 'style2', 
								),
								'std' => ''
							),
							array(
								"type" => "dropdown",
								"heading" => esc_html__("Feature", 'univero'),
								"param_name" => "featured",
								'value' 	=> array(									
									esc_html__('No', 'univero') => 0, 
									esc_html__('Yes', 'univero') => 1, 
								),
								'std' => ''
							),
						),
					),
			    	array(
		                "type" => "dropdown",
		                "heading" => esc_html__('Columns', 'univero'),
		                "param_name" => 'columns',
		                "value" => array(1,2,3,4,6),
		            ),
					array(
						"type" => "textfield",
						"heading" => esc_html__("Extra class name", 'univero'),
						"param_name" => "el_class",
						"description" => esc_html__("If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.", 'univero')
					)
			   	)
			));
			
			// Ninzio Brands
			vc_map( array(
			    "name" => esc_html__("Ninzio Brands",'univero'),
			    "base" => "ninzio_brands",
			    "class" => "",
			    "description"=> esc_html__('Display brands on front end', 'univero'),
			    "category" => esc_html__('Ninzio Elements', 'univero'),
			    "params" => array(
			    	array(
						"type" => "textfield",
						"heading" => esc_html__("Title", 'univero'),
						"param_name" => "title",
						"value" => '',
						"admin_label"	=> true
					),
					array(
						"type" => "textfield",
						"heading" => esc_html__("Number", 'univero'),
						"param_name" => "number",
						"value" => ''
					),
				 	array(
						"type" => "dropdown",
						"heading" => esc_html__("Layout Type", 'univero'),
						"param_name" => "layout_type",
						'value' 	=> array(
							esc_html__('Carousel', 'univero') => 'carousel', 
							esc_html__('Grid', 'univero') => 'grid'
						),
						'std' => ''
					),
					array(
						"type" => "dropdown",
						"heading" => esc_html__("Style Type", 'univero'),
						"param_name" => "style",
						'value' 	=> array(
							esc_html__('Border Dasher', 'univero') => '', 
							esc_html__('Border Solid', 'univero') => 'solid',
							esc_html__('No Border', 'univero') => 'no-border',
						),
						'std' => ''
					),
					array(
		                "type" => "dropdown",
		                "heading" => esc_html__('Columns','univero'),
		                "param_name" => 'columns',
		                "value" => array(1,2,3,4,5,6),
		            ),
		            array(
						"type" => "dropdown",
						"heading" => esc_html__("Rows for style Carousel", 'univero'),
						"param_name" => "rows",
						"value" => array(1,2,3),
					),
					array(
						"type" => "textfield",
						"heading" => esc_html__("Extra class name", 'univero'),
						"param_name" => "el_class",
						"description" => esc_html__("If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.", 'univero')
					)
			   	)
			));
			
			vc_map( array(
			    "name" => esc_html__("Ninzio Socials link",'univero'),
			    "base" => "ninzio_socials_link",
			    "description"=> esc_html__('Show socials link', 'univero'),
			    "category" => esc_html__('Ninzio Elements', 'univero'),
			    "params" => array(
			    	array(
						"type" => "textfield",
						"heading" => esc_html__("Title", 'univero'),
						"param_name" => "title",
						"value" => '',
						"admin_label"	=> true
					),
					array(
						"type" => "textarea",
						"heading" => esc_html__("Description", 'univero'),
						"param_name" => "description",
						"value" => '',
					),
					array(
						"type" => "textfield",
						"heading" => esc_html__("Facebook Page URL", 'univero'),
						"param_name" => "facebook_url",
						"value" => '',
						"admin_label"	=> true
					),
					array(
						"type" => "textfield",
						"heading" => esc_html__("Twitter Page URL", 'univero'),
						"param_name" => "twitter_url",
						"value" => '',
						"admin_label"	=> true
					),
					array(
						"type" => "textfield",
						"heading" => esc_html__("Youtube Page URL", 'univero'),
						"param_name" => "youtube_url",
						"value" => '',
						"admin_label"	=> true
					),
					array(
						"type" => "textfield",
						"heading" => esc_html__("Pinterest Page URL", 'univero'),
						"param_name" => "pinterest_url",
						"value" => '',
						"admin_label"	=> true
					),
					array(
						"type" => "textfield",
						"heading" => esc_html__("Google Plus Page URL", 'univero'),
						"param_name" => "google-plus_url",
						"value" => '',
						"admin_label"	=> true
					),
					array(
						"type" => "textfield",
						"heading" => esc_html__("Instagram Page URL", 'univero'),
						"param_name" => "instagram_url",
						"value" => '',
						"admin_label"	=> true
					),
					array(
						"type" => "dropdown",
						"heading" => esc_html__("Align", 'univero'),
						"param_name" => "align",
						'value' 	=> array(
							esc_html__('Left', 'univero') => '', 
							esc_html__('Right', 'univero') => 'right'
						),
						'std' => ''
					),
					array(
						"type" => "dropdown",
						"heading" => esc_html__("Style", 'univero'),
						"param_name" => "style",
						'value' 	=> array(
							esc_html__('Small', 'univero') => '', 
							esc_html__('Large', 'univero') => 'large'
						),
						'std' => ''
					),
					array(
						"type" => "textfield",
						"heading" => esc_html__("Extra class name", 'univero'),
						"param_name" => "el_class",
						"description" => esc_html__("If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.", 'univero')
					)
			   	)
			));
			// newsletter
			vc_map( array(
			    "name" => esc_html__("Ninzio Newsletter",'univero'),
			    "base" => "ninzio_newsletter",
			    "class" => "",
			    "description"=> esc_html__('Show newsletter form', 'univero'),
			    "category" => esc_html__('Ninzio Elements', 'univero'),
			    "params" => array(
			    	array(
						"type" => "textfield",
						"heading" => esc_html__("Title", 'univero'),
						"param_name" => "title",
						"value" => '',
						"admin_label"	=> true
					),
					array(
						"type" => "textarea",
						"heading" => esc_html__("Description", 'univero'),
						"param_name" => "description",
						"value" => '',
					),
					array(
						"type" => "attach_image",
						"description" => esc_html__("Image Icon.", 'univero'),
						"param_name" => "image_icon",
						"value" => '',
						'heading'	=> esc_html__('Image Icon', 'univero' )
					),
					array(
		                'type' => 'dropdown',
		                'heading' => esc_html__( 'Style', 'univero' ),
		                'param_name' => 'style',
		                'value' => array(
		                    esc_html__( 'Style White', 'univero' ) 	=> 'style1',
		                    esc_html__( 'Style 2', 'univero' ) 	=> 'style2',
		                )
		            ),
					array(
						"type" => "textfield",
						"heading" => esc_html__("Extra class name", 'univero'),
						"param_name" => "el_class",
						"description" => esc_html__("If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.", 'univero')
					)
			   	)
			));
			// google map
			$map_styles = array( esc_html__('Choose a map style', 'univero') => '' );
			if ( is_admin() && class_exists('Univero_Google_Maps_Styles') ) {
				$styles = Univero_Google_Maps_Styles::styles();
				foreach ($styles as $style) {
					$map_styles[$style['title']] = $style['slug'];
				}
			}
			vc_map( array(
			    "name" => esc_html__("Ninzio Google Map",'univero'),
			    "base" => "ninzio_googlemap",
			    "description" => esc_html__('Diplay Google Map', 'univero'),
			    "category" => esc_html__('Ninzio Elements', 'univero'),
			    "params" => array(
			    	array(
						"type" => "textfield",
						"heading" => esc_html__("Title", 'univero'),
						"param_name" => "title",
						"admin_label" => true,
						"value" => '',
					),
					array(
		                "type" => "textarea",
		                "class" => "",
		                "heading" => esc_html__('Description','univero'),
		                "param_name" => "des",
		            ),
		            array(
		                'type' => 'googlemap',
		                'heading' => esc_html__( 'Location', 'univero' ),
		                'param_name' => 'location',
		                'value' => ''
		            ),
		            array(
		                'type' => 'hidden',
		                'heading' => esc_html__( 'Latitude Longitude', 'univero' ),
		                'param_name' => 'lat_lng',
		                'value' => '21.0173222,105.78405279999993'
		            ),
		            array(
						"type" => "textfield",
						"heading" => esc_html__("Map height", 'univero'),
						"param_name" => "height",
						"value" => '',
					),
					array(
						"type" => "textfield",
						"heading" => esc_html__("Map Zoom", 'univero'),
						"param_name" => "zoom",
						"value" => '13',
					),
		            array(
		                'type' => 'dropdown',
		                'heading' => esc_html__( 'Map Type', 'univero' ),
		                'param_name' => 'type',
		                'value' => array(
		                    esc_html__( 'roadmap', 'univero' ) 		=> 'ROADMAP',
		                    esc_html__( 'hybrid', 'univero' ) 	=> 'HYBRID',
		                    esc_html__( 'satellite', 'univero' ) 	=> 'SATELLITE',
		                    esc_html__( 'terrain', 'univero' ) 	=> 'TERRAIN',
		                )
		            ),
		            array(
						"type" => "attach_image",
						"heading" => esc_html__("Custom Marker Icon", 'univero'),
						"param_name" => "marker_icon"
					),
					array(
		                'type' => 'dropdown',
		                'heading' => esc_html__( 'Custom Map Style', 'univero' ),
		                'param_name' => 'map_style',
		                'value' => $map_styles
		            ),
					array(
						"type" => "textfield",
						"heading" => esc_html__("Extra class name", 'univero'),
						"param_name" => "el_class",
						"description" => esc_html__("If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.", 'univero')
					)
			   	)
			));
			// Testimonial
			vc_map( array(
	            "name" => esc_html__("Ninzio Testimonials",'univero'),
	            "base" => "ninzio_testimonials",
	            'description'=> esc_html__('Display Testimonials In FrontEnd', 'univero'),
	            "class" => "",
	            "category" => esc_html__('Ninzio Elements', 'univero'),
	            "params" => array(
	              	array(
						"type" => "textfield",
						"heading" => esc_html__("Title", 'univero'),
						"param_name" => "title",
						"admin_label" => true,
						"value" => '',
					),
	              	array(
		              	"type" => "textfield",
		              	"heading" => esc_html__("Number", 'univero'),
		              	"param_name" => "number",
		              	"value" => '4',
		            ),
		            array(
		              	"type" => "textfield",
		              	"heading" => esc_html__("Columns", 'univero'),
		              	"param_name" => "columns",
		              	"value" => '1',
		            ),
		            array(
		                "type" => "dropdown",
		                "heading" => esc_html__('Layout Type','univero'),
		                "param_name" => 'layout_type',
		                'value' 	=> array(
		                	esc_html__('Style 1 ', 'univero') => 'style1', 
		                	esc_html__('Style 2', 'univero') => 'style2',
		                	esc_html__('Style 3', 'univero') => 'style3',
						),
						'std' => ''
		            ),
		            array(
						"type" => "textfield",
						"heading" => esc_html__("Extra class name", 'univero'),
						"param_name" => "el_class",
						"description" => esc_html__("If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.", 'univero')
					)
	            )
	        ));
	        
	        // Gallery Images
			vc_map( array(
	            "name" => esc_html__("Ninzio Gallery", 'univero'),
	            "base" => "ninzio_gallery",
	            'description'=> esc_html__('Display Gallery In FrontEnd', 'univero'),
	            "category" => esc_html__('Ninzio Elements', 'univero'),
	            "params" => array(
	            	array(
		              	"type" => "textfield",
		              	"heading" => esc_html__("Title", 'univero'),
		              	"param_name" => "title",
		            ),
		            array(
		              	"type" => "textfield",
		              	"heading" => esc_html__("Number Gallery", 'univero'),
		              	"param_name" => "number",
		              	'value' => '8',
		            ),
					array(
		                "type" => "dropdown",
		                "heading" => esc_html__('Columns', 'univero'),
		                "param_name" => 'columns',
		                "value" => $columns
		            ),
		            array(
		                "type" => "dropdown",
		                "heading" => esc_html__('Layout Type', 'univero'),
		                "param_name" => 'layout_type',
		                'value' => array(
							esc_html__('Grid', 'univero') => 'grid',
							esc_html__('Carousel', 'univero') => 'carousel',
							esc_html__('Mansory', 'univero') => 'mansory',
							esc_html__('Mansory Special', 'univero') => 'mansory-special',
						)
		            ),
		            array(
		                "type" => "dropdown",
		                "heading" => esc_html__('Image Style', 'univero'),
		                "param_name" => 'image_style',
		                'value' => array(
							esc_html__('Style 1', 'univero') => 'style1', 
							esc_html__('Style 2', 'univero') => 'style2',
							esc_html__('Style 3 (Footer)', 'univero') => 'style3',
						)
		            ),
		            array(
						'type' => 'textfield',
						'heading' => esc_html__( 'Thumbnail size', 'univero' ),
						'param_name' => 'thumbsize',
						'description' => esc_html__( 'Enter thumbnail size. Example: thumbnail, medium, large, full or other sizes defined by current theme. Alternatively enter image size in pixels: 200x100 (Width x Height) . ', 'univero' ),
						'dependency' => array(
							'element' => 'layout_type',
							'value' => array('mansory', 'grid', 'carousel'),
						),
					),
					array(
						'type' => 'checkbox',
						'heading' => esc_html__( 'Show Categories Filter', 'univero' ),
						'param_name' => 'show_categories_filter',
						'description' => esc_html__( 'Enables to show categories filter.', 'univero' ),
						'value' => array( esc_html__( 'Yes, to show  categories filter', 'univero' ) => 'yes' ),
						'dependency' => array(
							'element' => 'layout_type',
							'value' => array('mansory', 'mansory-special'),
						),
					),
		            array(
						'type' => 'checkbox',
						'heading' => esc_html__( 'Show Load More', 'univero' ),
						'param_name' => 'show_loadmore',
						'description' => esc_html__( 'Enables to show load more.', 'univero' ),
						'value' => array( esc_html__( 'Yes, to show load more', 'univero' ) => 'yes' ),
						'dependency' => array(
							'element' => 'layout_type',
							'value' => array('mansory', 'mansory-special', 'grid'),
						),
					),
		            array(
						"type" => "textfield",
						"heading" => esc_html__("Extra class name", 'univero'),
						"param_name" => "el_class",
						"description" => esc_html__("If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.", 'univero')
					)
	            )
	        ));
	        // Ninzio Video
			vc_map( array(
	            "name" => esc_html__("Ninzio Video", 'univero'),
	            "base" => "ninzio_video",
	            'description'=> esc_html__('Display Video In FrontEnd', 'univero'),
	            "class" => "",
	            "category" => esc_html__('Ninzio Elements', 'univero'),
	            "params" => array(
	              	array(
						"type" => "textfield",
						"heading" => esc_html__("Title", 'univero'),
						"param_name" => "title",
						"admin_label" => true,
						"value" => '',
					),
	              	array(
						"type" => "attach_image",
						"heading" => esc_html__("Image", 'univero'),
						"param_name" => "image"
					),
					array(
		                "type" => "textfield",
		                "heading" => esc_html__('Youtube Video Link','univero'),
		                "param_name" => 'video_link'
		            ),
		            array(
						"type" => "textfield",
						"heading" => esc_html__("Extra class name", 'univero'),
						"param_name" => "el_class",
						"description" => esc_html__("If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.", 'univero')
					)
	            )
	        ));
	        // News
			vc_map( array(
	            "name" => esc_html__("Ninzio News", 'univero'),
	            "base" => "ninzio_news",
	            'description'=> esc_html__('Display News', 'univero'),
	            "class" => "",
	            "category" => esc_html__('Ninzio Elements', 'univero'),
	            "params" => array(
			        // Query
			        array(
						'type' => 'textfield',
						'heading' => esc_html__('Number of items', 'univero'),
						'param_name' => 'items',
						'value' => '3',
			        ),
			        array(
						'type' => 'textfield',
						'heading' => esc_html__('Category Slug (Optional)', 'univero'),
						'param_name' => 'cat_slug',
						'value' => '',
						'description'	=> esc_html__('Displaying posts that have this category. Using category-slug.', 'univero'),
			        ),
			        // Controls
					array(
						'type'       => 'dropdown',
						'heading'    => esc_html__( 'Item: Auto Scroll?', 'univero' ),
						'param_name' => 'auto_scroll',
						'value'      => array(
							'No' => 'false',
							'Yes' => 'true',
						),
						'std'		=> 'false',
					),
					// Columns
					array(
						'type'       => 'dropdown',
						'heading'    => esc_html__( 'Screen > 1000px.', 'univero' ),
						'param_name' => 'column',
						'group'      => esc_html__( 'Columns', 'univero' ),
						'value'      => array(
							'1 Column' => '1c',
							'2 Columns' => '2c',
							'3 Columns' => '3c',
							'4 Columns' => '4c',
							'5 Columns' => '5c',
							'6 Columns' => '6c',
							'7 Columns' => '7c',
						),
						'std'		=> '3c',
					),
					array(
						'type'       => 'dropdown',
						'heading'    => esc_html__( 'Screen from 600px to 1000px.', 'univero' ),
						'param_name' => 'column2',
						'group'      => esc_html__( 'Columns', 'univero' ),
						'value'      => array(
							'1 Column' => '1c',
							'2 Columns' => '2c',
							'3 Columns' => '3c',
							'4 Columns' => '4c',
							'5 Columns' => '5c',
						),
						'std'		=> '2c',
					),
					array(
						'type'       => 'dropdown',
						'heading'    => esc_html__( 'Screen < 600px.', 'univero' ),
						'param_name' => 'column3',
						'group'      => esc_html__( 'Columns', 'univero' ),
						'value'      => array(
							'1 Column' => '1c',
							'2 Columns' => '2c',
							'3 Columns' => '3c',
							'4 Columns' => '4c',
						),
						'std'		=> '1c',
					),
	            )
	        ));
	        // Features Box
			vc_map( array(
	            "name" => esc_html__("Ninzio Features Box",'univero'),
	            "base" => "ninzio_features_box",
	            'description'=> esc_html__('Display Features In FrontEnd', 'univero'),
	            "category" => esc_html__('Ninzio Elements', 'univero'),
	            "params" => array(
	            	
					array(
						'type' => 'param_group',
						'heading' => esc_html__('Members Settings', 'univero' ),
						'param_name' => 'items',
						'description' => '',
						'value' => '',
						'params' => array(
							array(
				                "type" => "textfield",
				                "heading" => esc_html__('Icon Font', 'univero'),
				                "param_name" => "icon_font",
				            ),
							array(
				                "type" => "textfield",
				                "class" => "",
				                "heading" => esc_html__('Title', 'univero'),
				                "param_name" => "title",
				            ),
				            array(
				                "type" => "textarea",
				                "class" => "",
				                "heading" => esc_html__('Description','univero'),
				                "param_name" => "description",
				            ),
				            array(
				                "type" => "dropdown",
				                "heading" => esc_html__('Featured','univero'),
				                "param_name" => 'featured',
				                'value' => array(
									esc_html__('No', 'univero') => 0, 
									esc_html__('Yes', 'univero') => 1,
								),
				            ),
						),
					),
	             	array(
		              	"type" => "textfield",
		              	"heading" => esc_html__("Number Columns", 'univero'),
		              	"param_name" => "number",
		              	'value' => '1',
		            ),
		            array(
		                "type" => "dropdown",
		                "heading" => esc_html__('Layout Type','univero'),
		                "param_name" => 'layout_type',
		                'value' => array(
							esc_html__('Grid', 'univero') => 'grid', 
							esc_html__('List', 'univero') => 'list',
						),
		            ),
		           	array(
		                "type" => "dropdown",
		                "heading" => esc_html__('Style','univero'),
		                "param_name" => 'style',
		                'value' 	=> array(
							esc_html__('Style 1', 'univero') => 'style1', 
							esc_html__('Style 2', 'univero') => 'style2',
							esc_html__('Style 3', 'univero') => 'style3',
						),
						'dependency' => array(
							'element' => 'layout_type',
							'value' => array('grid'),
						),
		            ),
		            array(
						"type" => "textfield",
						"heading" => esc_html__("Extra class name", 'univero'),
						"param_name" => "el_class",
						"description" => esc_html__("If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.", 'univero')
					)
	            )
	        ));
			$custom_menus = array();
			if ( is_admin() ) {
				$menus = get_terms( 'nav_menu', array( 'hide_empty' => false ) );
				if ( is_array( $menus ) && ! empty( $menus ) ) {
					foreach ( $menus as $single_menu ) {
						if ( is_object( $single_menu ) && isset( $single_menu->name, $single_menu->slug ) ) {
							$custom_menus[ $single_menu->name ] = $single_menu->slug;
						}
					}
				}
			}
			// Menu
			vc_map( array(
			    "name" => esc_html__("Ninzio Custom Menu",'univero'),
			    "base" => "ninzio_custom_menu",
			    "class" => "",
			    "description"=> esc_html__('Show Custom Menu', 'univero'),
			    "category" => esc_html__('Ninzio Elements', 'univero'),
			    "params" => array(
			    	array(
						"type" => "textfield",
						"heading" => esc_html__("Title", 'univero'),
						"param_name" => "title",
						"value" => '',
						"admin_label"	=> true
					),
					array(
						'type' => 'dropdown',
						'heading' => esc_html__( 'Menu', 'univero' ),
						'param_name' => 'nav_menu',
						'value' => $custom_menus,
						'description' => empty( $custom_menus ) ? esc_html__( 'Custom menus not found. Please visit Appearance > Menus page to create new menu.', 'univero' ) : esc_html__( 'Select menu to display.', 'univero' ),
						'admin_label' => true,
						'save_always' => true,
					),
					array(
						"type" => "textfield",
						"heading" => esc_html__("Extra class name", 'univero'),
						"param_name" => "el_class",
						"description" => esc_html__("If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.", 'univero')
					)
			   	)
			));

			// Contact Info
			vc_map( array(
	            "name" => esc_html__('Ninzio Contact Info','univero'),
	            "base" => "ninzio_contact_info",
	            'description'=> esc_html__('Display contact info In FrontEnd', 'univero'),
	            "class" => "",
	            "category" => esc_html__('Ninzio Elements', 'univero'),
	            "params" => array(
	            	array(
						"type" => "textfield",
						"heading" => esc_html__("Title", 'univero'),
						"param_name" => "title",
						"admin_label" => true,
						"value" => '',
					),
					array(
						'type' => 'param_group',
						'heading' => esc_html__('Contact Info Settings', 'univero' ),
						'param_name' => 'items',
						'params' => array(
							
							array(
				                "type" => "textfield",
				                "class" => "",
				                "heading" => esc_html__('Title','univero'),
				                "param_name" => "title",
				            ),
				            array(
				                "type" => "textarea",
				                "class" => "",
				                "heading" => esc_html__('Description','univero'),
				                "param_name" => "description",
				            ),
							array(
								"type" => "attach_image",
								"description" => esc_html__("If you upload an image, icon will not show.", 'univero'),
								"param_name" => "image",
								"value" => '',
								'heading'	=> esc_html__('Icon Image', 'univero' )
							),
						),
					),
		            array(
						"type" => "textfield",
						"heading" => esc_html__("Extra class name", 'univero'),
						"param_name" => "el_class",
						"description" => esc_html__("If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.", 'univero')
					)
	            )
	        ));
		}
	}
	add_action( 'vc_after_set_mode', 'univero_load_load_theme_element', 99 );

	class WPBakeryShortCode_ninzio_title_heading extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_call_action extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_brands extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_socials_link extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_newsletter extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_googlemap extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_testimonials extends WPBakeryShortCode {}

	class WPBakeryShortCode_ninzio_counter extends WPBakeryShortCode {
		public function __construct( $settings ) {
			parent::__construct( $settings );
			$this->load_scripts();
		}

		public function load_scripts() {
			wp_register_script('jquery-counterup', get_template_directory_uri().'/js/jquery.counterup.min.js', array('jquery'), false, true);
		}
	}
	class WPBakeryShortCode_ninzio_gallery extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_video extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_features_box extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_custom_menu extends WPBakeryShortCode {}
	class WPBakeryShortCode_ninzio_contact_info extends WPBakeryShortCode {}
}