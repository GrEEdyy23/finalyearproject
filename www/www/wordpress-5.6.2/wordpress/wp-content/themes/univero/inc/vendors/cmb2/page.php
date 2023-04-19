<?php

if ( !function_exists( 'univero_page_metaboxes' ) ) {
	function univero_page_metaboxes(array $metaboxes) {
		global $wp_registered_sidebars;
        $sidebars = array();

        if ( !empty($wp_registered_sidebars) ) {
            foreach ($wp_registered_sidebars as $sidebar) {
                $sidebars[$sidebar['id']] = $sidebar['name'];
            }
        }
        $headers = array_merge( array('global' => esc_html__( 'Global Setting', 'univero' )), univero_get_header_layouts() );
        $footers = array_merge( array('global' => esc_html__( 'Global Setting', 'univero' )), univero_get_footer_layouts() );

		$prefix = 'ninzio_page_';
	    $fields = array(
			array(
				'name' => esc_html__( 'Select Layout', 'univero' ),
				'id'   => $prefix.'layout',
				'type' => 'select',
				'options' => array(
					'main' => esc_html__('Main Content Only', 'univero'),
					'left-main' => esc_html__('Left Sidebar - Main Content', 'univero'),
					'main-right' => esc_html__('Main Content - Right Sidebar', 'univero'),
					'left-main-right' => esc_html__('Left Sidebar - Main Content - Right Sidebar', 'univero')
				)
			),
			array(
                'id' => $prefix.'fullwidth',
                'type' => 'select',
                'name' => esc_html__('Is Full Width?', 'univero'),
                'default' => 'no',
                'options' => array(
                    'no' => esc_html__('No', 'univero'),
                    'yes' => esc_html__('Yes', 'univero')
                )
            ),
            array(
                'id' => $prefix.'left_sidebar',
                'type' => 'select',
                'name' => esc_html__('Left Sidebar', 'univero'),
                'options' => $sidebars
            ),
            array(
                'id' => $prefix.'right_sidebar',
                'type' => 'select',
                'name' => esc_html__('Right Sidebar', 'univero'),
                'options' => $sidebars
            ),
            array(
                'id' => $prefix.'show_breadcrumb',
                'type' => 'select',
                'name' => esc_html__('Show Breadcrumb?', 'univero'),
                'options' => array(
                    'no' => esc_html__('No', 'univero'),
                    'yes' => esc_html__('Yes', 'univero')
                ),
                'default' => 'yes',
            ),
            array(
                'id' => $prefix.'breadcrumb_title',
                'type' => 'text',
                'name' => esc_html__('Breadcrumb Title', 'univero'),
            ),
            array(
                'id' => $prefix.'breadcrumb_layout',
                'type' => 'select',
                'name' => esc_html__('Breadcrumb Layout', 'univero'),
                'options' => array(
                    'layout1' => esc_html__('Layout 1', 'univero'),
                    'layout2' => esc_html__('Layout 2', 'univero'),
                ),
                'default' => 'layout1'
            ),
            array(
                'id' => $prefix.'breadcrumb_color',
                'type' => 'colorpicker',
                'name' => esc_html__('Breadcrumb Background Color', 'univero')
            ),
            array(
                'id' => $prefix.'breadcrumb_image',
                'type' => 'file',
                'name' => esc_html__('Breadcrumb Background Image', 'univero')
            ),
            array(
                'id' => $prefix.'header_type',
                'type' => 'select',
                'name' => esc_html__('Header Layout Type', 'univero'),
                'description' => esc_html__('Choose a header for your website.', 'univero'),
                'options' => $headers,
                'default' => 'global'
            ),
            array(
                'id' => $prefix.'footer_type',
                'type' => 'select',
                'name' => esc_html__('Footer Layout Type', 'univero'),
                'description' => esc_html__('Choose a footer for your website.', 'univero'),
                'options' => $footers,
                'default' => 'global'
            ),
            array(
                'id' => $prefix.'extra_class',
                'type' => 'text',
                'name' => esc_html__('Extra Class', 'univero'),
                'description' => esc_html__('If you wish to style particular content element differently, then use this field to add a class name and then refer to it in your css file.', 'univero')
            )
    	);
		
	    $metaboxes[$prefix . 'display_setting'] = array(
			'id'                        => $prefix . 'display_setting',
			'title'                     => esc_html__( 'Display Settings', 'univero' ),
			'object_types'              => array( 'page' ),
			'context'                   => 'normal',
			'priority'                  => 'high',
			'show_names'                => true,
			'fields'                    => $fields
		);

	    return $metaboxes;
	}
}
add_filter( 'cmb2_meta_boxes', 'univero_page_metaboxes' );

if ( !function_exists( 'univero_cmb2_style' ) ) {
	function univero_cmb2_style() {
		wp_enqueue_style( 'univero-cmb2-style', get_template_directory_uri() . '/inc/vendors/cmb2/assets/style.css', array(), '1.0' );
	}
}
add_action( 'admin_enqueue_scripts', 'univero_cmb2_style' );


