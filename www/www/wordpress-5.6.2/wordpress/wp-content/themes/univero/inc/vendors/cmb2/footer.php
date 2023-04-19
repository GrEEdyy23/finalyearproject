<?php

if ( !function_exists( 'univero_footer_metaboxes' ) ) {
	function univero_footer_metaboxes(array $metaboxes) {
		$prefix = 'ninzio_footer_';
	    $fields = array(
			array(
				'name' => esc_html__( 'Footer Style', 'univero' ),
				'id'   => $prefix.'style_class',
				'type' => 'select',
				'options' => array(
					'lighting' => esc_html__('Lighting', 'univero'),
					'dark' => esc_html__('Dark', 'univero')
				)
			),
    	);
		
	    $metaboxes[$prefix . 'display_setting'] = array(
			'id'                        => $prefix . 'display_setting',
			'title'                     => esc_html__( 'Display Settings', 'univero' ),
			'object_types'              => array( 'ninzio_footer' ),
			'context'                   => 'normal',
			'priority'                  => 'high',
			'show_names'                => true,
			'fields'                    => $fields
		);

	    return $metaboxes;
	}
}
add_filter( 'cmb2_meta_boxes', 'univero_footer_metaboxes' );
