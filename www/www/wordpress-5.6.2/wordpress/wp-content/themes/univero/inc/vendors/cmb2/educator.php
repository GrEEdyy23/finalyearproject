<?php

if ( !function_exists( 'univero_educator_metaboxes' ) ) {
	function univero_educator_metaboxes(array $metaboxes) {
		if ( !defined('EDR_PT_COURSE') ) {
			return;
		}
		$prefix = 'ninzio_educator_';
	    $fields = array(
	    	array(
				'name' => esc_html__( 'Course Location', 'univero' ),
				'id'   => $prefix.'location',
				'type' => 'text',
			),
	    	array(
				'name' => esc_html__( 'Start Course', 'univero' ),
				'id'   => $prefix.'startcourse',
				'type' => 'text_date',
			),
	    	array(
				'name' => esc_html__( 'Course Duration', 'univero' ),
				'id'   => $prefix.'duration',
				'type' => 'text',
				'description' => esc_html__( 'Enter duration time', 'univero' ),
			),
			array(
			    'name' => esc_html__( 'Language', 'univero' ),
			    'id'   => $prefix.'langauge',
			    'type' => 'text'
			),
			array(
				'name' => esc_html__( 'Course Capacity', 'univero' ),
				'id'   => $prefix.'capacity',
				'type' => 'text',
				'default' => 50
			),
			array(
				'name' => esc_html__( 'Certificate', 'univero' ),
				'id'   => $prefix.'certificate',
				'type' => 'select',
			    'options' => array(
			        0 => esc_html__( 'No', 'univero' ),
			        1 => esc_html__( 'Yes', 'univero' )
			    ),
			)
    	);
	    $metaboxes[$prefix . 'display_setting'] = array(
			'id'                        => $prefix . 'display_setting',
			'title'                     => esc_html__( 'More Information', 'univero' ),
			'object_types'              => array( EDR_PT_COURSE ),
			'context'                   => 'normal',
			'priority'                  => 'low',
			'show_names'                => true,
			'fields'                    => $fields
		);

	    $metaboxes[ $prefix . 'display_gallery' ] = array(
			'id'                        => $prefix . 'info',
			'title'                     => esc_html__( 'Gallery Images', 'univero' ),
			'object_types'              => array( EDR_PT_COURSE ),
			'context'                   => 'side',
			'priority'                  => 'low',
			'show_names'                => true,
			'fields'                    => array(
			   	array(
				    'name' => '',
				    'id'   => $prefix . 'gallery_images',
				    'type' => 'file_list',
				    'text' => array(
				        'add_upload_files_text' => esc_html__('Add Images', 'univero'),
				        'remove_image_text' => esc_html__('Remove Image', 'univero'),
				        'file_text' => esc_html__('File:', 'univero'),
				        'file_download_text' => esc_html__('Download', 'univero'),
				        'remove_text' => esc_html__('Remove', 'univero'),
				    ),
				)
			)
		);
		
	    $prefix = 'ninzio_lesson_';
	    $fields = array(
	    	array(
				'name' => esc_html__( 'Lesson Duration', 'univero' ),
				'id'   => $prefix.'duration',
				'type' => 'text',
				'description' => esc_html__( 'Enter duration time', 'univero' ),
			),
    	);
	    $metaboxes[$prefix . 'display_setting'] = array(
			'id'                        => $prefix . 'display_setting',
			'title'                     => esc_html__( 'Lesson Options', 'univero' ),
			'object_types'              => array( EDR_PT_LESSON ),
			'context'                   => 'normal',
			'priority'                  => 'low',
			'show_names'                => true,
			'fields'                    => $fields
		);

		
	    return $metaboxes;
	}
}
add_filter( 'cmb2_meta_boxes', 'univero_educator_metaboxes' );
