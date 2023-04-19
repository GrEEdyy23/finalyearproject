<?php

function univero_educator_get_courses( $course_type, $number = -1, $author = false, $ids = array() ) {
	switch ( $course_type ) {
	    case 'most_recent' : 
	       $args = array( 
	            'posts_per_page' => $number, 
	            'orderby' => 'date', 
	            'order' => 'DESC',
	            'post_type' => EDR_PT_COURSE
	        );
	        break;

	    case 'random' : 
	        $args = array(
	            'post_type' => EDR_PT_COURSE,
	            'posts_per_page' => $number, 
	            'orderby' => 'rand'
	        );
	        break;

	    default : 
	     	$args = array(
	            'post_type' => EDR_PT_COURSE,
	            'posts_per_page' => $number, 
	            'orderby' => 'rand'
	        );
	        break;
	}
	if ( $author ) {
		$args['author'] = $author;
	}
	if ( !empty($ids) ) {
		$args['post__in'] = $ids;
	}
	return new WP_Query( $args );
}

function univero_educator_get_meta($key) {
	$prefix = 'ninzio_educator_';
	return get_post_meta( get_the_ID(), $prefix.$key, true );
}

function univero_educator_get_students_by_course($course_id) {
	$obj = Edr_Entries::get_instance();
	$entries = $obj->get_entries( array('course_id' => $course_id, 'entry_status' => 'inprogress') );
	return $entries;
}

if ( !function_exists('univero_course_content_class') ) {
	function univero_course_content_class( $class ) {
		$page = 'archive';
		if ( is_singular( EDR_PT_COURSE ) ) {
            $page = 'single';
        }
		if ( univero_get_config('course_'.$page.'_fullwidth') ) {
			return 'container-fluid';
		}
		return $class;
	}
}
add_filter( 'univero_course_content_class', 'univero_course_content_class', 1 , 1  );

if ( !function_exists('univero_get_course_layout_configs') ) {
	function univero_get_course_layout_configs() {
		$page = 'archive';
		if ( is_singular( EDR_PT_COURSE ) || is_singular( EDR_PT_LESSON ) || is_singular( EDR_PT_MEMBERSHIP ) ) {
            $page = 'single';
        }
		$left = univero_get_config('course_'.$page.'_left_sidebar');
		$right = univero_get_config('course_'.$page.'_right_sidebar');

		switch ( univero_get_config('course_'.$page.'_layout') ) {
		 	case 'left-main':
		 		$configs['left'] = array( 'sidebar' => $left, 'class' => 'col-md-3 col-sm-12 col-xs-12'  );
		 		$configs['main'] = array( 'class' => 'col-md-9 col-sm-12 col-xs-12 pull-right' );
		 		break;
		 	case 'main-right':
		 		$configs['right'] = array( 'sidebar' => $right,  'class' => 'col-md-3 col-sm-12 col-xs-12 pull-right' ); 
		 		$configs['main'] = array( 'class' => 'col-md-9 col-sm-12 col-xs-12' );
		 		break;
	 		case 'main':
	 			$configs['main'] = array( 'class' => 'col-md-12 col-sm-12 col-xs-12' );
	 			break;
 			case 'left-main-right':
 				$configs['left'] = array( 'sidebar' => $left,  'class' => 'col-md-3 col-sm-12 col-xs-12'  );
		 		$configs['right'] = array( 'sidebar' => $right, 'class' => 'col-md-3 col-sm-12 col-xs-12' ); 
		 		$configs['main'] = array( 'class' => 'col-md-6 col-sm-12 col-xs-12' );
 				break;
		 	default:
		 		$configs['main'] = array( 'class' => 'col-md-12 col-sm-12 col-xs-12' );
		 		break;
		}

		return $configs; 
	}
}

function univero_edr_format_price($formatted, $currency, $price) {
	return '<span class="price-wrapper">'.$formatted.'</span>';
}
add_filter( 'edr_format_price', 'univero_edr_format_price', 10, 3 );

function univero_set_course_views($content) {
	global $post;
	if ( $post->post_type != EDR_PT_COURSE ) {
		return $content;
	}
    $count_key = 'edr_views_count';
    $count = get_post_meta($post->ID, $count_key, true);
    if ($count == '') {
        $count = 0;
        delete_post_meta($post->ID, $count_key);
        add_post_meta($post->ID, $count_key, '0');
    } else {
        $count++;
        $value = sanitize_text_field($count);
        update_post_meta($post->ID, $count_key, $value);
    }
    return $content;
}
//To keep the count accurate, lets get rid of prefetching
remove_action( 'wp_head', 'adjacent_posts_rel_link_wp_head', 10, 0);

add_filter( 'the_content', 'univero_set_course_views' );

function univero_get_course_views() {
	global $post;
	return get_post_meta( $post->ID, 'edr_views_count', true );
}

function univero_educator_search_filter($query) {
    if ($query->is_search && !is_admin() ) {
    	if ( isset($query->query_vars) && isset($query->query_vars['post_type']) && $query->query_vars['post_type'] == EDR_PT_COURSE ) {
    		
    		if (isset($_GET['_difficulty']) && $_GET['_difficulty']) {
    			$meta_query[] = array(
	                'key' => '_edr_difficulty',
	                'value' => $_GET['_difficulty']
	            );
	            $query->set('meta_query', $meta_query);
    		}
    		if (isset($_GET['_lecturer']) && $_GET['_lecturer']) {
    			$query->set('author', $_GET['_lecturer']);
    		}

    		if (isset($_GET['_category']) && $_GET['_category']) {
    			$query->set('tax_query', array(
					array(
						'taxonomy' => 'edr_course_category',
						'field'    => 'id',
						'terms'    => $_GET['_category'],
					),
				));
    		}
    	}
    }
	return $query;
}
add_filter('pre_get_posts', 'univero_educator_search_filter');

function univero_educator_display_lessons( $course_id ) {
	$obj_courses = Edr_Courses::get_instance();
	$syllabus = $obj_courses->get_syllabus( $course_id );

	if ( ! empty( $syllabus ) ) {
		Edr_View::the_template( 'lesson-syllabus', array(
			'syllabus' => $syllabus,
			'lessons'  => $obj_courses->get_syllabus_lessons( $syllabus ),
		) );
	} else {
		Edr_View::the_template( 'lesson-lessons', array(
			'lessons' => $obj_courses->get_course_lessons( $course_id ),
		) );
	}
}

add_filter( 'template_include', 'univero_educator_templates' );
function univero_educator_templates( $template ) {
	$post_type = get_post_type();
	$custom_post_types = array( EDR_PT_COURSE => 'course', EDR_PT_LESSON => 'lesson', EDR_PT_MEMBERSHIP => 'membership' );

	if ( !empty( $custom_post_types[$post_type] ) ) {
		$post_type = $custom_post_types[$post_type];
		if ( is_archive() ) {
			return locate_template( 'archive-'.$post_type.'.php' );
		}

		if ( is_single() ) {
			return locate_template( 'single-'.$post_type.'.php' );
		}
	}

	return $template;
}