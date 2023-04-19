<?php

if ( ! function_exists( 'univero_body_classes' ) ) {
	function univero_body_classes( $classes ) {
		global $post;
		if ( is_page() && is_object($post) ) {
			$class = get_post_meta( $post->ID, 'ninzio_page_extra_class', true );
			if ( !empty($class) ) {
				$classes[] = trim($class);
			}
		}
		if ( univero_get_config('preload') ) {
			$classes[] = 'ninzio-body-loading';
		}
		if ( univero_get_config('image_lazy_loading') ) {
			$classes[] = 'image-lazy-loading';
		}
		return $classes;
	}
	add_filter( 'body_class', 'univero_body_classes' );
}

if ( ! function_exists( 'univero_get_shortcode_regex' ) ) {
	function univero_get_shortcode_regex( $tagregexp = '' ) {
		// WARNING! Do not change this regex without changing do_shortcode_tag() and strip_shortcode_tag()
		// Also, see shortcode_unautop() and shortcode.js.
		return
			'\\['                                // Opening bracket
			. '(\\[?)'                           // 1: Optional second opening bracket for escaping shortcodes: [[tag]]
			. "($tagregexp)"                     // 2: Shortcode name
			. '(?![\\w-])'                       // Not followed by word character or hyphen
			. '('                                // 3: Unroll the loop: Inside the opening shortcode tag
			. '[^\\]\\/]*'                   // Not a closing bracket or forward slash
			. '(?:'
			. '\\/(?!\\])'               // A forward slash not followed by a closing bracket
			. '[^\\]\\/]*'               // Not a closing bracket or forward slash
			. ')*?'
			. ')'
			. '(?:'
			. '(\\/)'                        // 4: Self closing tag ...
			. '\\]'                          // ... and closing bracket
			. '|'
			. '\\]'                          // Closing bracket
			. '(?:'
			. '('                        // 5: Unroll the loop: Optionally, anything between the opening and closing shortcode tags
			. '[^\\[]*+'             // Not an opening bracket
			. '(?:'
			. '\\[(?!\\/\\2\\])' // An opening bracket not followed by the closing shortcode tag
			. '[^\\[]*+'         // Not an opening bracket
			. ')*+'
			. ')'
			. '\\[\\/\\2\\]'             // Closing shortcode tag
			. ')?'
			. ')'
			. '(\\]?)';                          // 6: Optional second closing brocket for escaping shortcodes: [[tag]]
	}
}

if ( ! function_exists( 'univero_tagregexp' ) ) {
	function univero_tagregexp() {
		return apply_filters( 'univero_custom_tagregexp', 'video|audio|playlist|video-playlist|embed|univero_media' );
	}
}

if ( !function_exists('univero_class_container_vc') ) {
	function univero_class_container_vc($class, $isfullwidth, $post_type) {
		global $post;
		$fullwidth = false;
		if ( $post_type == 'ninzio_megamenu' ) {
			$fullwidth = false;
		} elseif ( $post_type == 'ninzio_footer' ) {
			$fullwidth = true;
		} else {
			if ( is_page() ) {
				$fullwidth  = get_post_meta( $post->ID, 'ninzio_page_fullwidth', true );
				if ( $fullwidth == 'no' ) {
					$fullwidth = false;
				} else {
					$fullwidth = true;
				}
			} elseif ( is_woocommerce() ) {
				if ( is_singular('product') ) {
					$fullwidth  = univero_get_config( 'product_single_fullwidth', false );
				} else {
					$fullwidth  = univero_get_config( 'product_archive_fullwidth', false );
				}
			} else {
				if ( is_singular('post') ) {
					$fullwidth  = univero_get_config( 'post_single_fullwidth', false );
				} else {
					$fullwidth  = univero_get_config( 'post_archive_fullwidth', false );
				}
			}
		}

		if ( !$fullwidth || !$isfullwidth ) {
			return 'ninzio-'.$class;
		}
		return $class;
	}
}
add_filter( 'univero_class_container_vc', 'univero_class_container_vc', 1, 3);

if ( !function_exists('univero_get_header_layouts') ) {
	function univero_get_header_layouts() {
		$headers = array();
		$files = glob( get_template_directory() . '/headers/*.php' );
	    if ( !empty( $files ) ) {
	        foreach ( $files as $file ) {
	        	$header = str_replace( '.php', '', basename($file) );
	            $headers[$header] = $header;
	        }
	    }
		return $headers;
	}
}

if ( !function_exists('univero_get_header_layout') ) {
	function univero_get_header_layout() {
		global $post;
		if (is_object($post)) {
			if ( is_page() && is_object($post) && isset($post->ID) ) {
				return univero_page_header_layout();
			}
		}
		return univero_get_config('header_type');
	}
	add_filter( 'univero_get_header_layout', 'univero_get_header_layout' );
}

if ( !function_exists('univero_get_footer_layouts') ) {
	function univero_get_footer_layouts() {
		$footers = array();
		$args = array(
			'posts_per_page'   => -1,
			'offset'           => 0,
			'orderby'          => 'date',
			'order'            => 'DESC',
			'post_type'        => 'ninzio_footer',
			'post_status'      => 'publish',
			'suppress_filters' => true 
		);
		$posts = get_posts( $args );
		foreach ( $posts as $post ) {
			$footers[$post->post_name] = $post->post_title;
		}
		return $footers;
	}
}

if ( !function_exists('univero_get_footer_layout') ) {
	function univero_get_footer_layout() {
		if ( is_page() ) {
			global $post;
			$footer = '';
			if ( is_object($post) && isset($post->ID) ) {
				$footer = get_post_meta( $post->ID, 'ninzio_page_footer_type', true );
				if ( $footer == 'global' ) {
					return univero_get_config('footer_type', '');
				}
			}
			return $footer;
		}
		return univero_get_config('footer_type', '');
	}
	add_filter('univero_get_footer_layout', 'univero_get_footer_layout');
}

if ( !function_exists('univero_blog_content_class') ) {
	function univero_blog_content_class( $class ) {
		$page = 'archive';
		if ( is_singular( 'post' ) ) {
            $page = 'single';
        }
		if ( univero_get_config('blog_'.$page.'_fullwidth') ) {
			return 'container-fluid';
		}
		return $class;
	}
}
add_filter( 'univero_blog_content_class', 'univero_blog_content_class', 1 , 1  );


if ( !function_exists('univero_get_blog_layout_configs') ) {
	function univero_get_blog_layout_configs() {
		$page = 'archive';
		if ( is_singular( 'post' ) ) {
            $page = 'single';
        }
		$left = univero_get_config('blog_'.$page.'_left_sidebar');
		$right = univero_get_config('blog_'.$page.'_right_sidebar');

		switch ( univero_get_config('blog_'.$page.'_layout') ) {
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

if ( !function_exists('univero_page_content_class') ) {
	function univero_page_content_class( $class ) {
		global $post;
		if ( is_object($post) ) {
			$fullwidth = get_post_meta( $post->ID, 'ninzio_page_fullwidth', true );
			if ( !$fullwidth || $fullwidth == 'no' ) {
				return $class;
			}
		}
		return 'container-fluid';
	}
}
add_filter( 'univero_page_content_class', 'univero_page_content_class', 1 , 1  );

if ( !function_exists('univero_get_page_layout_configs') ) {
	function univero_get_page_layout_configs() {
		global $post;
		if ( is_object($post) ) {
			$left = get_post_meta( $post->ID, 'ninzio_page_left_sidebar', true );
			$right = get_post_meta( $post->ID, 'ninzio_page_right_sidebar', true );

			switch ( get_post_meta( $post->ID, 'ninzio_page_layout', true ) ) {
			 	case 'left-main':
			 		$configs['left'] = array( 'sidebar' => $left, 'class' => 'col-md-3 col-sm-3'  );
			 		$configs['main'] = array( 'class' => 'col-md-9 col-sm-9' );
			 		break;
			 	case 'main-right':
			 		$configs['right'] = array( 'sidebar' => $right,  'class' => 'col-md-3 col-sm-3' ); 
			 		$configs['main'] = array( 'class' => 'col-md-9 col-sm-9' );
			 		break;
		 		case 'main':
		 			$configs['main'] = array( 'class' => 'clearfix col-xs-12' );
		 			break;
	 			case 'left-main-right':
	 				$configs['left'] = array( 'sidebar' => $left,  'class' => 'col-md-3 col-sm-3'  );
			 		$configs['right'] = array( 'sidebar' => $right, 'class' => 'col-md-3 col-sm-3' ); 
			 		$configs['main'] = array( 'class' => 'col-md-6 col-sm-6' );
	 				break;
			 	default:
			 		$configs['main'] = array( 'class' => 'col-md-12' );
			 		break;
			}
		} else {
			$configs['main'] = array( 'class' => 'col-md-12' );
		}
		return $configs; 
	}
}

if ( !function_exists('univero_page_header_layout') ) {
	function univero_page_header_layout() {
		global $post;
		$header = get_post_meta( $post->ID, 'ninzio_page_header_type', true );
		if ( $header == 'global' ) {
			return univero_get_config('header_type');
		}
		return $header;
	}
}

if ( ! function_exists( 'univero_get_first_url_from_string' ) ) {
	function univero_get_first_url_from_string( $string ) {
		$pattern = "/^\b(?:(?:https?|ftp):\/\/)[-a-z0-9+&@#\/%?=~_|!:,.;]*[-a-z0-9+&@#\/%=~_|]/i";
		preg_match( $pattern, $string, $link );

		$link_return = ( ! empty( $link[0] ) ) ? $link[0] : false;
		$content = str_replace($link_return, "", $string);
        $content = apply_filters( 'the_content', $content);
        return array( 'link' => $link_return, 'content' => $content );
	}
}

if ( !function_exists( 'univero_get_link_attributes' ) ) {
	function univero_get_link_attributes( $string ) {
		preg_match( '/<a href="(.*?)">/i', $string, $atts );

		return ( ! empty( $atts[1] ) ) ? $atts[1] : '';
	}
}

if ( !function_exists( 'univero_post_media' ) ) {
	function univero_post_media( $content ) {
		$is_video = ( get_post_format() == 'video' ) ? true : false;
		$media = univero_get_first_url_from_string( $content );
		$media = $media['link'];
		if ( ! empty( $media ) ) {
			global $wp_embed;
			$content = do_shortcode( $wp_embed->run_shortcode( '[embed]' . $media . '[/embed]' ) );
		} else {
			$pattern = univero_get_shortcode_regex( univero_tagregexp() );
			preg_match( '/' . $pattern . '/s', $content, $media );
			if ( ! empty( $media[2] ) ) {
				if ( $media[2] == 'embed' ) {
					global $wp_embed;
					$content = do_shortcode( $wp_embed->run_shortcode( $media[0] ) );
				} else {
					$content = do_shortcode( $media[0] );
				}
			}
		}
		if ( ! empty( $media ) ) {
			$output = '<div class="entry-media">';
			$output .= ( $is_video ) ? '<div class="pro-fluid"><div class="pro-fluid-inner">' : '';
			$output .= $content;
			$output .= ( $is_video ) ? '</div></div>' : '';
			$output .= '</div>';

			return $output;
		}

		return false;
	}
}
if ( !function_exists( 'univero_random_key' ) ) {
    function univero_random_key($length = 5) {
        $characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
        $return = '';
        for ($i = 0; $i < $length; $i++) {
            $return .= $characters[rand(0, strlen($characters) - 1)];
        }
        return $return;
    }
}

if ( !function_exists( 'univero_post_gallery' ) ) {
	function univero_post_gallery( $content, $args = array() ) {
		$output = '';
		$defaults = array( 'size' => 'large' );
		$args = wp_parse_args( $args, $defaults );
	    $gallery_filter = univero_gallery_from_content( $content );
	    if (count($gallery_filter['ids']) > 0) {
        	$output .= '<div class="owl-carousel post-gallery-owl" data-smallmedium="1" data-extrasmall="1" data-items="1" data-carousel="owl" data-pagination="true" data-nav="true">';
                foreach($gallery_filter['ids'] as $attach_id) {
                    $output .= '<div class="gallery-item">';
                    $output .= wp_get_attachment_image($attach_id, $args['size'] );
                    $output .= '</div>';
                }
            $output .= '</div>';
        }
        return $output;
	}
}

if (!function_exists('univero_gallery_from_content')) {
    function univero_gallery_from_content($content) {

        $result = array(
            'ids' => array(),
            'filtered_content' => ''
        );

        preg_match('/\[gallery.*ids=.(.*).\]/', $content, $ids);
        if(!empty($ids)) {
            $result['ids'] = explode(",", $ids[1]);
            $content =  str_replace($ids[0], "", $content);
            $result['filtered_content'] = apply_filters( 'the_content', $content);
        }

        return $result;

    }
}

if ( !function_exists('univero_substring') ) {
    function univero_substring($string, $limit, $afterlimit = '[...]') {
        if ( empty($string) ) {
        	return $string;
        }
       	$string = explode(' ', strip_tags( $string ), $limit);

        if (count($string) >= $limit) {
            array_pop($string);
            $string = implode(" ", $string) .' '. $afterlimit;
        } else {
            $string = implode(" ", $string);
        }
        $string = preg_replace('`[[^]]*]`','',$string);
        return strip_shortcodes( $string );
    }
}

if ( !function_exists( 'univero_autocomplete_search' ) ) {
    function univero_autocomplete_search() {

        if ( univero_get_global_config('autocomplete_search') ) {
            wp_register_script( 'univero-autocomplete-js', get_template_directory_uri() . '/js/autocomplete-search-init.js', array('jquery','jquery-ui-autocomplete'), null, true);
            wp_enqueue_script( 'univero-autocomplete-js' );

            add_action( 'wp_ajax_univero_autocomplete_search', 'univero_autocomplete_suggestions' );
            add_action( 'wp_ajax_nopriv_univero_autocomplete_search', 'univero_autocomplete_suggestions' );
        }
    }
}

if ( !function_exists( 'univero_autocomplete_suggestions' ) ) {
    function univero_autocomplete_suggestions() {
        // Query for suggestions
        $args = array( 's' => $_REQUEST['term'] );
        if ( isset($_REQUEST['post_type']) ) {
        	$args['post_type'] = $_REQUEST['post_type'];
        }
        if ( isset($_REQUEST['category']) ) {
        	
        	if ( $args['post_type'] == 'product' ) {
        		$args['product_cat'] = $_REQUEST['category'];
        	} else {
        		$args['category'] = $_REQUEST['category'];
        	}
        }
        if ( !isset($args['post_type']) ) {
        	$args['post_type'] = array( 'post', 'product' );
        }
        $posts = get_posts( $args );
        $suggestions = array();
        $show_image = univero_get_config('show_search_product_image');
        $show_price = univero_get_config('search_type') == 'product' ? univero_get_config('show_search_product_price') : false;
        global $post;
        foreach ($posts as $post): setup_postdata($post);
            
            $suggestion = array();
            $suggestion['label'] = esc_html($post->post_title);
            $suggestion['link'] = get_permalink();
            if ( $show_image && has_post_thumbnail( $post->ID ) ) {
                $image = wp_get_attachment_image_src( get_post_thumbnail_id( $post->ID ), 'thumbnail' );
                $suggestion['image'] = $image[0];
            } else {
                $suggestion['image'] = '';
            }
            if ( $show_price ) {
            	$product = new WC_Product( get_the_ID() );
                $suggestion['price'] = esc_html__('Price', 'univero').' '.$product->get_price_html();
            } else {
                $suggestion['price'] = '';
            }

            $suggestions[]= $suggestion;
        endforeach;
        
        $response = $_GET["callback"] . "(" . json_encode($suggestions) . ")";
        echo wp_kses_post($response);
     
        exit;
    }
}


function univero_next_post_link($output, $format, $link, $post, $adjacent) {
    if (empty($post) || $post->post_type != 'post') {
        return $output;
    }
    $title = get_the_title( $post->ID );
    return '<div class="next-post post-nav">
	        <a class="before-hover" href="'.esc_url(get_permalink($post->ID)).'" title="'.esc_attr($title).'">
	            '.esc_html__('Next', 'univero').'<i class="univero-arrow-right"></i>'.'
	        </a>
	        <div class="on-hover">
	        	<h3><a class="nav-post-title" href="'.esc_url(get_permalink($post->ID)).'">'.$title.'</a></h3>
	        	<div class="col-xs-6 hidden">
			        <a href="'.esc_url(get_permalink($post->ID)).'" title="'.esc_attr($title).'">
			            '.get_the_post_thumbnail( $post->ID, 'thumbnail' ).'
			        </a>
		        </div>
		        <div class="col-xs-6 hidden">
			        <span class="date">'.get_the_time( 'M d , Y', $post->ID ).'</span>
		        </div>
	        </div>
        </div>';
    
}

add_filter( 'next_post_link', 'univero_next_post_link', 100, 5 );

function univero_previous_post_link($output, $format, $link, $post, $adjacent) {
    if (empty($post) || $post->post_type != 'post') {
        return $output;
    }
    $title = get_the_title( $post->ID );
    return '<div class="previous-post post-nav">
	        <a class="before-hover" href="'.esc_url(get_permalink($post->ID)).'" title="'.esc_attr($title).'">
	            <i class="univero-arrow-left"></i>'.esc_html__('Previous', 'univero').'
	        </a>
	        <div class="on-hover">
	        	<h3><a class="nav-post-title" href="'.esc_url(get_permalink($post->ID)).'">'.$title.'</a></h3>
	        	<div class="col-xs-12 hidden">
			        <span class="date">'.get_the_time( 'M d , Y', $post->ID ).'</span>
		        </div>
	        	<div class="col-xs-12 hidden">
			        <a href="'.esc_url(get_permalink($post->ID)).'" title="'.esc_attr($title).'">
			            '.get_the_post_thumbnail( $post->ID, 'thumbnail' ).'
			        </a>
		        </div>
		        
	        </div>
        </div>';
    
}
add_filter( 'previous_post_link', 'univero_previous_post_link', 100, 5 );


function univero_get_ajax_galleries() {
    $columns = isset($_POST['columns']) ? $_POST['columns'] : 4;
    $number = isset($_POST['number']) ? $_POST['number'] : 4;
    $page = isset($_POST['page']) ? $_POST['page'] : 1;
    $image_style = isset($_POST['image_style']) ? $_POST['image_style'] : 'style1';
    $layout_type = isset($_POST['layout_type']) ? $_POST['layout_type'] : 'grid';
    $thumbsize = isset($_POST['thumbsize']) ? $_POST['thumbsize'] : '';
    set_query_var( 'thumbsize', $thumbsize );
    $bcol = 12/$columns;

    $args = array(
		'post_type' => 'ninzio_gallery',
		'posts_per_page' => $number,
		'paged' => $page
	);
	$loop = new WP_Query( $args );
    if ( $loop->have_posts()) {
    	if ( $layout_type == 'grid' ) {
	        while ( $loop->have_posts() ) : $loop->the_post(); ?>
    			<?php
    				$img_url = get_the_post_thumbnail_url(get_the_ID(), 'full');
    			?>
    			<?php if ( !empty($img_url) ): ?>
					<div class="col-md-<?php echo esc_attr($bcol); ?>">
						<?php get_template_part('template-parts/gallery-item/'.$image_style); ?>
            		</div>
        		<?php endif; ?>
    		<?php endwhile;
		} elseif ( $layout_type == 'mansory-special' ) {
			$i = 1; $j = 0; while ( $loop->have_posts() ) : $loop->the_post(); ?>
				<?php
					
					$img_url = get_the_post_thumbnail_url(get_the_ID(), 'full');
					$terms = get_the_terms( get_the_ID(), 'ninzio_gallery_category' );
					$categories = '';
					if ( ! empty( $terms ) && ! is_wp_error( $terms ) ) {
						foreach ($terms as $term) {
							$categories .= $term->slug.' ';
						}
					}
				?>
				<?php if ( !empty($img_url) ): ?>
					<?php
						if ( $i > 8 ) {
	    					$j = 0;
	    				}
	    				if ( in_array($j, array(1,3,6,7)) ) {
	    					set_query_var( 'thumbsize', 'univero-gallery-thumb1' );
	    				} else {
	    					set_query_var( 'thumbsize', 'univero-gallery-thumb2' );
	    				}

	    				$i++; $j++;
					?>
					<div class="isotope-item col-md-<?php echo esc_attr($bcol); ?> col-sm-6 col-xs-12 <?php echo esc_attr($categories); ?>">
						<?php get_template_part('template-parts/gallery-item/'.$image_style); ?>
	            	</div>
	            <?php endif; ?>
			<?php endwhile;
		} else {
			while ( $loop->have_posts() ) : $loop->the_post(); ?>
				<?php
					$img_url = get_the_post_thumbnail_url(get_the_ID(), 'full');
					$terms = get_the_terms( get_the_ID(), 'ninzio_gallery_category' );
					$categories = '';
					if ( ! empty( $terms ) && ! is_wp_error( $terms ) ) {
						foreach ($terms as $term) {
							$categories .= $term->slug.' ';
						}
					}
				?>
				<?php if ( !empty($img_url) ): ?>
					<div class="isotope-item col-md-<?php echo esc_attr($bcol); ?> col-sm-6 col-xs-12 <?php echo esc_attr($categories); ?>">
						<?php get_template_part('template-parts/gallery-item/'.$image_style); ?>
	            	</div>
	            <?php endif; ?>
			<?php endwhile;
		}
		wp_reset_postdata();
    }
    exit();
}
add_action( 'wp_ajax_univero_get_ajax_galleries', 'univero_get_ajax_galleries' );
add_action( 'wp_ajax_nopriv_univero_get_ajax_galleries', 'univero_get_ajax_galleries' );


function univero_set_post_views($postID) {
    $count_key = '_views_count';
    $count = get_post_meta($postID, $count_key, true);
    if ($count == '') {
        $count = 0;
        delete_post_meta($postID, $count_key);
        add_post_meta($postID, $count_key, '0');
    } else {
        $count++;
        update_post_meta($postID, $count_key, $count);
    }
}

function univero_track_post_views ($post_id) {
    if ( is_singular('post') || is_singular('tribe_events') || (defined('EDR_PT_COURSE') && is_singular(EDR_PT_COURSE)) ) {
	    if ( empty ( $post_id) ) {
	        global $post;
	        if ( is_object($post) ) {
		        $post_id = $post->ID;    
		    }
	    }
	    if ( $post_id ) {
		    univero_set_post_views($post_id);
		}
	}
}
add_action( 'wp_head', 'univero_track_post_views');

function univero_reg_widget($class_name) {
	$funcs = array('register', 'widget');
	call_user_func(implode('_', $funcs), $class_name);
}