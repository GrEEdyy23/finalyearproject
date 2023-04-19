<?php

if ( ! function_exists( 'univero_post_tags' ) ) {
	function univero_post_tags() {
		$posttags = get_the_tags();
		if ( $posttags ) {
			echo '<span class="entry-tags-list"><strong>'.esc_html__( 'Tags: ' , 'univero' ).'</strong> ';
			$i = 1;
			$size = count( $posttags );
			foreach ( $posttags as $tag ) {
				echo '<a href="' . get_tag_link( $tag->term_id ) . '">';
				echo esc_attr($tag->name);
				echo '</a>';
				echo trim($i == ($size) ? '' :'&nbsp;');
				$i ++;
			}
			echo '</span>';
		}
	}
}

if ( ! function_exists( 'univero_post_format_link_helper' ) ) {
	function univero_post_format_link_helper( $content = null, $title = null, $post = null ) {
		if ( ! $content ) {
			$post = get_post( $post );
			$title = $post->post_title;
			$content = $post->post_content;
		}
		$link = univero_get_first_url_from_string( $content );
		if ( ! empty( $link ) ) {
			$title = '<a href="' . esc_url( $link ) . '" rel="bookmark">' . $title . '</a>';
			$content = str_replace( $link, '', $content );
		} else {
			$pattern = '/^\<a[^>](.*?)>(.*?)<\/a>/i';
			preg_match( $pattern, $content, $link );
			if ( ! empty( $link[0] ) && ! empty( $link[2] ) ) {
				$title = $link[0];
				$content = str_replace( $link[0], '', $content );
			} elseif ( ! empty( $link[0] ) && ! empty( $link[1] ) ) {
				$atts = shortcode_parse_atts( $link[1] );
				$target = ( ! empty( $atts['target'] ) ) ? $atts['target'] : '_self';
				$title = ( ! empty( $atts['title'] ) ) ? $atts['title'] : $title;
				$title = '<a href="' . esc_url( $atts['href'] ) . '" rel="bookmark" target="' . $target . '">' . $title . '</a>';
				$content = str_replace( $link[0], '', $content );
			} else {
				$title = '<a href="' . esc_url( get_permalink() ) . '" rel="bookmark">' . $title . '</a>';
			}
		}
		$out['title'] = '<h2 class="entry-title">' . $title . '</h2>';
		$out['content'] = $content;

		return $out;
	}
}

if ( !function_exists('univero_get_page_title') ) {
	function univero_get_page_title() {
		$title = '';
		if ( !is_front_page() || is_paged() ) {
			global $post;
			$homeLink = esc_url( home_url() );

			if ( is_home() ) {
				$title = univero_get_config('blogs_breadcrumb_title', 'Blogs');
			} elseif (is_category()) {
				$title = univero_get_config('blogs_breadcrumb_title', 'Blogs');
			} elseif (is_day()) {
				$title = get_the_time('d');
			} elseif (is_month()) {
				$title = get_the_time('F');
			} elseif (is_year()) {
				$title = get_the_time('Y');
			} elseif (is_single() && !is_attachment()) {
				if ( get_post_type() == 'post' ) {
					$title = univero_get_config('blog_breadcrumb_title', 'Single Post');
				} elseif( get_post_type() == 'tribe_events' ) {
					$title = univero_get_config('event_breadcrumb_title', 'Single Event');
				} elseif( defined('EDR_PT_COURSE') && get_post_type() == EDR_PT_COURSE ) {
					$title = univero_get_config('course_breadcrumb_title', 'Single Course');
				} else {
					$title = get_the_title();
				}
			} elseif ( !is_single() && !is_page() && get_post_type() != 'post' && !is_404() && !is_author() && !is_search() ) {
				$post_type = get_post_type_object(get_post_type());
				
				if (is_object($post_type)) {
					if ( $post_type->name == 'post' ) {
						$title = univero_get_config('blogs_breadcrumb_title', 'Blogs');
					} elseif ( defined('EDR_PT_COURSE') && $post_type->name == EDR_PT_COURSE ) {
						$title = univero_get_config('courses_breadcrumb_title', 'Course');
					} elseif ( $post_type->name == 'tribe_events' ) {
						$title = univero_get_config('events_breadcrumb_title', 'Event');
					} elseif( get_post_type() == 'page' ) {
						global $wp_query;
						$term =	$wp_query->queried_object;

						if ( !empty($term) && ( (!empty($term->taxonomy) && $term->taxonomy == 'tribe_events_cat') ||  (!empty($term->name) && $term->name == 'tribe_events') ) ) {
							$title = univero_get_config('events_breadcrumb_title', 'Event');
						} else {
							$title = $post_type->labels->singular_name;
						}
					} elseif (is_object($post_type)) {
						$title = $post_type->labels->singular_name;
					} else {
						global $wp_query;
						$term =	$wp_query->queried_object;
						if ( !empty($term->taxonomy) && $term->taxonomy == 'tribe_events_cat' ) {
							$title = univero_get_config('events_breadcrumb_title', 'Event');
						}
					}
				} else {
					$title = esc_html__('Search results', 'univero');
				}
			} elseif (is_attachment()) {
				$title = get_the_title();
			} elseif ( is_page() && !$post->post_parent ) {
				$title =  get_post_meta( $post->ID, 'breadcrumb_title', true );
				if ( empty($title) ) {
					$title = get_the_title();
				}
			} elseif ( is_page() && $post->post_parent ) {
				$title =  get_post_meta( $post->ID, 'breadcrumb_title', true );
				if ( empty($title) ) {
					$title = get_the_title();
				}
			} elseif ( is_search() ) {
				$title = esc_html__('Search results','univero');
			} elseif ( is_tag() ) {
				$title = sprintf(__('Posts tagged "%s"','univero'), get_search_query());
			} elseif ( is_author() ) {
				global $author;
				$userdata = get_userdata($author);
				$title = $userdata->display_name;
			} elseif ( is_404() ) {
				$title = esc_html__('Error 404', 'univero');
			}
		}
		return $title;
	}
}
if ( ! function_exists( 'univero_breadcrumbs' ) ) {
	function univero_breadcrumbs() {

		$delimiter = ' ';
		$home = esc_html__('Home', 'univero');
		$before = '<li class="active">';
		$after = '</li>';
		

		if ( !is_front_page() || is_paged()) {
			global $post;
			$homeLink = esc_url( home_url() );

			
			echo '<div class="breadcrumb"><ol class="list-breadcrumb">';
			echo '<li><a href="' . $homeLink . '">' . $home . '</a> ' . $delimiter . '</li> ';

			if (is_category()) {
				global $wp_query;
				$cat_obj = $wp_query->get_queried_object();
				$thisCat = $cat_obj->term_id;
				$thisCat = get_category($thisCat);
				$parentCat = get_category($thisCat->parent);
				echo '<li>';
				if ($thisCat->parent != 0)
					echo get_category_parents($parentCat, TRUE, '</li><li>');
				echo single_cat_title('', false) . '</li>';
			} elseif (is_day()) {
				echo '<li><a href="' . esc_url( get_year_link(get_the_time('Y')) ) . '">' . get_the_time('Y') . '</a></li> ' . $delimiter . ' ';
				echo '<li><a href="' . esc_url( get_month_link(get_the_time('Y'),get_the_time('m')) ) . '">' . get_the_time('F') . '</a></li> ' . $delimiter . ' ';
				echo wp_kses_post($before) . get_the_time('d') . $after;
			} elseif (is_month()) {
				echo '<li><a href="' . esc_url( get_year_link(get_the_time('Y')) ) . '">' . get_the_time('Y') . '</a></li> ' . $delimiter . ' ';
				echo wp_kses_post($before) . get_the_time('F') . $after;
			} elseif (is_year()) {
				echo wp_kses_post($before) . get_the_time('Y') . $after;
			} elseif (is_single() && !is_attachment()) {
				if ( get_post_type() != 'post' ) {
					$post_type = get_post_type_object(get_post_type());
					$slug = $post_type->rewrite;
					echo '<li><a href="' . $homeLink . '/' . $slug['slug'] . '/">' . $post_type->labels->singular_name . '</a></li> ' . $delimiter . ' ';
					echo wp_kses_post($before) . get_the_title() . $after;
				} else {
					$cat = get_the_category(); $cat = $cat[0];
					echo '<li>'.get_category_parents($cat, TRUE, '</li><li>');
					echo get_the_title() . '</li>';
				}
			} elseif (!is_single() && !is_page() && get_post_type() != 'post' && !is_404() && !is_author() && !is_search()) {
				$post_type = get_post_type_object(get_post_type());
				
				if ( get_post_type() == 'tribe_events' ) {
					echo wp_kses_post($before) . esc_html__( 'Events', 'univero' ) . $after;
				} elseif (is_object($post_type)) {
					echo wp_kses_post($before) . $post_type->labels->singular_name . $after;
				}
			} elseif (is_attachment()) {
				$parent = get_post($post->post_parent);
				$cat = get_the_category($parent->ID);
				echo '<li>';
				if ( !empty($cat) ) {
					$cat = $cat[0];
					echo get_category_parents($cat, TRUE, '</li><li>');
				}
				if ( !empty($parent) ) {
					echo '<a href="' . esc_url( get_permalink($parent) ) . '">' . $parent->post_title . '</a></li><li>';
				}
				echo '<span class="active">'.get_the_title() . $after;
			} elseif ( is_page() && !$post->post_parent ) {
				echo wp_kses_post($before) . get_the_title() . $after;
			} elseif ( is_page() && $post->post_parent ) {
				$parent_id  = $post->post_parent;
				$breadcrumbs = array();
				while ($parent_id) {
					$page = get_page($parent_id);
					$breadcrumbs[] = '<li><a href="' . esc_url( get_permalink($page->ID) ) . '">' . get_the_title($page->ID) . '</a></li>';
					$parent_id  = $page->post_parent;
				}
				$breadcrumbs = array_reverse($breadcrumbs);
				foreach ($breadcrumbs as $crumb) {
					echo wp_kses_post($crumb) . ' ' . $delimiter . ' ';
				}
				echo wp_kses_post($before) . get_the_title() . $after;
			} elseif ( is_search() ) {
				echo wp_kses_post($before) . sprintf(esc_html__('Search results for "%s"','univero'), get_search_query()) . $after;
			} elseif ( is_tag() ) {
				echo wp_kses_post($before) . esc_html__('Posts tagged "', 'univero'). single_tag_title('', false) . '"' . $after;
			} elseif ( is_author() ) {
				global $author;
				$userdata = get_userdata($author);
				echo wp_kses_post($before) . $userdata->display_name . $after;
			} elseif ( is_404() ) {
				echo wp_kses_post($before) . esc_html__('Error 404', 'univero') . $after;
			} elseif (is_home()){
				echo wp_kses_post($before) . esc_html__('Blog', 'univero') . $after;
			}

			echo '</ol></div>';
			
		}
	}
}

if ( ! function_exists( 'univero_render_breadcrumbs' ) ) {
	function univero_render_breadcrumbs() {
		global $post;

		$show = true;
		$style = array();
		$layout = 'layout1';
		if ( is_page() && is_object($post) ) {
			$layout =  get_post_meta( $post->ID, 'breadcrumb_layout', true );
			if ( empty($layout) ) {
				$layout = 'layout1';
			}
			$show = get_post_meta( $post->ID, 'ninzio_page_show_breadcrumb', true );
			if ( $show == 'no' ) {
				return ''; 
			}
			$bgimage = get_post_meta( $post->ID, 'ninzio_page_breadcrumb_image', true );
			$bgcolor = get_post_meta( $post->ID, 'ninzio_page_breadcrumb_color', true );
			$style = array();
			if ( $bgcolor ) {
				$style[] = 'background-color:'.$bgcolor;
			}
			if ( $bgimage ) { 
				$style[] = 'background-image:url(\''.esc_url($bgimage).'\')';
			}

		} elseif ( is_singular('post') || is_category() || is_home() || is_tag() || is_author() || is_day() || is_month() || is_year() || is_search()) {
			$key = 'blogs';
			if ( is_singular('post') ) {
				$key = 'blog';
			}
			$layout = univero_get_config($key.'_breadcrumbs_layout', 'layout1');
			$show = univero_get_config('show_'.$key.'_breadcrumbs', true);
			if ( !$show  ) {
				return ''; 
			}
			$breadcrumb_img = univero_get_config($key.'_breadcrumb_image');
	        $breadcrumb_color = univero_get_config($key.'_breadcrumb_color');
	        $style = array();
	        if ( $breadcrumb_color ) {
	            $style[] = 'background-color:'.$breadcrumb_color;
	        }
	        if ( isset($breadcrumb_img['url']) && !empty($breadcrumb_img['url']) ) {
	            $style[] = 'background-image:url(\''.esc_url($breadcrumb_img['url']).'\')';
	        }
		} elseif ( is_singular('tribe_events') || is_tax('tribe_events_cat') || is_post_type_archive('tribe_events') ) {
			$key = 'events';
			if ( is_singular('tribe_events') ) {
				$key = 'event';
			}
			$layout = univero_get_config($key.'_breadcrumbs_layout', 'layout1');
			$show = univero_get_config('show_'.$key.'_breadcrumbs', true);
			if ( !$show  ) {
				return ''; 
			}
			$breadcrumb_img = univero_get_config($key.'_breadcrumb_image');
	        $breadcrumb_color = univero_get_config($key.'_breadcrumb_color');
	        $style = array();
	        if ( $breadcrumb_color ) {
	            $style[] = 'background-color:'.$breadcrumb_color;
	        }
	        if ( isset($breadcrumb_img['url']) && !empty($breadcrumb_img['url']) ) {
	            $style[] = 'background-image:url(\''.esc_url($breadcrumb_img['url']).'\')';
	        }
		} elseif ( defined('EDR_PT_COURSE') && 	(is_singular(EDR_PT_COURSE) || is_tax('edr_course_category') || is_post_type_archive(EDR_PT_COURSE))  ) {
			$key = 'courses';
			if ( is_singular(EDR_PT_COURSE) ) {
				$key = 'course';
			}
			$layout = univero_get_config($key.'_breadcrumbs_layout', 'layout1');
			$show = univero_get_config('show_'.$key.'_breadcrumbs', true);
			if ( !$show  ) {
				return ''; 
			}
			$breadcrumb_img = univero_get_config($key.'_breadcrumb_image');
	        $breadcrumb_color = univero_get_config($key.'_breadcrumb_color');
	        $style = array();
	        if ( $breadcrumb_color ) {
	            $style[] = 'background-color:'.$breadcrumb_color;
	        }
	        if ( isset($breadcrumb_img['url']) && !empty($breadcrumb_img['url']) ) {
	            $style[] = 'background-image:url(\''.esc_url($breadcrumb_img['url']).'\')';
	        }
		}
		
		$estyle = !empty($style)? ' style="'.implode(";", $style).'"':"";
		$title = univero_get_page_title();
		if ( $layout == 'layout1' ) {
			?>
			<section id="ninzio-breadscrumb" class="ninzio-breadscrumb <?php echo esc_attr($layout); ?>">
				<div class="ninzio-breadscrumb-top" <?php echo trim($estyle); ?>>
					<div class="breadscrumb-title">
						<h2 class="bread-title"><?php echo wp_kses_post($title); ?></h2>
					</div>
				</div>
				<div class="ninzio-breadscrumb-bottom">
					<div class="container">
						<div class="breadscrumb-inner">
							<?php univero_breadcrumbs(); ?>
						</div>
					</div>
				</div>
			</section>
			<?php
		} else {
			?>
			<section id="ninzio-breadscrumb" class="ninzio-breadscrumb <?php echo esc_attr($layout); ?>" <?php echo trim($estyle); ?>>
				<div class="container">
					<div class="wrapper-breads">
						<div class="breadscrumb-inner">
							<div class="breadscrumb-title pull-left">
								<h2 class="bread-title"><?php echo wp_kses_post($title); ?></h2>
							</div>
							<div class="pull-right">
								<?php univero_breadcrumbs(); ?>
							</div>
						</div>
					</div>
				</div>
			</section>
			<?php
		}
	}
}

if ( ! function_exists( 'univero_paging_nav' ) ) {
	function univero_paging_nav() {
		global $wp_query, $wp_rewrite;

		if ( $wp_query->max_num_pages < 2 ) {
			return;
		}

		$paged        = get_query_var( 'paged' ) ? intval( get_query_var( 'paged' ) ) : 1;
		$pagenum_link = html_entity_decode( get_pagenum_link() );
		$query_args   = array();
		$url_parts    = explode( '?', $pagenum_link );

		if ( isset( $url_parts[1] ) ) {
			wp_parse_str( $url_parts[1], $query_args );
		}

		$pagenum_link = remove_query_arg( array_keys( $query_args ), $pagenum_link );
		$pagenum_link = trailingslashit( $pagenum_link ) . '%_%';

		$format  = $wp_rewrite->using_index_permalinks() && ! strpos( $pagenum_link, 'index.php' ) ? 'index.php/' : '';
		$format .= $wp_rewrite->using_permalinks() ? user_trailingslashit( $wp_rewrite->pagination_base . '/%#%', 'paged' ) : '?paged=%#%';

		// Set up paginated links.
		$links = paginate_links( array(
			'base'     => $pagenum_link,
			'format'   => $format,
			'total'    => $wp_query->max_num_pages,
			'current'  => $paged,
			'mid_size' => 1,
			'add_args' => array_map( 'urlencode', $query_args ),
			'prev_text' => esc_html__( '', 'univero' ),
			'next_text' => esc_html__( '', 'univero' ),
		) );

		if ( $links ) :

		?>
		<nav class="navigation paging-navigation" role="navigation">
			<h1 class="screen-reader-text hidden"><?php esc_html_e( 'Posts navigation', 'univero' ); ?></h1>
			<div class="ninzio-pagination">
				<?php echo wp_kses_post($links); ?>
			</div><!-- .pagination -->
		</nav><!-- .navigation -->
		<?php
		endif;
	}
}

if ( ! function_exists( 'univero_post_nav' ) ) {
	function univero_post_nav() {
		// Don't print empty markup if there's nowhere to navigate.
		$previous = ( is_attachment() ) ? get_post( get_post()->post_parent ) : get_adjacent_post( false, '', true );
		$next     = get_adjacent_post( false, '', false );

		if ( ! $next && ! $previous ) {
			return;
		}

		?>
		<nav class="navigation post-navigation" role="navigation">
			<h3 class="screen-reader-text"><?php esc_html_e( 'Post navigation', 'univero' ); ?></h3>
			<div class="nav-links clearfix">
				<?php
				if ( is_attachment() ) :
					previous_post_link( '%link','<div class="col-lg-6"><span class="meta-nav">'. esc_html__('Published In', 'univero').'</span></div>');
				else :
					previous_post_link( '%link','<div class="pull-left"><span class="meta-nav">'. esc_html__('Previous Post', 'univero').'</span></div>' );
					next_post_link( '%link', '<div class="pull-right"><span class="meta-nav">' . esc_html__('Next Post', 'univero').'</span><span></span></div>');
				endif;
				?>
			</div><!-- .nav-links -->
		</nav><!-- .navigation -->
		<?php
	}
}

if ( !function_exists('univero_pagination') ) {
    function univero_pagination($per_page, $total, $max_num_pages = '') {
    	global $wp_query, $wp_rewrite;
        ?>
        <div class="ninzio-pagination">
        	<?php
        	$prev = esc_html__('Previous','univero');
        	$next = esc_html__('Next','univero');
        	$pages = $max_num_pages;
        	$args = array('class'=>'pull-left');

        	$wp_query->query_vars['paged'] > 1 ? $current = $wp_query->query_vars['paged'] : $current = 1;
	        if ( empty($pages) ) {
	            global $wp_query;
	            $pages = $wp_query->max_num_pages;
	            if ( !$pages ) {
	                $pages = 1;
	            }
	        }
	        $pagination = array(
	            'base' => @add_query_arg('paged','%#%'),
	            'format' => '',
	            'total' => $pages,
	            'current' => $current,
	            'prev_text' => $prev,
	            'next_text' => $next,
	            'type' => 'array'
	        );

	        if( $wp_rewrite->using_permalinks() ) {
	            $pagination['base'] = user_trailingslashit( trailingslashit( remove_query_arg( 's', get_pagenum_link( 1 ) ) ) . 'page/%#%/', 'paged' );
	        }
	        
	        if ( isset($_GET['s']) ) {
	            $cq = $_GET['s'];
	            $sq = str_replace(" ", "+", $cq);
	        }
	        
	        if ( !empty($wp_query->query_vars['s']) ) {
	            $pagination['add_args'] = array( 's' => $sq);
	        }
	        $paginations = paginate_links( $pagination );
	        if ( !empty($paginations) ) {
	            echo '<ul class="pagination '.esc_attr( $args["class"] ).'">';
	                foreach ($paginations as $key => $pg) {
	                    echo '<li>'. $pg .'</li>';
	                }
	            echo '</ul>';
	        }
        	?>
            
        </div>
    <?php
    }
}

if ( !function_exists('univero_comment_form') ) {
	function univero_comment_form($arg, $class = ' btn-theme') {
		global $post;
		if ('open' == $post->comment_status) {
			ob_start();
	      	comment_form($arg);
	      	$form = ob_get_clean();
	      	?>
	      	<div class="commentform row reset-button-default">
		    	<div class="col-sm-12">
			    	<?php
			      	echo str_replace('id="submit"','id="submit" class="btn '.$class.'"', $form);
			      	?>
		      	</div>
	      	</div>
	      	<?php
	      }
	}
}

if (!function_exists('univero_list_comment') ) {
	function univero_list_comment($comment, $args, $depth) {
		if ( is_file(get_template_directory().'/comments-list.php') ) {
	        require get_template_directory().'/comments-list.php';
      	}
	}
}

if ( !function_exists( 'univero_print_style_footer' ) ) {
	function univero_print_style_footer() {
    	$footer = univero_get_footer_layout();
    	if ( $footer ) {
    		$args = array(
				'name'        => $footer,
				'post_type'   => 'ninzio_footer',
				'post_status' => 'publish',
				'numberposts' => 1
			);
			$posts = get_posts($args);
			foreach ( $posts as $post ) {
	    		return get_post_meta( $post->ID, '_wpb_shortcodes_custom_css', true );
	 	 	}
    	}
    	return false;
	}
}

function univero_display_footer_builder($footer) {
	global $footer_builder;
	$footer_builder = true;
	$args = array(
		'name'        => $footer,
		'post_type'   => 'ninzio_footer',
		'post_status' => 'publish',
		'numberposts' => 1
	);
	$posts = get_posts($args);

	foreach ( $posts as $post ) {
		$class = get_post_meta( $post->ID, 'ninzio_footer_style_class', true );
		echo '<div class="footer-builder-wrapper '. esc_attr($class. ' '.$post->post_name) .'">';
		echo do_shortcode ($post->post_content);
		echo '</div>';
	}
	$footer_builder = false;
}


function univero_get_blogs_layout_type() {
	$layout = univero_get_config( 'blog_display_mode', 'grid' );
	$layout = !empty($layout) ? $layout : 'grid';
	return $layout;
}

function univero_get_blog_thumbsize() {
	$thumbsize = univero_get_config( 'blog_item_thumbsize', '' );
	return $thumbsize;
}

/*
 * create placeholder
 * var size: array( width, height )
 */
function univero_create_placeholder($size) {
	return "data:image/svg+xml;charset=utf-8,%3Csvg xmlns%3D'http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg' viewBox%3D'0 0 ".$size[0]." ".$size[1]."'%2F%3E";
}

function univero_display_image($img) {
	if ( !empty($img) && isset($img[0]) ) {
		if (univero_get_config('image_lazy_loading')) {
			$placeholder_image = univero_create_placeholder(array($img[1], $img[2]));
			?>
			<div class="image-wrapper">
				<img src="<?php echo esc_url($placeholder_image); ?>" data-src="<?php echo esc_url_raw($img[0]); ?>" alt="<?php esc_attr_e('Image', 'univero'); ?>" class="unveil-image">
			</div>
			<?php
		} else {
			?>
			<div class="image-wrapper">
				<img src="<?php echo esc_url_raw($img[0]); ?>" alt="<?php esc_attr_e('Image', 'univero'); ?>">
			</div>
			<?php
		}
	}
}