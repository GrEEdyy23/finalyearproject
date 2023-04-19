<?php
$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );

$args = array(
	'post_type' => 'ninzio_gallery',
	'posts_per_page' => $number,
);
$loop = new WP_Query( $args );

if ( $loop->have_posts() ):
	$_id = univero_random_key();
	$bcol = floor(12/$columns);
	set_query_var( 'thumbsize', $thumbsize );
?>	
	<div id="widget-gallery-<?php echo esc_attr($_id); ?>" class="widget widget-gallery <?php echo esc_attr($el_class.' '.$layout_type);?>">
	    <?php if ($title!=''): ?>
	        <h3 class="widget-title">
	            <span><?php echo esc_attr( $title ); ?></span>
	        </h3>
	    <?php endif; ?>

	    <div class="widget-content">
	    	<?php if ( $layout_type == 'mansory' && $show_categories_filter ) { ?>
	    		<?php
	    			$terms = get_terms( 'ninzio_gallery_category', array(
					    'hide_empty' => false,
					) );
					if ( ! empty( $terms ) && ! is_wp_error( $terms ) ) {
	    		?>
						<ul class="isotope-filter list-tab-v1 list-unstyled list-inline" data-related-grid="isotope-items-<?php echo esc_attr($_id); ?>">
							<li><a href="#" class="hvr-fade active" data-filter="*"><?php esc_html_e('Show All', 'univero'); ?></a></li>
					      	<?php foreach ($terms as $term) { ?>
					      		<li><a href="#" class="hvr-fade" data-filter=".<?php echo esc_attr($term->slug); ?>"><?php echo wp_kses_post($term->name); ?></a></li>
					      	<?php } ?>
					    </ul>
			    	<?php } ?>
			<?php } ?>

			<div class="gallery-content clearfix">
		    	<?php if ( $layout_type == 'grid' ) { ?>
		    		<div class="row">
			    		<?php while ( $loop->have_posts() ) : $loop->the_post(); ?>
			    			<?php
			    				$img_url = get_the_post_thumbnail_url(get_the_ID(), 'full');
			    			?>
			    			<?php if ( !empty($img_url) ): ?>
								<div class="col-lg-<?php echo esc_attr($bcol); ?> col-sm-4 col-md-4 col-xs-12 mobile-col-3">
									<?php get_template_part('template-parts/gallery-item/'.$image_style); ?>
	                    		</div>
	                		<?php endif; ?>
			    		<?php endwhile; ?>
					</div>
				<?php } elseif ( $layout_type == 'carousel' ) { ?>
		    		<div class="owl-carousel main-gallery" data-items="1" data-carousel="owl" data-smallmedium="1" data-extrasmall="1" data-pagination="false" data-nav="false">
			    		<?php while ( $loop->have_posts() ) : $loop->the_post(); ?>
			    			<?php
			    				$img_url = get_the_post_thumbnail_url(get_the_ID(), 'full');
			    			?>
			    			<?php if ( !empty($img_url) ): ?>
								<?php get_template_part('template-parts/gallery-item/'.$image_style); ?>
	                		<?php endif; ?>
			    		<?php endwhile; ?>
					</div>
					<div class="owl-carousel thumbs-gallery" data-items="4" data-carousel="owl" data-smallmedium="4" data-extrasmall="4" data-pagination="false" data-nav="false">
			    		<?php while ( $loop->have_posts() ) : $loop->the_post(); ?>
			    			<?php $img_url = get_the_post_thumbnail_url(get_the_ID(), 'medium'); ?>
			    			<?php if ( !empty($img_url) ): ?>
								<figure class="gallery-thumb-image">
							    	<?php univero_display_post_image('medium'); ?>   
								</figure>
	                		<?php endif; ?>
			    		<?php endwhile; ?>
					</div>
				<?php } elseif ( $layout_type == 'mansory-special' ) { ?>
					<div class="row">
						<?php
						    wp_enqueue_script( 'univero-isotope-js',get_template_directory_uri() . '/js/isotope.pkgd.js', array( 'jquery' ) );
						?>
						<div id="isotope-items-<?php echo esc_attr($_id); ?>" class="isotope-items" data-isotope-duration="400">
				    		<?php $i=1; $j = 0; while ( $loop->have_posts() ) : $loop->the_post(); ?>
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
									<div class="isotope-item col-md-<?php echo esc_attr($bcol); ?> col-sm-6 col-xs-12 <?php echo esc_attr($categories); ?> <?php echo esc_attr($i); ?>">
										<?php get_template_part('template-parts/gallery-item/'.$image_style); ?>
			                    	</div>
				                <?php endif; ?>
				    		<?php endwhile; ?>
				    	</div>
					</div>
				<?php } else { ?>
					<div class="row">
						<?php
						    wp_enqueue_script( 'univero-isotope-js',get_template_directory_uri() . '/js/isotope.pkgd.js', array( 'jquery' ) );
						?>
						<div id="isotope-items-<?php echo esc_attr($_id); ?>" class="isotope-items" data-isotope-duration="400">
				    		<?php while ( $loop->have_posts() ) : $loop->the_post(); ?>
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
				    		<?php endwhile; ?>
				    	</div>
					</div>
				<?php } ?>
			</div>
			<?php if ( $show_loadmore ) { $max_pages = $loop->max_num_pages; ?>
				<p class="gallery-button">
					<a href="javascript:void(0);" class="gallery-showmore-btn btn btn-outline <?php echo esc_attr($max_pages <= 1 ? ' hidden' : ''); ?>" data-page="1" data-max-page="<?php echo esc_attr($max_pages); ?>" data-number="<?php echo esc_attr($number); ?>" data-image_style="<?php echo esc_attr($image_style); ?>" data-columns="<?php echo esc_attr($columns); ?>" data-id="widget-gallery-<?php echo esc_attr($_id); ?>" data-layout_type="<?php echo esc_attr($layout_type); ?>" data-thumbsize="<?php echo esc_attr($thumbsize); ?>">
					    <?php esc_html_e('Load More', 'univero'); ?>
					</a>
				</p>
			<?php } ?>
		</div>
	</div>
	<?php wp_reset_postdata(); ?>
<?php endif; ?>