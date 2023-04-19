<?php 

$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );

$bcol = 12/$columns;
$categories = (array) vc_param_group_parse_atts( $categories );
if ( !empty($categories) ): ?>
	<div class="widget-course-categories <?php echo esc_attr($el_class.' '.$layout_type); ?>">
		<?php if ( $layout_type == 'grid' ) { ?>
			<div class="row">
				<?php foreach ($categories as $item): ?>
					<?php
						if ( empty($item['category']) ) {
							continue;
						}
						$term = get_term_by( 'slug', $item['category'], 'edr_course_category' );
						if ( ! empty( $term ) && ! is_wp_error( $term ) ) {
							$link = get_term_link( $term, 'edr_course_category' );
							$img_url = '';
							if ( isset($item['image']) && $item['image'] ) {
								$img = wp_get_attachment_image_src($item['image'], 'full');
								if (isset($img[0]) && $img[0]) {
									$img_url = $img[0];
								}
							}
					?>
						<div class="col-xs-6 col-sm-<?php echo esc_attr($bcol); ?>">
							<div class="category-wrapper bg-theme <?php echo esc_attr($img_url ? 'has-background' : ''); ?>" style="<?php echo trim($img_url ? 'background: url('.esc_url($img_url).')' : ''); ?>">
								<a href="<?php echo esc_url($link); ?>">
									<div  class="category-wrapper-box">
										<?php if ( isset($item['label']) && $item['label'] ): ?>
											<span class="label"><?php echo esc_html($item['label']); ?></span>
										<?php endif; ?>
										<?php if ( isset($item['icon_image']) && $item['icon_image'] ): ?>
											<?php $img = wp_get_attachment_image_src($item['icon_image'], 'full'); ?>
											<?php if (isset($img[0]) && $img[0]) { ?>
								    			<?php univero_display_image($img); ?>
											<?php } ?>
										<?php elseif( isset($item['icon_font']) && $item['icon_font'] ) : ?>
											<div class="icon">
												<i class="<?php echo esc_attr($item['icon_font']); ?>"></i>
											</div>
										<?php endif; ?>
										<div class="title">
											<span>
												<?php echo wp_kses_post($item['name']); ?>												
											</span>
										</div>		
									</div>								
								</a>
							</div>
						</div>
					<?php } ?>
				<?php endforeach; ?>
			</div>
		<?php } else { ?>
			<div class="owl-carousel owl-carousel-top" data-items="<?php echo esc_attr($columns); ?>" data-carousel="owl" data-smallmedium="2" data-extrasmall="2" data-pagination="<?php echo esc_attr(($show_pagination && count($categories) > $columns)  ? 'true' : 'false'); ?>" data-nav="<?php echo esc_attr($show_nav ? 'true' : 'false'); ?>">
				<?php foreach ($categories as $item): ?>
					<?php
						if ( empty($item['category']) ) {
							continue;
						}
						$term = get_term_by( 'slug', $item['category'], 'edr_course_category' );
						if ( ! empty( $term ) && ! is_wp_error( $term ) ) {
							$link = get_term_link( $term, 'edr_course_category' );
							$img_url = '';
							if ( isset($item['image']) && $item['image'] ) {
								$img = wp_get_attachment_image_src($item['image'], 'full');
								if (isset($img[0]) && $img[0]) {
									$img_url = $img[0];
								}
							}
					?>
							<div class="category-wrapper bg-theme <?php echo esc_attr($img_url ? 'has-background' : ''); ?>" style="<?php echo trim($img_url ? 'background: url('.esc_url($img_url).')' : ''); ?>">
								<a href="<?php echo esc_url($link); ?>">
									<?php if ( isset($item['label']) && $item['label'] ): ?>
										<span class="label"><?php echo esc_html($item['label']); ?></span>
									<?php endif; ?>
									<?php if ( isset($item['icon_image']) && $item['icon_image'] ): ?>
										<?php $img = wp_get_attachment_image_src($item['icon_image'], 'full'); ?>
										<?php if (isset($img[0]) && $img[0]) { ?>
							    			<?php univero_display_image($img); ?>
										<?php } ?>
									<?php elseif( isset($item['icon_font']) && $item['icon_font'] ) : ?>
										<div class="icon">
											<i class="<?php echo esc_attr($item['icon_font']); ?>"></i>
										</div>
									<?php endif; ?>
									<div class="title">
										<span><?php echo wp_kses_post($item['name']); ?></span>
									</div>
								</a>
							</div>
					<?php } ?>
				<?php endforeach; ?>
			</div>
		<?php } ?>
	</div>
<?php endif; ?>