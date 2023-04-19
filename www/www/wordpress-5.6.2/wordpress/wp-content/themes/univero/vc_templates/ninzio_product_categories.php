<?php

$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );

$categories = (array) vc_param_group_parse_atts( $categories );
if ( !empty($categories) ) {
	?>
	<div class="widget-product-categories <?php echo esc_attr($el_class); ?>">
		<div class="widget-content">
			<?php if ( $layout_type == 'carousel' ) { ?>
				<div class="owl-carousel" data-items="<?php echo esc_attr($columns); ?>" data-carousel="owl" data-smallmedium="2" data-extrasmall="1" data-pagination="<?php echo esc_attr($show_pagination ? 'true' : 'false'); ?>" data-nav="<?php echo esc_attr($show_nav ? 'true' : 'false'); ?>">
					<?php foreach ($categories as $category): ?>
						<?php if ( !empty($category['category']) ) { 
							$term = get_term_by( 'slug', $category['category'], 'product_cat');
							if ( !empty($term) ) {
								if ( isset($category['image']) && $category['image'] ){
									$image = wp_get_attachment_image_src($category['image'], 'full');
								}
						?>
							<div class="category-item">
								<div class="category-image">
									<a href="<?php echo esc_url(get_term_link($term)); ?>">
										<?php if ( !empty($image) ) { ?>
											<?php univero_display_image($image); ?>
										<?php } ?>
									</a>
								</div>
								<div class="category-item-content"> 
									<?php if ( isset($category['title']) && $category['title'] ) { ?>
									<h3><a href="<?php echo esc_url(get_term_link($term)); ?>"><?php echo wp_kses_post($category['title']); ?></a></h3>
								<?php } ?>
								<div class="product-count"><?php echo sprintf(_n('%d Product', '%d Products', $term->count, 'univero'), $term->count); ?></div>
								</div>								
							</div>
						<?php } ?>
						<?php } ?>
					<?php endforeach; ?>
				</div>
			<?php } else { $bcol = 12/$columns; ?>
				<div class="row">
					<?php foreach ($categories as $category): ?>
						<?php if ( !empty($category['category']) ) { 
							$term = get_term_by( 'slug', $category['category'], 'product_cat');
							if ( !empty($term) ) {
								if ( isset($category['image']) && $category['image'] ){
									$image = wp_get_attachment_image_src($category['image'], 'full');
								}
						?>
							<div class="category-item col-sm-<?php echo esc_attr($bcol); ?>">
								<div class="category-image">
									<a href="<?php echo esc_url(get_term_link($term)); ?>">
										<?php if ( !empty($image) ) { ?>
											<?php univero_display_image($image); ?>
										<?php } ?>
									</a>
								</div>
								<div class="category-item-content"> 
									<?php if ( isset($category['title']) && $category['title'] ) { ?>
									<h3><a href="<?php echo esc_url(get_term_link($term)); ?>"><?php echo wp_kses_post($category['title']); ?></a></h3>
								<?php } ?>
								<div class="product-count"><?php echo sprintf(_n('%d Product', '%d Products', $term->count, 'univero'), $term->count); ?></div>
								</div>
							</div>
						<?php } ?>
						<?php } ?>
					<?php endforeach; ?>
				</div>
			<?php } ?>
		</div>
	</div>
	<?php
}