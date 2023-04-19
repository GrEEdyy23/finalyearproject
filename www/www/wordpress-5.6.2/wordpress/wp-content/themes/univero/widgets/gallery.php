<?php

extract( $args );
extract( $instance );
$title = apply_filters('widget_title', $instance['title']);

if ( $title ) {
    echo wp_kses_post($before_title . $title . $after_title);
}

$args = array(
	'post_type' => 'ninzio_gallery',
	'posts_per_page' => $number_post,
);
$loop = new WP_Query( $args );
if ( $loop->have_posts() ) :
	$bcol = 4;
	$_id = univero_random_key();
?>
	<div class="widget-content widget-gallery widget-gallery-sidebar" id="widget-gallery-<?php echo esc_attr($_id); ?>">
		<div class="row">
    		<?php while ( $loop->have_posts() ) : $loop->the_post(); ?>
    			<?php
    				$img_url = get_the_post_thumbnail_url(get_the_ID(), 'full');
    			?>
    			<?php if ( !empty($img_url) ): ?>
					<div class="col-md-<?php echo esc_attr($bcol); ?> col-xs-2">
						<div class="gallery-item">
							<figure class="gallery-item-image">
								<a href="<?php echo esc_url_raw($img_url); ?>" class="popup-image-gallery" title="<?php echo esc_attr(get_the_title()); ?>">
							    	<?php univero_display_post_image('thumbnail'); ?>
							    </a>
							</figure>
						</div>
            		</div>
        		<?php endif; ?>
    		<?php endwhile; ?>
		</div>
		<?php wp_reset_postdata(); ?>
	</div>
<?php endif; ?>