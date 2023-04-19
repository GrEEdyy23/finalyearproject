<?php

$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );

$loop = univero_get_events( $orderby, $number );
$bcol = 12/$columns;
?>
<?php if ( $loop->have_posts() ) : ?>
<div class="widget-event <?php echo esc_attr($el_class); ?>">
	
    <div class="widget-content">
        <?php if ( $layout_type == 'grid' ) { ?>
			<div class="row">
				<?php $i = 0; while ( $loop->have_posts() ) : $loop->the_post(); ?>
					<div class="col-md-<?php echo esc_attr($bcol); ?> col-sm-6 <?php echo esc_attr($i%$columns == 0 ? 'md-clearfix':''); ?> <?php echo esc_attr($i%2 == 0 ? 'sm-clearfix':''); ?>">
						<?php tribe_get_template_part( 'list/single', 'event' ) ?>
					</div>
				<?php $i++; endwhile; ?>
			</div>
		<?php } elseif ( $layout_type == 'carousel' ) { ?>
			<div class="owl-carousel owl-carousel-top" data-items="<?php echo esc_attr($columns); ?>" data-carousel="owl" data-smallmedium="2" data-extrasmall="1" data-pagination="<?php echo esc_attr($show_pagination ? 'true' : 'false'); ?>" data-nav="<?php echo esc_attr($show_nav ? 'true' : 'false'); ?>">
			    <?php while ( $loop->have_posts() ): $loop->the_post(); ?>
	               	<?php tribe_get_template_part( 'list/single', 'event' ) ?>
			    <?php endwhile; ?>
			</div> 
		<?php } ?>
		<?php wp_reset_postdata(); ?>
	</div>
</div>
<?php endif; ?>