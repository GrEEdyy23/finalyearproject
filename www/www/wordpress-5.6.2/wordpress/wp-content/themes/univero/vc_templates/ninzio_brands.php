<?php
$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );

$bcol = 12/$columns;
if($columns == 5) $bcol='c5';
$args = array(
	'post_type' => 'ninzio_brand',
	'posts_per_page' => $number,
	'row'            => 1,
);
$loop = new WP_Query($args);

?>
<div class="widget widget-brands <?php echo esc_attr($el_class.' '.$style); ?>">
    <?php if ($title!=''): ?>
        <h3 class="widget-title">
            <span><?php echo esc_attr( $title ); ?></span>
            <?php if ( isset($subtitle) && $subtitle ): ?>
                <span class="subtitle"><?php echo esc_html($subtitle); ?></span>
            <?php endif; ?>
        </h3>
    <?php endif; ?>
    <div class="widget-content">
    	<?php if ( $loop->have_posts() ): ?>
    		<?php if ( $layout_type == 'carousel' ): ?>
    			<div class="owl-carousel products owl-carousel-top" data-items="<?php echo esc_attr($columns); ?>" data-smallmedium="3" data-extrasmall="2" data-carousel="owl" data-pagination="false" data-nav="true">
		    		<?php $count=0; while ( $loop->have_posts() ): $loop->the_post(); ?>
		    			<?php if ($count%$rows == 0) { ?>
		    				<div class="item">
	    				<?php } ?>
	    					<div class="item-wrapper">
				                <?php $link = get_post_meta( get_the_ID(), 'ninzio_brand_link', true); ?>
				                <?php $link = $link ? $link : '#'; ?>
								<a href="<?php echo esc_url($link); ?>" target="_blank">
									<?php the_post_thumbnail( 'full' ); ?>
								</a>
							</div>
						<?php if ($count%$rows == ($rows - 1) || $count == ($loop->post_count - 1) ) { ?>
				        	</div>
				        <?php } ?>
		    		<?php $count++; endwhile; ?>
	    		</div>
	    	<?php else: ?>
	    		<div class="row no-margin">
		    		<?php $count = 1; while ( $loop->have_posts() ): $loop->the_post(); ?>
		    			<div class="item col-md-<?php echo esc_attr($bcol); ?> col-xs-6 <?php if($count%$columns == 1) echo 'first-child'; if($count%$columns == 0) echo 'last-child'; ?>">
			                <?php $link = get_post_meta( get_the_ID(), 'ninzio_brand_link', true); ?>
			                <?php $link = $link ? $link : '#'; ?>
							<a href="<?php echo esc_url($link); ?>" target="_blank">
								<?php the_post_thumbnail( 'full' ); ?>
							</a>
				        </div>
		    		<?php $count++; endwhile; ?>
	    		</div>
	    	<?php endif; ?>
    	<?php endif; ?>
    	<?php wp_reset_postdata(); ?>
    </div>
</div>