<?php
$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );
$args = array(
	'post_type' => 'ninzio_testimonial',
	'posts_per_page' => $number,
	'post_status' => 'publish',
);
$loop = new WP_Query($args);
?>
<div class="widget-testimonials <?php echo esc_attr($el_class. ' '. $layout_type); ?> <?php echo esc_attr(($layout_type =='style3')?'under-line widget':''); ?>">
	<?php if ($title!=''): ?>
        <h3 class="widget-title">
            <span><?php echo esc_attr( $title ); ?></span>
        </h3>
    <?php endif; ?>

	<?php if ( $loop->have_posts() ): ?>
        <div class="owl-carousel <?php echo esc_attr(($layout_type =='style3')?'owl-carousel-top':''); ?>" data-items="<?php echo esc_attr($columns); ?>" data-carousel="owl" data-pagination="true" data-nav="true" data-smallmedium="1" data-extrasmall="1">
            <?php while ( $loop->have_posts() ): $loop->the_post(); ?>
                <?php
                $job = get_post_meta( get_the_ID(), 'ninzio_testimonial_job', true );
                $link = get_post_meta( get_the_ID(), 'ninzio_testimonial_link', true );
                ?>
                <div class="testimonials-body">
                    <div class="testimonials-profile">
                        <div class="testimonial-avatar">
                            <?php the_post_thumbnail('widget'); ?>
                        </div>                        

                        <div class="clearfix info-bottom">                            
                            <div class="testimonial-meta">
                                <div class="info">
                                    <?php if (!empty($link)) { ?>
                                      <h3 class="name-client"><a href="<?php echo esc_url_raw($link); ?>"><?php the_title(); ?></a></h3>
                                    <?php } else { ?>
                                      <h3 class="name-client"><?php the_title(); ?></h3>
                                    <?php } ?>
                                    <span class="job"> <span class="space">-</span> <?php echo wp_kses_post($job); ?></span>   
                                </div>
                            </div>
                        </div>
                        <div class="description"><?php the_excerpt(); ?></div>

                    </div> 
                </div>
            <?php endwhile; ?>
        </div>
	<?php endif; ?>
</div>
<?php wp_reset_postdata(); ?>