<?php

extract( $args );
extract( $instance );
$title = apply_filters('widget_title', $instance['title']);

if ( $title ) {
    echo wp_kses_post($before_title . $title . $after_title);
}

$loop = univero_educator_get_courses($orderby, $number_post);

if ( $loop->have_posts() ) :
?>
	<div class="widget-content">
		<?php while ( $loop->have_posts() ) : $loop->the_post(); ?>
			<div class="course-item">
				<?php Edr_View::template_part( 'content-course-list', 'simple' ); ?>
			</div>
		<?php endwhile; ?>
		<?php wp_reset_postdata(); ?>
	</div>
<?php endif; ?>