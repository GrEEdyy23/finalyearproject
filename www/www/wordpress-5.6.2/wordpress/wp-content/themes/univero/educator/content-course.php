<?php
/**
 * Renders each course in the shortcode-courses.php template.
 *
 * @version 1.0.1
 */

$edr_courses = Edr_Courses::get_instance();
$course_id = get_the_ID();
$price = $edr_courses->get_course_price( $course_id );
$price_str = ( $price > 0 ) ? '<span class="letter-0">'.edr_format_price( $price ).'</span>' : _x( 'Free', 'price', 'univero' );
$thumb_size = apply_filters( 'edr_courses_thumb_size', 'univero-course-grid');
?>
<article id="course-<?php echo intval( $course_id ); ?>" class="edr-course edr-course-grid">
	<div class="edr-thumbnail-wrapper">
		<?php if ( has_post_thumbnail() ) : ?>
			<div class="edr-course__image">
				<a href="<?php the_permalink(); ?>"><?php the_post_thumbnail( $thumb_size ); ?></a>
			</div>

			<div class="edr-course__price <?php echo esc_attr($price > 0?'':'free-label'); ?>"><?php echo wp_kses_post($price_str); ?></div>
		<?php endif; ?>
	</div>
	<div class="edr-course__content">
		<header class="edr-course__header">
			<h2 class="edr-course__title"><a href="<?php the_permalink(); ?>"><?php the_title(); ?></a></h2>
			<div class="edr-teacher">
				<?php echo get_the_author(); ?>
			</div>
            <?php if (has_excerpt()) { ?>
                <div class="meta-excerpt"><?php echo wp_kses_post(univero_substring( get_the_excerpt(), 18, '...' )); ?></div>
            <?php } ?>
		</header>
		<div class="meta-data clearfix">
			<div class="header-meta">
				<?php
					$total_rating = univero_get_total_rating( get_the_ID() );
				?>
				<div class="entry-info pull-left">
					<?php $registered = univero_educator_get_students_by_course( get_the_ID() ); ?>
					<div class="edr-registered">
						<i class="univero-users"></i> <?php echo count($registered); ?>
					</div>
					<div class="edr-comments">
						<i class="univero-chat2"></i> <?php comments_number( '0', '1', '%' ); ?>
					</div>
				</div>
				<div class="course-review pull-right">
					<?php univero_print_review($total_rating); ?>
				</div>
			</div>		
		</div>
	</div>
</article>