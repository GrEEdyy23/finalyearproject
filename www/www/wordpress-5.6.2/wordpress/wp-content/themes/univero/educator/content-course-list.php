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
$thumb_size = apply_filters( 'edr_courses_thumb_size', 'univero-course-list');
?>
<article id="course-<?php echo intval( $course_id ); ?>" class="edr-course edr-course-list">
	<div class="row">
		<div class="col-md-6 col-sm-5 col-xs-12">
			<div class="edr-thumbnail-wrapper">
				<?php if ( has_post_thumbnail() ) : ?>
					<div class="edr-course__image">
						<a href="<?php the_permalink(); ?>"><?php the_post_thumbnail( $thumb_size ); ?></a>
					</div>
				<?php endif; ?>
				<div class="edr-course__price <?php echo esc_attr($price > 0?'':'free-label'); ?>">					
					<span><?php echo wp_kses_post($price_str); ?></span>
				</div>
			</div>
		</div>
		<div class="col-md-6 col-sm-7 col-xs-12">
			<header class="edr-course__header">
				<h2 class="edr-course__title"><a href="<?php the_permalink(); ?>"><?php the_title(); ?></a></h2>
				<div class="edr-teacher">
					<?php echo get_the_author(); ?>
				</div>
	            <?php if (has_excerpt()) { ?>
	                <div class="meta-excerpt"><?php echo wp_kses_post(univero_substring( get_the_excerpt(), 25, '...' )); ?></div>
	            <?php } ?>
	            
			</header>
			<div class="meta-data">
				<div class="header-meta clearfix">
					<?php
						$total_rating = univero_get_total_rating( get_the_ID() );
					?>
					<ul class="info list-inline list-unstyled pull-left">
						<?php $registered = univero_educator_get_students_by_course( get_the_ID() ); ?>
						<li class="edr-registered">
							<i class="univero-users"></i>
							<span><?php echo count($registered); ?></span>
						</li>
						<li class="edr-comments">
							<i class="univero-chat2"></i> 
							<span><?php comments_number( '0', '1', '%' ); ?></span>
						</li>
					</ul>
					<div class="course-review pull-left">
						<?php univero_print_review($total_rating); ?>
					</div>
				</div>
			</div>			
		</div>
	</div>
</article>