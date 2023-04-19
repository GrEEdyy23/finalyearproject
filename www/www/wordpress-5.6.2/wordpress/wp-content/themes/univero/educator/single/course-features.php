<?php
$course_id = get_the_ID();
$duration = univero_educator_get_meta('duration');
$registered = univero_educator_get_students_by_course( $course_id );
$certificate = univero_educator_get_meta('certificate');
$langauge = univero_educator_get_meta('langauge');
$prerequisites = get_post_meta( $course_id, '_edr_prerequisites', true );
$difficulty = get_post_meta( $course_id, '_edr_difficulty', true );
$capacity = univero_educator_get_meta('capacity');
$startcourse = univero_educator_get_meta('startcourse');
$location = univero_educator_get_meta('location');

$obj = Edr_Courses::get_instance();
$lesson = $obj->get_course_lessons($course_id);
$nb_lesson = is_array($lesson) ? count($lesson) : 0;
?>

<ul class="course-features">
	<?php if ( !empty($registered) ) { ?>
		<li>
			<i class="univero-heart2"></i>
			<span><?php esc_html_e( 'Students:', 'univero' ); ?></span>
			<span><?php echo count( $registered ); ?></span>
		</li>
	<?php } ?>
	<?php if ( !empty($duration) ) { ?>
		<li>
			<i class="univero-time"></i>
			<span><?php esc_html_e( 'Duration:', 'univero' ); ?></span>
			<span><?php echo wp_kses_post($duration); ?></span>
		</li>
	<?php } ?>
	<?php if ( !empty($nb_lesson) ) { ?>
		<li>
			<i class="univero-asside"></i>
			<span><?php esc_html_e( 'Lessons:', 'univero' ); ?></span>
			<span><?php echo wp_kses_post($nb_lesson); ?></span>
		</li>
	<?php } ?>
	<?php if ( !empty($langauge) ) { ?>
		<li>
			<i class="univero-location"></i>
			<span><?php esc_html_e( 'Language:', 'univero' ); ?></span>
			<span><?php echo wp_kses_post( $langauge ); ?></span>
		</li>
	<?php } ?>
	<?php if ( !empty($location) ) { ?>
		<li>
			<i class="univero-location"></i>
			<span><?php esc_html_e( 'Location:', 'univero' ); ?></span>
			<span><?php echo wp_kses_post($location); ?></span>
		</li>
	<?php } ?>
	<li>
		<i class="univero-heart2"></i>
		<span><?php esc_html_e( 'Prerequisites:', 'univero' ); ?></span>
		<span><?php echo wp_kses_post($prerequisites ? esc_html__('Yes', 'univero') : esc_html__('No', 'univero') ); ?></span>
	</li>
	<?php if ( !empty($difficulty) ) { ?>
		<li>
			<i class="univero-level"></i>
			<span><?php esc_html_e( 'Skill Level:', 'univero' ); ?></span>
			<span><?php echo wp_kses_post( $difficulty ); ?></span>
		</li>
	<?php } ?>
	<?php if ( !empty($capacity) ) { ?>
		<li>
			<i class="univero-time"></i>
			<span><?php esc_html_e( 'Course Capacity:', 'univero' ); ?></span>
			<span><?php echo wp_kses_post( $capacity ); ?></span>
		</li>
	<?php } ?>
	<?php if ( !empty($startcourse) ) { ?>
		<li>
			<i class=" univero-users"></i>
			<span><?php esc_html_e( 'Start Course:', 'univero' ); ?></span>
			<span><?php echo wp_kses_post($startcourse); ?></span>
		</li>
	<?php } ?>
	<li>
		<i class="univero-certificate"></i>
		<span><?php esc_html_e( 'Certificate:', 'univero' ); ?></span>
		<span><?php echo wp_kses_post($certificate ? esc_html__('Yes', 'univero') : esc_html__('No', 'univero') ); ?></span>
	</li>
</ul>
