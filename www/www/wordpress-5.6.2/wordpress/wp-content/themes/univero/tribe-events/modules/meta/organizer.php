<?php
/**
 * Single Event Meta (Organizer) Template
 *
 * Override this template in your own theme by creating a file at:
 * [your-theme]/tribe-events/modules/meta/details.php
 *
 * @package TribeEventsCalendar
 */

$phone = tribe_get_organizer_phone();
$email = tribe_get_organizer_email();
$website = tribe_get_organizer_website_link();
?>

<div class="organizer_wrapper hidden">
	<div class="media">
		<div class="media-left">
			<i class="fa fa-university"></i>
		</div>
		<div class="media-body">
			<div class="media-info-inner">
				<h3><?php echo esc_html__('Organizer', 'univero'); ?></h3>
				<span class="media-content">
					<?php echo tribe_get_organizer_label_singular(); ?>
					<?php if ( ! empty( $phone ) ): ?>
						<span class="tribe-events-span">, <?php echo esc_html( $phone ); ?></span>
					<?php endif ?>
					<?php if ( ! empty( $website ) ): ?>
						<span class="tribe-events-span">
							, <?php echo wp_kses_post( $website ); ?> 
						</span>
					<?php endif ?>
				</span>
			</div>
		</div>
	</div>
</div>