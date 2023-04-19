<?php
/**
 * Single Event Template
 * A single event. This displays the event title, description, meta, and
 * optionally, the Google map for the event.
 *
 * Override this template in your own theme by creating a file at [your-theme]/tribe-events/single-event.php
 *
 * @package TribeEventsCalendar
 *
 */

if ( ! defined( 'ABSPATH' ) ) {
	die( '-1' );
}

$events_label_singular = tribe_get_event_label_singular();
$events_label_plural = tribe_get_event_label_plural();
$event_id = get_the_ID();

?>

<?php while ( have_posts() ) :  the_post(); ?>
	<?php
	global $post;
	$start = strtotime($post->EventStartDate);
	?>
	<div class="clearfix single-event">

		<div id="tribe-events-content" class="tribe-events-single">
			<!-- Notices -->
			<?php tribe_the_notices(); ?>
			<div class="tribe-events-schedule hidden tribe-clearfix">
				<?php echo tribe_events_event_schedule_details( $event_id, '<div class="events-meta">', '</div>' ); ?>
				<?php if ( tribe_get_cost() ) : ?>
					<span class="tribe-events-divider">|</span>
					<span class="tribe-events-cost"><?php echo tribe_get_cost( null, true ) ?></span>
				<?php endif; ?>
			</div>

			<!-- Event content -->
			<?php do_action( 'tribe_events_single_event_before_the_content' ) ?>
			<div class="tribe-events-single-event-description tribe-events-content entry-content description">
				
				<div class="image-event">
					<?php the_post_thumbnail('full'); ?>
				</div>

				<?php the_title( '<h1 class="tribe-events-single-event-title">', '</h1>' ); ?>

				<div class="event-metas">
					<!-- meta -->
					<?php do_action( 'tribe_events_single_event_before_the_meta' ) ?>
					<?php
						if ( ! apply_filters( 'tribe_events_single_event_meta_legacy_mode', false ) ) {
							tribe_get_template_part( 'modules/meta' );
						} else {
							echo tribe_events_single_event_meta();
						}
					?>
					<?php do_action( 'tribe_events_single_event_after_the_meta' ) ?>
				</div>

				<div class="clearfix"></div>
				
				<!-- content -->
				<?php the_content(); ?>
				
				<?php if( univero_get_config('show_event_social_share', false) ) { ?>
               		<div class="social-share">
               			<div class="share-label"><?php esc_html_e('Share This Event', 'univero'); ?></div>
	                    <?php get_template_part( 'template-parts/sharebox' ); ?>
	                </div>
                <?php } ?>

                <?php
                if ( univero_get_config('show_event_releated', true) ) { ?>
					<div class="related-events">
						<?php get_template_part( 'template-parts/event-releated' ); ?>
					</div>
                <?php } ?>

			</div>

			<?php if( get_post_type() == Tribe__Events__Main::POSTTYPE && tribe_get_option( 'showComments', false ) ) comments_template() ?>

		</div><!-- #tribe-events-content -->
	</div>
<?php endwhile; ?>