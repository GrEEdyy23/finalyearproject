<?php
/**
 * Single Event Meta Template
 *
 * Override this template in your own theme by creating a file at:
 * [your-theme]/tribe-events/modules/meta.php
 *
 * @package TribeEventsCalendar
 */

do_action( 'tribe_events_single_meta_before' );


// Do we want to group venue meta separately?
$set_venue_apart = true;
?>

<?php
do_action( 'tribe_events_single_event_meta_primary_section_start' );
?>
<div class="event-meta">
	<?php
	// Always include the main event details in this first section
	tribe_get_template_part( 'modules/meta/details' );

	?>

	<?php if ( $set_venue_apart ) :
		tribe_get_template_part( 'modules/meta/venue' );
	endif;
	?>

	<!-- category -->
	<?php
	$terms = get_the_terms( get_the_ID(), 'tribe_events_cat' );
	if ( $terms ) {
		?>
		<div class="category_wrapper">
			<i class="univero-map"></i>
			<span class="media-content">
				<?php
				foreach ($terms as $term) {
					?>
					<a href="<?php echo esc_url(get_term_link($term)); ?>"><?php echo wp_kses_post($term->name); ?></a>
					<?php
				}
				?>
			</span>
		</div>
		<?php
	}
	?>
	<div class="author">
		<i class="univero-user3" aria-hidden="true"></i>
		<?php the_author(); ?>
	</div>
	<div class="comments"><i class="univero-speech-bubble" aria-hidden="true"></i> <?php comments_number( '0 Comment', '1 Comment', '% Comments' ); ?></div>
	<?php $viewed = (int)get_post_meta(get_the_ID(), '_views_count', true); ?>
    <div class="views"><i class="univero-eye" aria-hidden="true"></i> <?php echo sprintf(_n('%d View', '%d Views', $viewed, 'univero'), $viewed); ?></div> 
</div>

<?php
do_action( 'tribe_events_single_event_meta_primary_section_end' );

do_action( 'tribe_events_single_meta_after' );