<?php
/**
 * List View Single Event
 * This file contains one event in the list view
 *
 * Override this template in your own theme by creating a file at [your-theme]/tribe-events/list/single-event.php
 *
 * @package TribeEventsCalendar
 *
 */
if ( ! defined( 'ABSPATH' ) ) {
	die( '-1' );
}

global $post;
$start = strtotime($post->EventStartDate);

?>
<div class="tribe-events-list event-grid clearfix">
    <!-- Event date -->
    
    <div class="tribe-events-image clearfix">
      <?php echo tribe_event_featured_image( null, 'univero-event-thumb' ) ?>
      <!-- Event Title -->
      <div class="entry-date-wrapper">        
        <div class="entry-date bg-theme">
          <span class="day"><?php echo esc_attr( date('d', $start) ); ?></span>
          <span class="month-year"><?php echo esc_attr( date('M', $start) ); ?></span>
        </div>
      </div>

      <div class="tribe-events-title-wrapper">
        <?php do_action( 'tribe_events_before_the_event_title' ) ?>
        <h2 class="tribe-events-list-event-title">
          <a class="tribe-event-url" href="<?php echo esc_url( tribe_get_event_link() ); ?>" title="<?php the_title_attribute() ?>" rel="bookmark">
            <?php the_title() ?>
          </a>
        </h2>
        <?php do_action( 'tribe_events_after_the_event_title' ) ?>
      </div>
    </div>
    
</div>