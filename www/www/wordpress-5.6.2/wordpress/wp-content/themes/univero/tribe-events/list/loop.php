<?php
/**
 * List View Loop
 * This file sets up the structure for the list loop
 *
 * Override this template in your own theme by creating a file at [your-theme]/tribe-events/list/loop.php
 *
 * @package TribeEventsCalendar
 *
 */

if ( ! defined( 'ABSPATH' ) ) {
	die( '-1' );
} ?>

<?php
global $post;
global $more;
$more = false;

$display_mode = univero_get_config('event_archive_display_mode', 'list');

$columns = univero_get_config('event_archive_columns', 3);
$bcol = floor( 12 / $columns );
$class = 'col-md-'.$bcol.($columns > 1 ? ' col-sm-6' : '');
?>

<div class="tribe-events-loop clearfix">
	<?php if ($display_mode == 'grid') { ?>
		<div class="row">
		<?php $count = 1; while ( have_posts() ) : the_post(); ?>
			<div class="<?php echo esc_attr($class); ?> <?php echo esc_attr($count%$columns == 1 ? ' md-clearfix':''); ?> <?php echo esc_attr(($columns > 1 && $count%2 == 1) ? ' sm-clearfix' : ''); ?>">
				<?php do_action( 'tribe_events_inside_before_loop' ); ?>

				<?php
				$post_parent = '';
				if ( $post->post_parent ) {
					$post_parent = ' data-parent-post-id="' . esc_attr(absint( $post->post_parent )) . '"';
				}
				?>
				<div id="post-<?php the_ID() ?>" class="<?php tribe_events_event_classes() ?>" <?php echo wp_kses_post($post_parent); ?>>
					<?php tribe_get_template_part( 'list/single', 'event' ) ?>
				</div>


				<?php do_action( 'tribe_events_inside_after_loop' ); ?>
			</div>
		<?php $count++; endwhile; ?>
	<?php } else { ?>
		<?php while ( have_posts() ) : the_post(); ?>
			<?php do_action( 'tribe_events_inside_before_loop' ); ?>

			<?php
			$post_parent = '';
			if ( $post->post_parent ) {
				$post_parent = ' data-parent-post-id="' . esc_attr(absint( $post->post_parent )) . '"';
			}
			?>
				<?php tribe_get_template_part( 'list/single', 'event' ) ?>

			<?php do_action( 'tribe_events_inside_after_loop' ); ?>
		<?php endwhile; ?>
		</div>
	<?php } ?>
</div><!-- .tribe-events-loop -->
