<?php
// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

$cls = $css = '';

extract( shortcode_atts( array(
	'items'		=> '3',
	'auto_scroll' => 'false',
	'cat_slug' => '',
	'column'		=> '3c',
	'column2'		=> '2c',
	'column3'		=> '1c'
), $atts ) );

if ( empty( $items ) ) return;

$column = intval( $column );
$column2 = intval( $column2 );
$column3 = intval( $column3 );

$query_args = array(
    'post_type' => 'post',
    'posts_per_page' => $items
);

if ( ! empty( $cat_slug ) )
	$query_args['category_name'] = $cat_slug;

$query = new WP_Query( $query_args );
if ( ! $query->have_posts() ) { return; }
ob_start(); ?>

<div class="widget-news <?php echo esc_attr( $cls ); ?>" data-auto="<?php echo esc_attr( $auto_scroll ); ?>" data-gap="0" data-column="<?php echo esc_attr( $column ); ?>" data-column2="<?php echo esc_attr( $column2 ); ?>" data-column3="<?php echo esc_attr( $column3 ); ?>">
<?php if ( $query->have_posts() ) : ?>

	<div class="owl-carousel owl-theme">
    <?php while ( $query->have_posts() ) : $query->the_post(); ?>

		<div class="post-grid">
			<div >
				<div class="entry-image-wrapper">
					<?php
					echo '<div class="entry-thumb">';
					echo get_the_post_thumbnail( get_the_ID(), 'univero-course-grid' );
					echo '</div>';
					echo '<div class="entry-date bg-theme"><span class="day">'.get_the_date("d").'</span><span class="month">'.get_the_date("M").'</span></div>';
					?>
				</div><!-- /.thumb-wrap -->

                <div class="entry-content clearfix">
					<?php echo '<h4 class="entry-title"><a href="'. esc_url( get_the_permalink() ) .'">'. get_the_title() .'</a></h4>'; ?>
					<div class="entry-description" style="margin: 0 0 18px"><?php echo wp_trim_words( get_the_excerpt(), 14, '...' ); ?></div>

					<a href="<?php echo esc_url( get_the_permalink() ); ?>" class="readmore">Read More</a>

                </div><!-- /.text-wrap -->
	        </div>
	    </div><!-- /.news-item -->
	    
	<?php endwhile; ?>
	</div><!-- /.owl-carousel -->

<?php endif; ?>

<?php wp_reset_postdata(); ?>
</div><!-- /.widget-news -->
<?php
$return = ob_get_clean();
echo $return;