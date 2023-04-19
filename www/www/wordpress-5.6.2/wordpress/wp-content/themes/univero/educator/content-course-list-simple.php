<?php
/**
 * Renders each course in the shortcode-courses.php template.
 *
 * @version 1.0.1
 */

$edr_courses = Edr_Courses::get_instance();
$thumb_size = apply_filters( 'edr_courses_thumb_size', 'univero-course-thumb');
$bookmark = isset($bookmark) ? $bookmark : false;
?>
<article <?php if ($bookmark) { ?> id="bookmark-course-<?php echo esc_attr( get_the_ID() ); ?>" <?php } ?> class="edr-course edr-course-list-simple">
	<div class="media">
		<div class="media-left">
			<div class="edr-thumbnail-wrapper">
				<?php if ( has_post_thumbnail() ) : ?>
					<div class="edr-course__image">
						<a href="<?php the_permalink(); ?>"><?php the_post_thumbnail( $thumb_size ); ?></a>
					</div>
				<?php endif; ?>
			</div>
		</div>
		<div class="media-body">
			<div class="edr-course__header">
				<h2 class="edr-course__title"><a href="<?php the_permalink(); ?>"><?php the_title(); ?></a></h2>
				<div class="course-review">
                    <?php
                        $total_rating = univero_get_total_rating( get_the_ID() );
                    ?>
                    <?php univero_print_review($total_rating); ?>
                </div>
			</div>
		</div>
	</div>
</article>