<?php
$course_id = get_the_ID();

$gallery = $images = get_post_meta($course_id, 'ninzio_educator_gallery_images', true);

if ( has_post_thumbnail() ) {
	$thumbnail_id = get_post_thumbnail_id();
	$thumbnail = wp_get_attachment_image_src( $thumbnail_id, 'full' );

	if ( is_array($images) ) {
		if ( !in_array($thumbnail_id, $images) ) {
			$gallery = array($thumbnail_id => $thumbnail[0]) + $images;
		}
	} else {
		$gallery = array($thumbnail_id => $thumbnail[0]);
	}
}
if ( !empty($gallery) ) {
	?>
	<div class="course-gallery">
		<div class="course-gallery-preview course-box-image-inner">

			<div class="owl-carousel course-gallery-preview-owl" data-smallmedium="1" data-extrasmall="1" data-items="1" data-carousel="owl" data-pagination="false" data-nav="false" data-margin="1" <?php echo (count($gallery) > 1 ? 'data-loop="false"' : ''); ?>>
				<?php foreach ( $gallery as $id => $src ) : ?>
					<a href="<?php echo esc_url( $src ); ?>" rel="univero-course-gallery">
						<?php echo wp_get_attachment_image( $id , 'univero-course-gallery' );?>
					</a>
				<?php endforeach; ?>
			</div>
		</div>

		<div class="owl-carousel course-gallery-index" data-smallmedium="5" data-extrasmall="3" data-items="5" data-carousel="owl" data-pagination="false" data-nav="false">
			<?php $index = 0; ?>
			<?php foreach ( $gallery as $id => $src ) : ?>
				<div <?php echo ( 0 == $index ) ? 'class="active thumb-link"' : 'class="thumb-link"'; ?>>
					<a href="<?php echo esc_url( $src ); ?>">
						<?php echo wp_get_attachment_image( $id, 'univero-course-gallery-thumb' ); ?>
					</a>
					<?php $index++; ?>
				</div>
			<?php endforeach; ?>
		</div>
	</div><!-- /.course-gallery -->
	<?php
}