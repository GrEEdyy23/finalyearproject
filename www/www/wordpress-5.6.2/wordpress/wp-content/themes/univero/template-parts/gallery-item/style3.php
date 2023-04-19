<?php
$img_url = get_the_post_thumbnail_url(get_the_ID(), 'full');
$url = get_post_meta( get_the_ID(), 'ninzio_gallery_url', true );
$thumbsize = isset($thumbsize) ? $thumbsize : '';
?>

<div class="gallery-item style1">
	<a href="<?php echo esc_url_raw($img_url); ?>" class="popup-image-gallery" title="<?php echo esc_attr(get_the_title()); ?>">
		<figure class="gallery-item-image">
	    	<?php univero_display_post_image($thumbsize); ?>
		</figure>
	</a>
</div>