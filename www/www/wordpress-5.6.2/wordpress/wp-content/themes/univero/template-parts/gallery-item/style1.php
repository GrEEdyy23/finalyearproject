<?php
$img_url = get_the_post_thumbnail_url(get_the_ID(), 'full');
$url = get_post_meta( get_the_ID(), 'ninzio_gallery_url', true );
$thumbsize = isset($thumbsize) ? $thumbsize : '';
?>

<div class="gallery-item style1">
	<figure class="gallery-item-image">
    	<?php univero_display_post_image($thumbsize); ?>   
	</figure>	
	<div class="gallery-item-content">
		<?php if ( $url ) { ?>
    		<a href="<?php echo esc_url_raw($url); ?>" class="icon-theme icon-theme--small icon-theme--light">
    			<i class="univero-link"></i>
    		</a>
		<?php } ?>
		<a href="<?php echo esc_url_raw($img_url); ?>" class="popup-image-gallery icon-theme icon-theme--small icon-theme--light" title="<?php echo esc_attr(get_the_title()); ?>">
			<i class="univero-magnifier4"></i>
		</a>
	</div>
</div>