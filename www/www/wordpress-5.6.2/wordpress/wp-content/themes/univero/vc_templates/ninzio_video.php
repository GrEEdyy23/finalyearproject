<?php
$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );
?>
<div class="widget widget-video <?php echo esc_attr($el_class);?>">
    <div class="video-wrapper-inner">
    	<div class="video">
    		<?php $img = wp_get_attachment_image_src($image, 'full'); ?>
    		<?php univero_display_image($img); ?>

            <div class="video-icon-wrap">
                <a class="popup-video" href="<?php echo esc_url_raw($video_link); ?>">
                    <i class="icon univero-play-button"></i>
                </a>
                <?php if ( !empty($title) ): ?>
                    <h3 class="title">
                        <span><?php echo wp_kses_post($title); ?></span>
                    </h3>
                <?php endif; ?>
            </div>
    	</div>
	</div>
</div>