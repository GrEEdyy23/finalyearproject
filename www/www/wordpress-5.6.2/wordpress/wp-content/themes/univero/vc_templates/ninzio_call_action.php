<?php
$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );
$img = wp_get_attachment_image_src($image,'full');
?>
<div class="clearfix widget-action <?php echo esc_attr($el_class.' '.$style); ?>">
    <?php if ( !empty($img) && isset($img[0]) ): ?>
        <div class="img-left">
            <img src="<?php echo esc_url_raw($img[0]); ?>" alt="<?php esc_attr_e('Image', 'univero'); ?>">
        </div>
    <?php endif; ?>
    <div class="info">
	<?php if($title!=''): ?>
        <h3 class="title" >
           <span><?php echo esc_attr( $title ); ?></span>
        </h3>
    <?php endif; ?>
    <?php if(wpb_js_remove_wpautop( $content, true )){ ?>
        <div class="description">
            <?php echo wpb_js_remove_wpautop( $content, true ); ?>
        </div>
    <?php } ?>
    </div>
    <?php if( $linkbutton1 !='' ){ ?>
        <div class="action">
            <a class="btn <?php echo esc_attr( $buttons1 ); ?>" href="<?php echo esc_attr( $linkbutton1 ); ?>"> <span><?php echo wp_kses_post( $textbutton1 ); ?></span> </a>
        </div>
    <?php } ?>
</div>