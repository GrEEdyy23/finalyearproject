<?php
$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );
?>
<div class="widget widget-text-heading <?php echo esc_attr($el_class).' '.esc_attr($style); ?>">
    <?php if ( $title !='' ) { ?>
        <h3 class="title">
            <?php echo wp_kses_post( $title ); ?>
        </h3>
    <?php } ?>
    <?php if ( $descript !='' ) { ?>
        <div class="description">
            <?php echo wp_kses_post( $descript ); ?>
        </div>
    <?php } ?>
</div>