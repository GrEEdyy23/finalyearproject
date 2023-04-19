<?php

$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );


$loop = univero_get_products( array(), $type, 1, $number );
?>
<?php if ( $loop->have_posts() ) : ?>
    <div class="widget widget-products <?php echo esc_attr($el_class); ?>">
        <div class="widget-content woocommerce">
            
            <?php wc_get_template( 'layout-products/'.$layout_type.'.php' , array( 'loop' => $loop, 'columns' => $columns, 'number' => $number, 'show_pagination' => $show_pagination, 'show_nav' => $show_nav ) ); ?>

        </div>
    </div>
<?php endif; ?>