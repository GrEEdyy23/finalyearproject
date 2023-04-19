<?php 
global $product;
$image_attributes = wp_get_attachment_image_src( get_post_thumbnail_id($product->get_id() ), 'blog-thumbnails' );
$availability      = $product->get_availability();
?>
<div class="product-block grid" data-product-id="<?php echo esc_attr($product->get_id()); ?>">
    <div class="block-inner">
        <figure class="image <?php echo esc_attr($availability['class'] == 'out-of-stock'?'out':''); ?>">
            <?php 
                $sale = univero_show_percent_disount();
                if ($sale) { ?>
                <span class="percent-sale bg-theme">
                    <span class="percent-sale-iiner">- <?php echo wp_kses_post($sale); ?></span>                            
                </span>
            <?php }else{ ?>
                <?php woocommerce_show_product_loop_sale_flash(); ?>
            <?php } ?>
            <?php
                // Availability
                $availability_html = empty( $availability['availability'] ) ? '' : '<span class="stock ' . esc_attr( $availability['class'] ) . '">' . esc_html( $availability['availability'] ) . '</span>';
                echo apply_filters( 'woocommerce_stock_html', $availability_html, $availability['availability'], $product );
            ?>
            <a title="<?php echo esc_attr(get_the_title()); ?>" href="<?php the_permalink(); ?>" class="product-image">
                <?php
                    /**
                    * woocommerce_before_shop_loop_item_title hook
                    *
                    * @hooked woocommerce_show_product_loop_sale_flash - 10
                    * @hooked woocommerce_template_loop_product_thumbnail - 10
                    */
                    univero_swap_images();
                ?>
            </a>

            <div class="groups-button">
                <?php if (univero_get_config('show_quickview', true)) { ?>
                    <div class="quick-view">
                        <a href="<?php the_permalink(); ?>" class="quickview icon-theme icon-theme--small icon-theme--light" data-productslug="<?php echo esc_attr($post->post_name); ?>">
                           <i class="univero-eye"> </i>
                        </a>
                    </div>
                <?php } ?>
                <?php do_action( 'woocommerce_after_shop_loop_item' ); ?>
            </div>
        </figure>
    </div>
    <div class="product-entry-content">
        <h3 class="name"><a href="<?php the_permalink(); ?>"><?php the_title(); ?></a></h3>

        <div class="rating">
            <?php
                if($rating_html = wc_get_rating_html( $product->get_average_rating() )){
                    echo wp_kses_post( wc_get_rating_html( $product->get_average_rating() ) );

                }else{
                    echo '<div class="star-rating"></div>';
                }
            ?>
        </div>

        <?php
            /**
            * woocommerce_after_shop_loop_item_title hook
            *
            * @hooked woocommerce_template_loop_rating - 5
            * @hooked woocommerce_template_loop_price - 10
            */
            remove_action('woocommerce_after_shop_loop_item_title','woocommerce_template_loop_rating',5);
            do_action( 'woocommerce_after_shop_loop_item_title');
        ?>
    </div>
    
</div>