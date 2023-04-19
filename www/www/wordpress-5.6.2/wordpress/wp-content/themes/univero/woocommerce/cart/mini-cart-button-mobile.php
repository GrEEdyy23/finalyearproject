<?php   global $woocommerce; ?>
<div class="ninzio-topcart ninzio-topcart-mobile">
    <div class="dropdown">
        <a class="dropdown-toggle mini-cart" data-toggle="dropdown" aria-expanded="true" role="button" aria-haspopup="true" data-delay="0" href="#" title="<?php esc_attr_e('View your shopping cart', 'univero'); ?>">
            <span class="text-skin cart-icon">            	
                              
                <span class="total-price"><?php echo WC()->cart->get_cart_subtotal(); ?></span>
                <span class="icon"><i class="univero-shop3"></i></span>
            </span>
        </a>            
        <div class="dropdown-menu">
            <div class="widget_shopping_cart_content">
            </div>
        </div>
    </div>
</div>