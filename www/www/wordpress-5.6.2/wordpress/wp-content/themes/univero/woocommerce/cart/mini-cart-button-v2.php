<?php global $woocommerce; ?>
<div class="ninzio-topcart">
    <div id="cart" class="dropdown version-1">
        <a class="dropdown-toggle mini-cart icon-theme icon-theme--gray icon-theme--small" data-toggle="dropdown" aria-expanded="true" role="button" aria-haspopup="true" data-delay="0" href="#" title="<?php esc_attr_e('View your shopping cart', 'univero'); ?>">
            <span class="text-skin cart-icon">
            	<span class="count"><?php echo sprintf($woocommerce->cart->cart_contents_count); ?></span>
                <i class="univero-shop3"></i>
            </span>
        </a>            
        <div class="dropdown-menu">
            <div class="widget_shopping_cart_content">
            </div>
        </div>
    </div>
</div>