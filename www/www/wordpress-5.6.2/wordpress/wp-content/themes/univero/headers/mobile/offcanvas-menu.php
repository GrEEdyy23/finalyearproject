<div id="ninzio-mobile-menu" class="ninzio-offcanvas"> 
    <div class="ninzio-offcanvas-body">
        
        <nav class="navbar navbar-offcanvas navbar-static" role="navigation">
            <?php
                $args = array(
                    'theme_location' => 'primary',
                    'container_class' => 'navbar-collapse navbar-offcanvas-collapse',
                    'menu_class' => 'nav navbar-nav',
                    'fallback_cb' => '',
                    'menu_id' => 'main-mobile-menu',
                    'walker' => new Univero_Mobile_Menu()
                );
                wp_nav_menu($args);
            ?>
        </nav>
        
        <?php get_template_part( 'template-parts/productsearchform' ); ?>

        <?php if ( univero_get_config('show_woo_cart') && defined('UNIVERO_WOOCOMMERCE_ACTIVED') && UNIVERO_WOOCOMMERCE_ACTIVED ): ?>
            <div class="top-cart">
                <?php get_template_part( 'woocommerce/cart/mini-cart-button-mobile' ); ?>
            </div>
        <?php endif; ?>
        
        <?php if ( is_active_sidebar( 'topbar-right-sidebar' ) ) : ?>
            <!--
            <div class="widget-area" role="complementary">
                <?php dynamic_sidebar( 'topbar-right-sidebar' ); ?>
            </div>
        -->
        <?php endif; ?>

        <ul class="mobile-social-links list-inline list-unstyled">
            <?php
                $social_links = univero_get_config('header_topbar_social_link');
                $social_icons = univero_get_config('header_topbar_social_icon');
                if ( !empty($social_links) ) {
                    foreach ($social_links as $key => $value) {
                        ?>
                        <li class="social-item"><a href="<?php echo esc_url($value); ?>"><i class="<?php echo esc_attr($social_icons[$key]); ?>"></i></a></li>
                        <?php
                    }
                }
            ?>
        </ul>
    </div>
</div>