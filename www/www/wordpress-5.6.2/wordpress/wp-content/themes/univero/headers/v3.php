<header id="ninzio-header" class="site-header ninzio-header header-v5 hidden-sm hidden-xs <?php echo univero_get_config('enable_header_mobile_1024', false) ? 'hidden-1024' : ''; ?>" role="banner">
    <div class="headertop <?php echo (univero_get_config('keep_header') ? 'main-sticky-header-wrapper' : ''); ?>">
        <div class="header-main clearfix <?php echo (univero_get_config('keep_header') ? 'main-sticky-header' : ''); ?>">
            <div class="container">
                <div class="header-inner">
                    <div class="header-inner-wrapper">
                    <!-- LOGO -->
                        <div class="header-logo">
                            <div class="logo-in-theme text-center">
                                <?php get_template_part( 'template-parts/logo/logo-v5' ); ?>
                            </div>
                        </div>
                        <div class="header-meta">
                            <?php if ( has_nav_menu( 'primary' ) ) : ?>
                                <div class="main-menu">
                                    <nav data-duration="400" class="hidden-xs hidden-sm ninzio-megamenu slide animate navbar" role="navigation">
                                    <?php   $args = array(
                                            'theme_location' => 'primary',
                                            'container_class' => 'collapse navbar-collapse',
                                            'menu_class' => 'nav navbar-nav megamenu',
                                            'fallback_cb' => '',
                                            'menu_id' => 'primary-menu',
                                            'walker' => new Univero_Nav_Menu()
                                        );
                                        wp_nav_menu($args);
                                    ?>
                                    </nav>
                                </div>
                            <?php endif; ?>
                        </div>
                        <div class="header-actions">
                            <?php get_template_part( 'template-parts/productsearchform-v1' ); ?>
                            <?php if ( univero_get_config('show_woo_cart') && defined('UNIVERO_WOOCOMMERCE_ACTIVED') && UNIVERO_WOOCOMMERCE_ACTIVED ): ?>
                                <!-- Setting -->
                                <div class="top-cart">
                                    <?php get_template_part( 'woocommerce/cart/mini-cart-button-v1' ); ?>                                   
                                </div>
                            <?php endif; ?>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>    
</header>