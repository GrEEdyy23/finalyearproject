<header id="ninzio-header" class="site-header ninzio-header header-v4 hidden-sm hidden-xs <?php echo univero_get_config('enable_header_mobile_1024', false) ? 'hidden-1024' : ''; ?>" role="banner">
    <div class="headertop <?php echo (univero_get_config('keep_header') ? 'main-sticky-header-wrapper' : ''); ?>">
        <div class="header-main <?php echo (univero_get_config('keep_header') ? 'main-sticky-header' : ''); ?>">
            <div class="container">
                <div class="header-inner">
                    <div class="header-inner-wrapper clearfix text-center">                       
                        <div class="header-meta pull-left">
                            <?php if ( has_nav_menu( 'primary_left' ) ) : ?>
                                <div class="main-menu">
                                    <nav data-duration="400" class="hidden-xs hidden-sm ninzio-megamenu slide animate navbar" role="navigation">
                                    <?php   $args = array(
                                            'theme_location' => 'primary_left',
                                            'container_class' => 'collapse navbar-collapse',
                                            'menu_class' => 'nav navbar-nav megamenu',
                                            'fallback_cb' => '',
                                            'menu_id' => 'primary_left',
                                            'walker' => new Univero_Nav_Menu()
                                        );
                                        wp_nav_menu($args);
                                    ?>
                                    </nav>
                                </div>
                            <?php endif; ?>
                        </div>
                        <!-- LOGO -->
                        <div class="header-logo">
                            <div class="logo-in-theme text-center">                                
                                <?php get_template_part( 'template-parts/logo/logo-v7' ); ?>
                            </div>
                        </div>
                        <div class="header-actions pull-right">
                            <div class="header-meta-2 pull-left">
                                <?php if ( has_nav_menu( 'primary_right' ) ) : ?>
                                    <div class="main-menu">
                                        <nav data-duration="400" class="hidden-xs hidden-sm ninzio-megamenu slide animate navbar" role="navigation">
                                        <?php   $args = array(
                                                'theme_location' => 'primary_right',
                                                'container_class' => 'collapse navbar-collapse',
                                                'menu_class' => 'nav navbar-nav megamenu',
                                                'fallback_cb' => '',
                                                'menu_id' => 'primary_right',
                                                'walker' => new Univero_Nav_Menu()
                                            );
                                            wp_nav_menu($args);
                                        ?>
                                        </nav>
                                    </div>
                                <?php endif; ?>
                            </div>
                            <div class="pull-left header-box-right">
                                <div class="pull-left">
                                    <?php get_template_part( 'template-parts/productsearchform-v2' ); ?>
                                </div>
                                <?php if ( univero_get_config('show_woo_cart') && defined('UNIVERO_WOOCOMMERCE_ACTIVED') && UNIVERO_WOOCOMMERCE_ACTIVED ): ?>
                                    <!-- Setting -->
                                    <div class="top-cart pull-left">
                                        <?php get_template_part( 'woocommerce/cart/mini-cart-button-v2' ); ?>                                   
                                    </div>
                                <?php endif; ?>
                            </div>                           
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>    
</header>