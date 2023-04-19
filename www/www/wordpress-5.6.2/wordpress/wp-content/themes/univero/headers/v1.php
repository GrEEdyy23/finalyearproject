<header id="ninzio-header" class="site-header ninzio-header header-v1 hidden-sm hidden-xs <?php echo univero_get_config('enable_header_mobile_1024', false) ? 'hidden-1024' : ''; ?>" role="banner">
    <div id="ninzio-topbar" class="ninzio-topbar">
        <div class="container">
            <div class="left-topbar pull-left">
                <ul class="list-social pull-left list-unstyled list-inline">
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
                <?php if ( univero_get_config('top_info') ): ?>
                    <div class="pull-left top-information">
                        <?php echo wp_kses_post(univero_get_config('top_info')); ?>
                    </div>
                <?php endif; ?>
            </div>
            <?php if ( univero_get_config('show_login_register', true) ) { ?>
                <div class="pull-right">
                    
                    <?php if ( is_active_sidebar( 'topbar-right-sidebar' ) ) : ?>
                        <div class="widget-area pull-left" role="complementary">
                            <?php dynamic_sidebar( 'topbar-right-sidebar' ); ?>
                        </div>
                    <?php endif; ?>
                    <div class="accept-account pull-left">
                        <?php if( !is_user_logged_in() ){ ?>
                                <div class="login-topbar">
                                    <a class="login" href="<?php echo esc_url( get_permalink( get_option('woocommerce_myaccount_page_id') ) ); ?>" title="<?php esc_attr_e('Login','univero'); ?>"><?php esc_html_e('Login', 'univero'); ?></a>
                                    <a class="register" href="<?php echo esc_url( get_permalink( get_option('woocommerce_myaccount_page_id') ) ); ?>" title="<?php esc_attr_e('Sign Up','univero'); ?>"><?php esc_html_e('Register', 'univero'); ?></a>
                                </div>
                            <?php } else { ?>
                                <?php if ( has_nav_menu( 'topmenu' ) ) : ?>
                                    <div class="site-header-topmenu">                                    
                                        <div class="dropdown login-topbar clearfix">
                                            <a class="account" href="#" data-toggle="dropdown" aria-expanded="true" role="button" aria-haspopup="true" data-delay="0">
                                                <?php esc_html_e( 'My Account', 'univero' ); ?>
                                                <span class="icon univero-arrow-bottom"></span>
                                            </a>
                                            <div class="dropdown-menu dropdown-menu-right">
                                                <?php
                                                    $args = array(
                                                        'theme_location' => 'topmenu',
                                                        'container_class' => 'collapse navbar-collapse',
                                                        'menu_class' => 'nav navbar-nav',
                                                        'fallback_cb' => '',
                                                        'menu_id' => 'topmenu-menu',
                                                        'walker' => new Univero_Nav_Menu()
                                                    );
                                                    wp_nav_menu($args);
                                                ?>
                                            </div>
                                        </div>                                    
                                    </div>
                                <?php endif; ?>
                        <?php } ?>
                    </div>
                </div>
            <?php } ?>
        </div>
    </div>
    <div class="headertop <?php echo (univero_get_config('keep_header') ? 'main-sticky-header-wrapper' : ''); ?>">
        <div class="header-main clearfix <?php echo (univero_get_config('keep_header') ? 'main-sticky-header' : ''); ?>">
            <div class="container">
                <div class="header-container">
                    <div class="header-inner">
                        <div class="header-inner-wrapper clearfix">
                        <!-- LOGO -->
                            <div class="header-logo pull-left">
                                <div class="logo-in-theme text-center">
                                    <?php get_template_part( 'template-parts/logo/logo' ); ?>
                                </div>
                            </div>
                            <div class="header-meta pull-left">
                                <?php if ( univero_get_config('header_contact_info') ): ?>
                                    <div class="contact-information">
                                        <?php echo wp_kses_post(univero_get_config('header_contact_info')); ?>
                                    </div>
                                <?php endif; ?>
                            </div>
                            <div class="header-actions pull-right">
                                <?php get_template_part( 'template-parts/productsearchform' ); ?>
                                <?php if ( univero_get_config('show_woo_cart') && defined('UNIVERO_WOOCOMMERCE_ACTIVED') && UNIVERO_WOOCOMMERCE_ACTIVED ): ?>
                                    <!-- Setting -->
                                    <div class="top-cart">
                                        <?php get_template_part( 'woocommerce/cart/mini-cart-button' ); ?>
                                    </div>
                                <?php endif; ?>
                            </div>
                        </div>
                    </div>
                    <div class="header-menu">
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
                </div>
            </div>
        </div>
    </div>    
</header>