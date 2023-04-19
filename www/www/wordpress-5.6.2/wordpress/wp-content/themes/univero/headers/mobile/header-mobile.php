<div id="ninzio-header-mobile" class="header-mobile hidden-lg hidden-md <?php echo univero_get_config('enable_header_mobile_1024', false) ? 'show-1024' : ''; ?> clearfix">
    <div class="container">
        <div class="header-mobile-top clearfix">
            <div class="logo-mobile-wrapper">
                <?php $logo = univero_get_config('media-mobile-logo'); ?>

                <?php if( isset($logo['url']) && !empty($logo['url']) ): ?>
                    <div class="logo">
                        <a href="<?php echo esc_url( home_url( '/' ) ); ?>" >
                            <img src="<?php echo esc_url( $logo['url'] ); ?>" alt="<?php bloginfo( 'name' ); ?>">
                        </a>
                    </div>
                <?php else: ?>
                    <div class="logo logo-theme">
                        <a href="<?php echo esc_url( home_url( '/' ) ); ?>" >
                            <img src="<?php echo esc_url_raw( get_template_directory_uri().'/images/logo-mobile.png'); ?>" alt="<?php bloginfo( 'name' ); ?>">
                        </a>
                    </div>
                <?php endif; ?>
            </div>
            <div class="site-mobile-actions">
                <div class="active-mobile">
                    <button data-toggle="offcanvas" class=" icon-theme icon-theme--large icon-theme--gray btn-offcanvas btn-toggle-canvas offcanvas" type="button">
                       <i class="univero-menu"></i>
                    </button>
                </div>
                <?php if ( has_nav_menu( 'topmenu' ) ) { ?>
                    <div class="dropdown">
                        <a class="account icon-theme icon-theme--large icon-theme--gray" href="#" data-toggle="dropdown" aria-expanded="true" role="button" aria-haspopup="true" data-delay="0">
                            <i class="univero-user1"></i>
                        </a>
                        <div class="dropdown-menu dropdown-menu-right">
                            <?php
                                $args = array(
                                    'theme_location'  => 'topmenu',
                                    'container_class' => '',
                                    'menu_class'      => 'menu-topbar list-unstyled'
                                );
                                wp_nav_menu($args);
                            ?>
                        </div>
                    </div>
                <?php } ?>
            </div>            
        </div>
    </div>
    <?php get_template_part( 'headers/mobile/offcanvas-menu' ); ?>
</div>