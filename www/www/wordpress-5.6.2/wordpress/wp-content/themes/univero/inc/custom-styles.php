<?php

if ( !function_exists ('univero_custom_styles') ) {
	function univero_custom_styles() {
		global $post;	
		
		ob_start();	
		?>
		
		
			<?php if ( univero_get_config('main_color') != "" ) : ?>
						
				.bg-theme,#ninzio-header.header-v5,			
				.ninzio-header.header-v6,
				.ninzio-header.header-v7 .cart-icon .count,
				.ninzio-header.header-v8 .cart-icon .count,
				.ninzio-header.header-v9 .cart-icon .count,
				.ninzio-header .cart-icon .count,
				.ninzio-header .top-cart .mini-cart .cart-icon .count,
				.search-form .dropdown-menu form .button-search,
				.icon-theme.icon-theme--success,
				.header-v3 .navbar-nav>li>a:after,
				.header-v4 .navbar-nav>li>a:before,
				.header-v5 .navbar-nav>li.active>a,
				.header-v5 .navbar-nav>li:hover>a,
				.header-v5 .navbar-nav>li:focus>a,
				.header-v5 .navbar-nav>li:active>a,
				.header-v7 .navbar-nav>li>a:before,
				.widget-testimonials.style2 .info:after,
				.widget-features-box.grid.style2 .feature-box,
				.edr-course .edr-course__price,
				#back-to-top,
				.btn-theme,
				#tribe-events .tribe-events-button,
				#tribe-events .tribe-events-button:hover,
				#tribe_events_filters_wrapper input[type="submit"],
				.tribe-events-button,
				.tribe-events-button.tribe-active:hover,
				.tribe-events-button.tribe-inactive,
				.tribe-events-button:hover,
				.tribe-events-calendar td.tribe-events-present div[id*="tribe-events-daynum-"],
				.wpb-js-composer .vc_tta.vc_general .vc_tta-panel.vc_active .vc_tta-controls-icon-position-left .vc_tta-controls-icon,
				.woocommerce #respond input#submit,
				.woocommerce a.button,
				.woocommerce button.button,
				.woocommerce input.button,
				.woocommerce-account .woocommerce-MyAccount-navigation .woocommerce-MyAccount-navigation-link.is-active>a,
				.woocommerce-account .woocommerce-MyAccount-navigation .woocommerce-MyAccount-navigation-link:hover>a,
				.woocommerce-account .woocommerce-MyAccount-navigation .woocommerce-MyAccount-navigation-link:active>a,
				.woocommerce-account .woocommerce-MyAccount-navigation .woocommerce-MyAccount-navigation-link.is-active>a::after,
				.woocommerce-account .woocommerce-MyAccount-navigation .woocommerce-MyAccount-navigation-link:hover>a::after,
				.woocommerce-account .woocommerce-MyAccount-navigation .woocommerce-MyAccount-navigation-link:active>a::after,
				.woocommerce #respond input#submit:hover,
				.woocommerce #respond input#submit:active,
				.woocommerce a.button:hover,
				.woocommerce a.button:active,
				.woocommerce button.button:hover,
				.woocommerce button.button:active,
				.woocommerce input.button:hover,
				.btn-theme:hover,
				.btn-theme:focus,
				.btn-theme:active,
				.btn-theme.active,
				.open>.btn-theme.dropdown-toggle,
				.woocommerce input.button:active,
				.vc_progress_bar .vc_general.vc_single_bar.vc_progress-bar-color-bar_green .vc_bar,
				.widget-features-box.grid.style3 .feature-box.featured .fbox-icon .fbox-icon-inner,
				.tribe-events-calendar td.tribe-events-present div[id*="tribe-events-daynum-"]>a,
				.ninzio-teacher-inner .socials a:hover,
				.ninzio-teacher-inner .socials a:focus,
				.ninzio-teacher-inner .socials a:active,
				.mfp-gallery .mfp-content button.mfp-close,
				.owl-controls .owl-dots .owl-dot.hover,
				.owl-controls .owl-dots .owl-dot.active,
				.sidebar .widget .widget-title::after,
				.ninzio-sidebar .widget .widget-title::after,
				.ninzio-pagination a:hover,
				.ninzio-pagination a:focus,
				.ninzio-pagination a:active,
				.ninzio-pagination span:active::after,
				.ninzio-pagination span:focus::after,
				.ninzio-pagination span:hover::after,
				.ninzio-pagination a:active::after,
				.ninzio-pagination a:focus::after,
				.ninzio-pagination a:hover::after,
				.ninzio-pagination span.current,
				.ninzio-pagination a.current,
				.detail-post .edr_course .info-meta .edr-buy-widget__link,
				.edr-membership-wrapper:hover .edr-membership-buy-link,
				.widget-membership.active .edr-membership-wrapper .edr-buy-widget__link,
				.widget-membership.active .edr-membership-wrapper .edr-membership-buy-link,
				.widget.widget-gallery.grid .gallery-item.style2:hover .gallery-item-title,
				.widget.widget-gallery.grid .gallery-item.style2:focus .gallery-item-title,
				.widget.widget-gallery.grid .gallery-item.style2:active .gallery-item-title,
				.widget-course-categories .category-wrapper .label,
				.course-lesson-sidebar .forward,
				.header-v2 .navbar-nav>li>a::before,
				.btn-outline.btn-success:hover,
				.btn-outline.btn-success:focus,
				.btn-outline.btn-success:active,
				.widget-action.center-white,
				.quickview-container .mfp-close,
				.widget-counters .widget-counters-item.style1,
				.list-tab-v1>li>.active,
				.list-tab-v1>li>.active:after,
				.list-tab-v1>li>a:hover,
				.list-tab-v1>li>a:hover:after,
				.list-tab-v1>li>a:focus:after,
				.list-tab-v1>li>a:active:after,			
				.widget-features-box .fbox-icon .fbox-icon-inner:hover,	
				.wpb-js-composer .vc_tta-container .vc_tta-color-green.vc_tta-style-modern .vc_tta-tab > a:hover .vc_tta-icon, 
				.wpb-js-composer .vc_tta-container .vc_tta-color-green.vc_tta-style-modern .vc_tta-tab > a:focus .vc_tta-icon, 
				.wpb-js-composer .vc_tta-container .vc_tta-color-green.vc_tta-style-modern .vc_tta-tab > a:active .vc_tta-icon,

				.btn-success:hover, .btn-success:focus, .btn-success:active, .btn-success.active, .open > .btn-success.dropdown-toggle, .btn-success,
				#ninzio-header.header-v1 .navbar-nav.megamenu, #ninzio-header.header-v1 .ninzio-megamenu, #ninzio-header.header-v1 .ninzio-megamenu:before, #ninzio-header.header-v1 .ninzio-megamenu:after,
				.navbar-nav .dropdown-menu > li.open > a:before, 
				.navbar-nav .dropdown-menu > li.active > a:before,
				.navbar-nav > li > ul > li > a:before,
				#ninzio-header.header-v5,
				.ninzio-header.header-v5 .sticky-header,
				#ninzio-header-mobile .navbar-offcanvas .navbar-nav li.active > a,
				.list-info .list-info-item .list-info-icon i::before
				{
					background-color: <?php echo esc_html( univero_get_config('main_color') ) ?>;
				}

				/* setting color*/
				.widget-features-box .fbox-icon .fbox-icon-inner,
				.widget-features-box.grid.style3 .feature-box.featured .ourservice-heading,
				.tagcloud a:focus,
				.tagcloud a:hover,
				a:hover,a:focus,.icon-theme,
				.course-lesson-sidebar .edr-lessons li:hover a,
				.course-lesson-sidebar .edr-lessons li.active a,
				.btn.btn-outline:hover,
				.edr-membership .btn-outline.edr-buy-widget__link:hover,
				.edr-membership .btn-outline.edr-membership-buy-link:hover,
				.edr_membership .btn-outline.edr-buy-widget__link:hover,
				.edr_membership .btn-outline.edr-membership-buy-link:hover,
				.btn.btn-outline:active,
				.edr-membership .btn-outline.edr-buy-widget__link:active,
				.edr-membership .btn-outline.edr-membership-buy-link:active,
				.edr_membership .btn-outline.edr-buy-widget__link:active,
				.edr_membership .btn-outline.edr-membership-buy-link:active,
				.btn.btn-outline:focus,
				.edr-membership .btn-outline.edr-buy-widget__link:focus,
				.edr-membership .btn-outline.edr-membership-buy-link:focus,
				.edr_membership .btn-outline.edr-buy-widget__link:focus,
				.edr_membership .btn-outline.edr-membership-buy-link:focus,
				.navbar-nav.megamenu .dropdown-menu>li>a.active,
				.navbar-nav.megamenu .dropdown-menu>li>a:hover,
				.navbar-nav.megamenu .dropdown-menu>li>a:active,
				.wpb-js-composer .vc_tta-container .vc_tta-color-green.vc_tta-style-modern .vc_tta-tab > a .vc_tta-icon,
				.wpb-js-composer .vc_tta-container .vc_tta-color-green.vc_tta-style-modern .vc_tta-tab.vc_active > a .vc_tta-icon,
				.wpb-js-composer .vc_tta-container .vc_tta-color-green.vc_tta-style-modern .vc_tta-tab > a:hover .vc_tta-title-text, 
				.wpb-js-composer .vc_tta-container .vc_tta-color-green.vc_tta-style-modern .vc_tta-tab > a:focus .vc_tta-title-text, 
				.wpb-js-composer .vc_tta-container .vc_tta-color-green.vc_tta-style-modern .vc_tta-tab > a:active .vc_tta-title-text
				{
					color: <?php echo esc_html( univero_get_config('main_color') ) ?>;
				}
				.text-theme,.owl-carousel .owl-controls .owl-nav .owl-prev:hover, .owl-carousel .owl-controls .owl-nav .owl-next:hover,
				.edr-course .edr-course__title a:hover, .edr-course .edr-course__title a:focus, .edr-course .edr-course__title a:active,
				.widget-counters .widget-counters-item.style2 .font-icon,.entry-title a:hover,.readmore:hover,.btn-outline.btn-success,
				form.login .input-submit ~ span.lost_password a, form.register .input-submit ~ span.lost_password a,.top-cart .name a:hover,
				#ninzio-header.header-v2 #ninzio-topbar .list-social > li > a:hover,#ninzio-header.header-v2 .navbar-nav > li > a:hover,
				.woocommerce div.product p.price, .woocommerce div.product span.price,.detail-post .entry-tag a:hover,.ninzio-header .list-social > li > a:hover,			
				.detail-post .edr_course .info-meta .course-price .edr-buy-widget__price,.edr-membership-wrapper .entry-description ul > li span::before
				{
					color: <?php echo esc_html( univero_get_config('main_color') ) ?> !important;
				}
				/* setting border color*/
				.woocommerce #respond input#submit, .woocommerce a.button, .woocommerce button.button, .woocommerce input.button,.btn-outline.btn-success,.btn-outline.btn-success:hover,.btn-outline.btn-success:focus,.btn-outline.btn-success:active,
				.tagcloud a:focus, .tagcloud a:hover,#review_form .comment-form input:focus, #review_form .comment-form textarea:focus,.detail-post .entry-tag a:hover,.icon-theme.icon-theme--success,
				.btn.btn-outline:hover, .edr-membership .btn-outline.edr-buy-widget__link:hover, .edr-membership .btn-outline.edr-membership-buy-link:hover, .edr_membership .btn-outline.edr-buy-widget__link:hover, 
				.edr_membership .btn-outline.edr-membership-buy-link:hover, .btn.btn-outline:active, .edr-membership .btn-outline.edr-buy-widget__link:active, .edr-membership .btn-outline.edr-membership-buy-link:active, 
				.edr_membership .btn-outline.edr-buy-widget__link:active, .edr_membership .btn-outline.edr-membership-buy-link:active, .btn.btn-outline:focus, .edr-membership .btn-outline.edr-buy-widget__link:focus, 
				.edr-membership .btn-outline.edr-membership-buy-link:focus, .edr_membership .btn-outline.edr-buy-widget__link:focus, .edr_membership .btn-outline.edr-membership-buy-link:focus,
				.header-v8 .navbar-nav > li:hover > a, .header-v8 .navbar-nav > li:focus > a, .header-v8 .navbar-nav > li:active > a, .header-v8 .navbar-nav > li.active > a,.edr-membership-wrapper:hover,
				.header-v9 .navbar-nav > li.open > a, .header-v9 .navbar-nav > li:hover > a, .header-v9 .navbar-nav > li:focus > a, .header-v9 .navbar-nav > li:active > a, .header-v9 .navbar-nav > li.active > a,
				.border-theme,.owl-carousel .owl-controls .owl-nav .owl-prev:hover, .owl-carousel .owl-controls .owl-nav .owl-next:hover,.readmore:hover,.form-control:focus,
				.edr-membership-wrapper:hover .edr-membership-buy-link,.widget-membership.active .edr-membership-wrapper,.widget-membership.active .edr-membership-wrapper .edr-buy-widget__link, 
				.widget-membership.active .edr-membership-wrapper .edr-membership-buy-link,.widget.widget-gallery.grid .gallery-item.style2:hover .gallery-item-title, .widget.widget-gallery.grid .gallery-item.style2:focus .gallery-item-title, 
				.widget.widget-gallery.grid .gallery-item.style2:active .gallery-item-title,.wpcf7-form .form-control:focus,.woocommerce form .form-row input.input-text:focus, .woocommerce form .form-row textarea:focus,
				.archive-shop div.product .information .cart .quantity input.qty:focus,.ninzio-pagination span.current, .ninzio-pagination a.current,#commentform .form-control:focus,
				#add_payment_method table.cart td.actions .coupon .input-text:focus, .woocommerce-cart table.cart td.actions .coupon .input-text:focus, .woocommerce-checkout table.cart td.actions .coupon .input-text:focus,
				.woocommerce #respond input#submit:hover, .woocommerce #respond input#submit:active, .woocommerce a.button:hover, .woocommerce a.button:active,				
				.woocommerce button.button:hover, .woocommerce button.button:active, .woocommerce input.button:hover, .woocommerce input.button:active,
				.quickview-container .summary .quantity input.qty:focus,				
				.wpb-js-composer .vc_tta-container .vc_tta-color-green.vc_tta-style-modern .vc_tta-tab > a:hover .vc_tta-icon, 
				.wpb-js-composer .vc_tta-container .vc_tta-color-green.vc_tta-style-modern .vc_tta-tab > a:focus .vc_tta-icon, 
				.wpb-js-composer .vc_tta-container .vc_tta-color-green.vc_tta-style-modern .vc_tta-tab > a:active .vc_tta-icon,
				.btn-theme:hover, .btn-theme:focus, .btn-theme:active, .btn-theme.active, .open > .btn-theme.dropdown-toggle,.btn-success,.btn-success:hover, .btn-success:focus, .btn-success:active, .btn-success.active, .open > .btn-success.dropdown-toggle				
				{
					border-color: <?php echo esc_html( univero_get_config('main_color') ) ?> !important;
				}
				.ninzio-page-loading #loader:after,.ninzio-page-loading #loader:before,.ninzio-page-loading #loader{
					border-color: <?php echo esc_html( univero_get_config('main_color') ) ?> transparent transparent !important;
				}
				.edr-course .edr-course__price.free-label:after,
				.edr-course .edr-course__price .letter-0:after,
				.navbar-nav.megamenu .dropdown-menu,
				.post-grid-v2:after,.woocommerce .percent-sale .percent-sale-iiner::before, .woocommerce span.onsale .percent-sale-iiner::before
				{
					border-top-color: <?php echo esc_html( univero_get_config('main_color') ) ?> !important;
				}
				.edr-course .edr-course__price.free-label:before,
				.edr-course .edr-course__price .letter-0:before,
				.woocommerce .percent-sale .percent-sale-iiner::after, .woocommerce span.onsale .percent-sale-iiner::after
				{
					border-bottom-color: <?php echo esc_html( univero_get_config('main_color') ) ?> !important;
				}
				.post-grid-v2:hover:after{
					box-shadow:0 -2px 0 0 <?php echo esc_html( univero_get_config('main_color') ) ?> inset;
				}
				.form-control:focus,#commentform .form-control:focus,
				.archive-shop div.product .information .cart .quantity input.qty:focus,
				.wpcf7-form .form-control:focus,.woocommerce form .form-row input.input-text:focus, 
				.woocommerce form .form-row textarea:focus,.quickview-container .summary .quantity input.qty:focus,
				#add_payment_method table.cart td.actions .coupon .input-text:focus, 
				.woocommerce-cart table.cart td.actions .coupon .input-text:focus, 
				.woocommerce-checkout table.cart td.actions .coupon .input-text:focus,
				#review_form .comment-form input:focus, #review_form .comment-form textarea:focus{
					box-shadow: 0 0 0 1px <?php echo esc_html( univero_get_config('main_color') ) ?> inset;					 
				}
			<?php endif; ?>

			/* check second color */ 
			<?php if ( univero_get_config('second_color') != "" ) : ?>

				/* seting background main */					
				.bg-theme-second,.btn-theme-second,.widget-counters .widget-counters-item.featured,
				.widget-counters .widget-counters-item.style1:hover, .widget-counters .widget-counters-item.style1:focus, .widget-counters .widget-counters-item.style1:active,
				.widget-features-box.grid.style2 .feature-box.featured, .widget-features-box.grid.style2 .feature-box:hover, .widget-features-box.grid.style2 .feature-box:active, .widget-features-box.grid.style2 .feature-box:focus
				{
					background-color: <?php echo esc_html( univero_get_config('second_color') ) ?> !important;
				}
				/* setting color*/
				.edr-course-list-simple .edr-course__price,
				.text-second,.second-color
				{
					color: <?php echo esc_html( univero_get_config('second_color') ) ?> !important;
				}
				/* setting border color*/
				.btn-theme-second
				{
					border-color: <?php echo esc_html( univero_get_config('second_color') ) ?>;
				}

			<?php endif; ?>

			/* check accent color */ 
			<?php if ( univero_get_config('accent_color') != "" ) : ?>

				/* seting background main */					
				#ninzio-header.header-v4,
				#ninzio-header.header-v4 .ninzio-megamenu,
				#ninzio-header.header-v4 .navbar-nav.megamenu,							
				#ninzio-topbar,.ninzio-header.header-v5 .top-cart .mini-cart .cart-icon .count
				{
					background-color: <?php echo esc_html( univero_get_config('accent_color') ) ?>;
				}
				/* setting color*/
				.widget.widget-text-heading .title,				
				#ninzio-footer .widget-newletter .widgettitle, 
				#ninzio-footer .widget-newletter .widget-title				
				{
					color: <?php echo esc_html( univero_get_config('accent_color') ) ?>;
				}
				/* setting border color*/
				.btn-theme-accent
				{
					border-color: <?php echo esc_html( univero_get_config('accent_color') ) ?>;
				}

			<?php endif; ?>

			
			/* button background */ 
			<?php if ( univero_get_config('button_color') != "" ) : ?>

				/* seting background main */
				.wpcf7-form .wpcf7-submit,				
				.btn-success,			
				.btn-theme,
				.more-link
				{
					background: <?php echo esc_html( univero_get_config('button_color') ) ?> !important;
				}
				/* setting border color*/		
				.wpcf7-form .wpcf7-submit,		
				.btn-theme,
				.btn-success,
				.more-link
				{
					border-color: <?php echo esc_html( univero_get_config('button_color') ) ?> !important;
				}

			<?php endif; ?>
			<?php if ( univero_get_config('button_text_color') != "" ) : ?>

				/* seting background main */
				.wpcf7-form .wpcf7-submit,
				.btn-theme,
				.btn-success,
				.more-link
				{
					color: <?php echo esc_html( univero_get_config('button_text_color') ) ?> !important;
				}

			<?php endif; ?>
			/* button background hover */ 
			<?php if ( univero_get_config('button_hover_color') != "" ) : ?>

				/* seting background main */
				.wpcf7-form .wpcf7-submit:hover,.wpcf7-form .wpcf7-submit:focus,.wpcf7-form .wpcf7-submit:active,
				.btn-theme:hover,.btn-theme:active,.btn-theme:focus,
				.more-link:hover, .more-link:focus, .more-link:active,
				.btn-success:hover, .btn-success:focus, .btn-success:active				
				{
					background: <?php echo esc_html( univero_get_config('button_hover_color') ) ?> !important;
				}
				/* setting border color*/
				.wpcf7-form .wpcf7-submit:hover,.wpcf7-form .wpcf7-submit:focus,.wpcf7-form .wpcf7-submit:active,
				.btn-theme:hover,.btn-theme:active,.btn-theme:focus,
				.more-link:hover, .more-link:focus, .more-link:active,
				.btn-success:hover, .btn-success:focus, .btn-success:active	
				{
					border-color: <?php echo esc_html( univero_get_config('button_hover_color') ) ?> !important;
				}

			<?php endif; ?>
			<?php if ( univero_get_config('button_hover_text_color') != "" ) : ?>

				/* seting background main */
				.wpcf7-form .wpcf7-submit:hover,.wpcf7-form .wpcf7-submit:focus,.wpcf7-form .wpcf7-submit:active,
				.btn-theme:hover,.btn-theme:active,.btn-theme:focus,
				.more-link:hover, .more-link:focus, .more-link:active,
				.btn-success:hover, .btn-success:focus, .btn-success:active	
				{
					color: <?php echo esc_html( univero_get_config('button_hover_text_color') ) ?> !important;
				}

			<?php endif; ?>
			/* Body Font */
			<?php
				$main_font = univero_get_config('main_font');
				if ( !empty($main_font) ) :
			?>
				/* seting background main */
				body,p,				
				.ui-autocomplete.ui-widget-content li .name a,
				.counters .counter,
				.widget-features-box.default .ourservice-heading,
				.widget-features-box.default .readmore,
				#reviews .title-info,
				.entry-tags-list,
				#commentform .input-title,
				.post-grid-v3 .date,
				.ninzio-sidebar .post-widget .blog-title,
				.ninzio-sidebar .post-widget h6,
				.widget-quicklink-menu .quicklink-heading,
				.terms-list,
				.edr-course .edr-teacher .description .author-title,
				.edr-course-list-simple .edr-course__price,
				.widget-title-special .edr-buy-widget__link,
				.widget-special-features .widget-price .price-label,
				.widget-special-features .widget-price .edr-buy-widget__price,
				.detail-post .edr_course .course-socials-bookmark .course-bookmark .ninzio-bookmark-add,
				.detail-post .edr_course .course-socials-bookmark .course-bookmark .ninzio-bookmark-added,
				.detail-post .edr_course .course-socials-bookmark .course-bookmark .ninzio-bookmark-not-login,
				.list-instructors .redmore,
				.send-message-instructor-form input:not(.btn),
				.send-message-instructor-form textarea,
				.single-tribe_events .event-meta h3,
				.single-tribe_events .times>div
				{
					<?php if ( isset($main_font['font-family']) && $main_font['font-family'] ) { ?>
						font-family: <?php echo esc_html( $main_font['font-family'] ) ?>;
					<?php } ?>
				}
				
			<?php endif; ?>

			<?php
				$second_font = univero_get_config('second_font');
				if ( !empty($second_font) ) :
			?>
				/* seting heading font */
				.form-control,
				.widget .widget-title,
				.widget .widgettitle,
				.widget .widget-heading,
				.list-tab-v1>li>a,
				h1, h2, h3, h4, h5, h6, .h1, .h2, .h3, .h4, .h5, .h6,
				.tagcloud a,
				.ninzio-footer ul>li>a,
				.widget-counters .counter-wrap,
				.navbar-nav>li>a,
				.navbar-nav .dropdown-menu>li>a,
				.ninzio-header .contact-information .box-content,
				.ninzio-teacher-inner .job,
				.testimonials-body .info,
				.title,.name,
				.navbar-nav .dropdown-menu .menu-megamenu-container ul>li>a,
				.edr-course .edr-teacher,
				.widget-course-categories .category-wrapper .label,
				.widget-course-categories .category-wrapper .title,
				.widget-counters .counter-wrap,
				.widget-counters .title,
				.edr-course-list .entry-info li span,
				.ninzio-teacher-inner .job,
				#course-review .comment .ninzio-author,
				.course-features,
				.detail-post .edr_course .info-meta .title,
				.edr-membership .edr-buy-widget__link,
				.edr-membership .edr-membership-buy-link,
				.edr_membership .edr-buy-widget__link,
				.edr_membership .edr-membership-buy-link,
				.edr-membership-wrapper .edr-membership__price,
				.tribe-events-list.event-list .entry-date-wrapper,
				.tribe-events-list.event-grid .entry-date-wrapper,
				.detail-post .edr_course .info-meta .course-review .rating-count,
				.edr-course-grid .entry-info .edr-registered,
				.edr-course-grid .entry-info .edr-comments,
				.tabs-v1 .nav-tabs li,
				.nav.tabs-product1>li>a,
				.post-navigation .navi,
				.section-contact .title-contact,
				.ninzio-pagination,
				.entry-date,
				.edr-course .edr-teacher,
				.vc_progress_bar .vc_general.vc_single_bar.vc_progress-bar-color-bar_green .vc_label,
				.ninzio-topbar .wpml-ls-legacy-dropdown .wpml-ls-sub-menu > li > a,
				#review_form .comment-form input,
				#review_form .comment-form textarea,
				.edr-course .edr-course__price,
				.ninzio-header .ninzio-search-form .ninzio-search,
				.navbar-nav>li>a,
				.newletters-1 .widgettitle,
				.ninzio-header .top-cart .mini-cart .cart-icon .count,
				.navbar-nav .dropdown-menu>li>a,
				.ninzio-header .contact-information .box-content,
				.navbar-nav .dropdown-menu .menu-megamenu-container ul>li>a,
				.header-mobile .top-cart .ninzio-topcart-mobile .mini-cart,
				.navbar-offcanvas .navbar-nav li>a,
				.header-mobile .ninzio-search-form .ninzio-search,
				.page-404 .big-font,
				.widget-newletter input,
				.detail-post .ninzio-social-share strong,
				.detail-post .entry-tag strong,
				.detail-post .entry-tag a
				{
					<?php if ( isset($second_font['font-family']) && $second_font['font-family'] ) { ?>
						font-family: <?php echo esc_html( $second_font['font-family'] ) ?> !important;
					<?php } ?>
				}
			<?php endif; ?>


			<?php
				$button_font = univero_get_config('button_font');
				if ( !empty($button_font) ) :
			?>
				/* seting Button Font */
				.btn,.button, button, input[type=submit], button[type=submit],.edr-buy-widget__link,.readmore,.rev-btn,
				.wpb-js-composer .vc_tta.vc_general .vc_tta-panel-title,
				#review_form .comment-form input#submit,
				.comment-list .comment-reply-link,
				.detail-post .edr_course .info-meta .edr-buy-widget__link,
				#tribe-events .tribe-events-button,
				.tribe-events-button,.edr-membership-buy-link
				{
					<?php if ( isset($button_font['font-family']) && $button_font['font-family'] ) { ?>
						font-family: <?php echo esc_html( $button_font['font-family'] ) ?> !important;
					<?php } ?>
				}
			<?php endif; ?>

	<?php
		$content = ob_get_clean();
		$content = str_replace(array("\r\n", "\r"), "\n", $content);
		$lines = explode("\n", $content);
		$new_lines = array();
		foreach ($lines as $i => $line) {
			if (!empty($line)) {
				$new_lines[] = trim($line);
			}
		}
		
		return implode($new_lines);
	}
}

?>
<?php add_action( 'wp_head', 'univero_custom_styles', 99 ); ?>