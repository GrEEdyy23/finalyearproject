<?php
/**
 * ReduxFramework Sample Config File
 * For full documentation, please visit: //docs.reduxframework.com/
 */

if (!class_exists('Univero_Redux_Framework_Config')) {

    class Univero_Redux_Framework_Config
    {
        public $args = array();
        public $sections = array();
        public $ReduxFramework;

        public function __construct()
        {
            if (!class_exists('ReduxFramework')) {
                return;
            }
            add_action('init', array($this, 'initSettings'), 10);
        }

        public function initSettings()
        {
            // Set the default arguments
            $this->setArguments();

            // Create the sections and fields
            $this->setSections();

            if (!isset($this->args['opt_name'])) { // No errors please
                return;
            }

            // If Redux is running as a plugin, this will remove the demo notice and links
            $this->ReduxFramework = new ReduxFramework($this->sections, $this->args);
        }

        public function setSections()
        {
            global $wp_registered_sidebars;
            $sidebars = array();

            if ( !empty($wp_registered_sidebars) ) {
                foreach ($wp_registered_sidebars as $sidebar) {
                    $sidebars[$sidebar['id']] = $sidebar['name'];
                }
            }
            $columns = array( '1' => esc_html__('1 Column', 'univero'),
                '2' => esc_html__('2 Columns', 'univero'),
                '3' => esc_html__('3 Columns', 'univero'),
                '4' => esc_html__('4 Columns', 'univero'),
                '5' => esc_html__('5 Columns', 'univero'),
                '6' => esc_html__('6 Columns', 'univero')
            );
            
            $general_fields = array();
            if ( !function_exists( 'wp_site_icon' ) ) {
                $general_fields[] = array(
                    'id' => 'media-favicon',
                    'type' => 'media',
                    'title' => esc_html__('Favicon Upload', 'univero'),
                    'desc' => esc_html__('', 'univero'),
                    'subtitle' => esc_html__('Upload a 16px x 16px .png or .gif image that will be your favicon.', 'univero'),
                );
            }
            $general_fields[] = array(
                'id' => 'preload',
                'type' => 'switch',
                'title' => esc_html__('Preload Website', 'univero'),
                'default' => true,
            );
            $general_fields[] = array(
                'id' => 'image_lazy_loading',
                'type' => 'switch',
                'title' => esc_html__('Image Lazy Loading', 'univero'),
                'default' => true,
            );
            // General Settings Tab
            $this->sections[] = array(
                'icon' => 'el-icon-cogs',
                'title' => esc_html__('General', 'univero'),
                'fields' => $general_fields
            );
            // Header
            $this->sections[] = array(
                'icon' => 'el el-website',
                'title' => esc_html__('Header', 'univero'),
                'fields' => array(
                    array(
                        'id' => 'media-logo',
                        'type' => 'media',
                        'title' => esc_html__('Logo Upload', 'univero'),
                        'subtitle' => esc_html__('Upload a .png or .gif image that will be your logo.', 'univero'),
                    ),
                    array(
                        'id' => 'media-mobile-logo',
                        'type' => 'media',
                        'title' => esc_html__('Mobile Logo Upload', 'univero'),
                        'subtitle' => esc_html__('Upload a .png or .gif image that will be your logo.', 'univero'),
                    ),
                    array(
                        'id' => 'header_type',
                        'type' => 'select',
                        'title' => esc_html__('Header Layout Type', 'univero'),
                        'subtitle' => esc_html__('Choose a header for your website.', 'univero'),
                        'options' => univero_get_header_layouts()
                    ),
                    array(
                        'id' => 'enable_header_mobile_1024',
                        'type' => 'switch',
                        'title' => esc_html__('Enable Mobile Header Screen 1024 ?', 'univero'),
                        'default' => false
                    ),
                    array(
                        'id' => 'keep_header',
                        'type' => 'switch',
                        'title' => esc_html__('Keep Header When Scroll Mouse', 'univero'),
                        'default' => false
                    ),
                    array(
                        'id' => 'show_woo_cart',
                        'type' => 'switch',
                        'title' => esc_html__('Show Woocommerce Cart', 'univero'),
                        'default' => true
                    ),
                    array(
                        'id' => 'show_search_form',
                        'type' => 'switch',
                        'title' => esc_html__('Show Search Form', 'univero'),
                        'default' => true
                    ),
                    array(
                        'id' => 'show_login_register',
                        'type' => 'switch',
                        'title' => esc_html__('Show Login/Register', 'univero'),
                        'default' => true
                    ),
                    array(
                        'id'         => 'header_topbar_social',
                        'type'       => 'repeater',
                        'title'      => esc_html__( 'Social Links', 'univero' ),
                        'fields'     => array(
                            array(
                                'id' => 'header_topbar_social_link',
                                'type' => 'text',
                                'title' => esc_html__('Link', 'univero'),
                            ),
                            array(
                                'id' => 'header_topbar_social_icon',
                                'type' => 'text',
                                'title' => esc_html__('Font Icon', 'univero'),
                            )
                        ),
                        'required' => array('header_type', '=', array('v1', 'v2', 'v3', 'v4'))
                    ),
                    array(
                        'id' => 'top_info',
                        'type' => 'editor',
                        'title' => esc_html__('TopBar Information', 'univero'),
                        'required' => array('header_type', '=', array('v1', 'v2', 'v3', 'v4'))
                    ),
                    array(
                        'id' => 'header_contact_info',
                        'type' => 'editor',
                        'title' => esc_html__('Header Contact Information', 'univero'),
                        'required' => array('header_type', '=', array('v1', 'v2'))
                    ),
                )
            );
            
            // Footer
            $this->sections[] = array(
                'icon' => 'el el-website',
                'title' => esc_html__('Footer', 'univero'),
                'fields' => array(
                    array(
                        'id' => 'footer_type',
                        'type' => 'select',
                        'title' => esc_html__('Footer Layout Type', 'univero'),
                        'subtitle' => esc_html__('Choose a footer for your website.', 'univero'),
                        'options' => univero_get_footer_layouts()
                    ),
                    array(
                        'id' => 'back_to_top',
                        'type' => 'switch',
                        'title' => esc_html__('Back To Top Button', 'univero'),
                        'subtitle' => esc_html__('Toggle whether or not to enable a back to top button on your pages.', 'univero'),
                        'default' => true,
                    ),
                )
            );

            // Blog settings
            $this->sections[] = array(
                'icon' => 'el el-pencil',
                'title' => esc_html__('Blog', 'univero'),
                'fields' => array(
                    

                )
            );
            // Archive Blogs settings
            $this->sections[] = array(
                'subsection' => true,
                'title' => esc_html__('Blog & Post Archives', 'univero'),
                'fields' => array(
                    array (
                        'id' => 'blogs_breadcrumbs_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Breadcrumbs Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'show_blogs_breadcrumbs',
                        'type' => 'switch',
                        'title' => esc_html__('Breadcrumbs', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'blogs_breadcrumb_title',
                        'type' => 'text',
                        'title' => esc_html__('Breadcrumbs Title', 'univero'),
                        'default' => 'Blog'
                    ),
                    array(
                        'id' => 'blogs_breadcrumbs_layout',
                        'type' => 'select',
                        'title' => esc_html__('Breadcrumbs Layout', 'univero'),
                        'options' => array(
                            'layout1' => esc_html__('Layout 1', 'univero'),
                            'layout2' => esc_html__('Layout 2', 'univero'),
                        ),
                        'default' => 'layout1'
                    ),
                    array (
                        'title' => esc_html__('Breadcrumbs Background Color', 'univero'),
                        'subtitle' => '<em>'.esc_html__('The breadcrumbs background color of the site.', 'univero').'</em>',
                        'id' => 'blogs_breadcrumb_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),
                    array(
                        'id' => 'blogs_breadcrumb_image',
                        'type' => 'media',
                        'title' => esc_html__('Breadcrumbs Background', 'univero'),
                        'subtitle' => esc_html__('Upload a .jpg or .png image that will be your breadcrumbs.', 'univero'),
                    ),
                    array (
                        'id' => 'blogs_general_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('General Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'blog_display_mode',
                        'type' => 'select',
                        'title' => esc_html__('Display Mode', 'univero'),
                        'options' => array(
                            'standard' => esc_html__('Standard', 'univero'),
                            'grid' => esc_html__('Grid Layout', 'univero'),
                            'list' => esc_html__('List Layout', 'univero'),
                        ),
                        'default' => 'grid'
                    ),
                    array(
                        'id' => 'blog_columns',
                        'type' => 'select',
                        'title' => esc_html__('Blog Columns', 'univero'),
                        'options' => $columns,
                        'default' => 4
                    ),
                    array(
                        'id' => 'blog_item_thumbsize',
                        'type' => 'text',
                        'title' => esc_html__('Thumbnail Size', 'univero'),
                        'desc' => esc_html__('Enter thumbnail size. Example: thumbnail, medium, large, full or other sizes defined by current theme.', 'univero'),
                    ),
                    array (
                        'id' => 'blogs_sidebar_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Sidebar Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'blog_archive_layout',
                        'type' => 'image_select',
                        'compiler' => true,
                        'title' => esc_html__('Sidebar position', 'univero'),
                        'subtitle' => esc_html__('Select the variation you want to apply on your store.', 'univero'),
                        'options' => array(
                            'main' => array(
                                'title' => esc_html__('Main Only', 'univero'),
                                'alt' => esc_html__('Main Only', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen1.png'
                            ),
                            'left-main' => array(
                                'title' => esc_html__('Left - Main Sidebar', 'univero'),
                                'alt' => esc_html__('Left - Main Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen2.png'
                            ),
                            'main-right' => array(
                                'title' => esc_html__('Main - Right Sidebar', 'univero'),
                                'alt' => esc_html__('Main - Right Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen3.png'
                            ),
                            'left-main-right' => array(
                                'title' => esc_html__('Left - Main - Right Sidebar', 'univero'),
                                'alt' => esc_html__('Left - Main - Right Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen4.png'
                            ),
                        ),
                        'default' => 'left-main'
                    ),
                    array(
                        'id' => 'blog_archive_fullwidth',
                        'type' => 'switch',
                        'title' => esc_html__('Is Full Width?', 'univero'),
                        'default' => false
                    ),
                    array(
                        'id' => 'blog_archive_left_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Archive Left Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for left sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    array(
                        'id' => 'blog_archive_right_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Archive Right Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for right sidebar.', 'univero'),
                        'options' => $sidebars
                        
                    ),
                    

                )
            );
            // Single Blogs settings
            $this->sections[] = array(
                'subsection' => true,
                'title' => esc_html__('Blog', 'univero'),
                'fields' => array(
                    array (
                        'id' => 'blog_breadcrumbs_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Breadcrumbs Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'show_blog_breadcrumbs',
                        'type' => 'switch',
                        'title' => esc_html__('Breadcrumbs', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'blog_breadcrumb_title',
                        'type' => 'text',
                        'title' => esc_html__('Breadcrumbs Title', 'univero'),
                        'default' => 'Blog'
                    ),
                    array(
                        'id' => 'blog_breadcrumbs_layout',
                        'type' => 'select',
                        'title' => esc_html__('Breadcrumbs Layout', 'univero'),
                        'options' => array(
                            'layout1' => esc_html__('Layout 1', 'univero'),
                            'layout2' => esc_html__('Layout 2', 'univero'),
                        ),
                        'default' => 'layout1'
                    ),
                    array (
                        'title' => esc_html__('Breadcrumbs Background Color', 'univero'),
                        'subtitle' => '<em>'.esc_html__('The breadcrumbs background color of the site.', 'univero').'</em>',
                        'id' => 'blog_breadcrumb_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),
                    array(
                        'id' => 'blog_breadcrumb_image',
                        'type' => 'media',
                        'title' => esc_html__('Breadcrumbs Background', 'univero'),
                        'subtitle' => esc_html__('Upload a .jpg or .png image that will be your breadcrumbs.', 'univero'),
                    ),
                    array (
                        'id' => 'blog_general_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('General Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'show_blog_social_share',
                        'type' => 'switch',
                        'title' => esc_html__('Show Social Share', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'show_blog_releated',
                        'type' => 'switch',
                        'title' => esc_html__('Show Releated Posts', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'number_blog_releated',
                        'type' => 'text',
                        'title' => esc_html__('Number of related posts to show', 'univero'),
                        'required' => array('show_blog_releated', '=', '1'),
                        'default' => 4,
                        'min' => '1',
                        'step' => '1',
                        'max' => '20',
                        'type' => 'slider'
                    ),
                    array(
                        'id' => 'releated_blog_columns',
                        'type' => 'select',
                        'title' => esc_html__('Releated Blogs Columns', 'univero'),
                        'required' => array('show_blog_releated', '=', '1'),
                        'options' => $columns,
                        'default' => 4
                    ),
                    array (
                        'id' => 'blog_sidebar_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Sidebar Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'blog_single_layout',
                        'type' => 'image_select',
                        'compiler' => true,
                        'title' => esc_html__('Sidebar position', 'univero'),
                        'subtitle' => esc_html__('Select the variation you want to apply on your store.', 'univero'),
                        'options' => array(
                            'main' => array(
                                'title' => esc_html__('Main Only', 'univero'),
                                'alt' => esc_html__('Main Only', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen1.png'
                            ),
                            'left-main' => array(
                                'title' => esc_html__('Left - Main Sidebar', 'univero'),
                                'alt' => esc_html__('Left - Main Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen2.png'
                            ),
                            'main-right' => array(
                                'title' => esc_html__('Main - Right Sidebar', 'univero'),
                                'alt' => esc_html__('Main - Right Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen3.png'
                            ),
                            'left-main-right' => array(
                                'title' => esc_html__('Left - Main - Right Sidebar', 'univero'),
                                'alt' => esc_html__('Left - Main - Right Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen4.png'
                            ),
                        ),
                        'default' => 'left-main'
                    ),
                    array(
                        'id' => 'blog_single_fullwidth',
                        'type' => 'switch',
                        'title' => esc_html__('Is Full Width?', 'univero'),
                        'default' => false
                    ),
                    array(
                        'id' => 'blog_single_left_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Single Blog Left Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for left sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    array(
                        'id' => 'blog_single_right_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Single Blog Right Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for right sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    

                )
            );
            // Woocommerce
            $this->sections[] = array(
                'icon' => 'el el-shopping-cart',
                'title' => esc_html__('Woocommerce', 'univero'),
                'fields' => array(
                )
            );
            // Archive settings
            $this->sections[] = array(
                'subsection' => true,
                'title' => esc_html__('Product Archives', 'univero'),
                'fields' => array(
                    array (
                        'id' => 'products_breadcrumbs_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Breadcrumbs Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'show_products_breadcrumbs',
                        'type' => 'switch',
                        'title' => esc_html__('Breadcrumbs', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'products_breadcrumb_title',
                        'type' => 'text',
                        'title' => esc_html__('Breadcrumbs Title', 'univero'),
                        'default' => 'Shop'
                    ),
                    array(
                        'id' => 'products_breadcrumbs_layout',
                        'type' => 'select',
                        'title' => esc_html__('Breadcrumbs Layout', 'univero'),
                        'options' => array(
                            'layout1' => esc_html__('Layout 1', 'univero'),
                            'layout2' => esc_html__('Layout 2', 'univero'),
                        ),
                        'default' => 'layout1'
                    ),
                    array (
                        'title' => esc_html__('Breadcrumbs Background Color', 'univero'),
                        'subtitle' => '<em>'.esc_html__('The breadcrumbs background color of the site.', 'univero').'</em>',
                        'id' => 'products_breadcrumb_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),
                    array(
                        'id' => 'products_breadcrumb_image',
                        'type' => 'media',
                        'title' => esc_html__('Breadcrumbs Background', 'univero'),
                        'subtitle' => esc_html__('Upload a .jpg or .png image that will be your breadcrumbs.', 'univero'),
                    ),
                    array (
                        'id' => 'products_general_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('General Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'number_products_per_page',
                        'type' => 'text',
                        'title' => esc_html__('Number of Products Per Page', 'univero'),
                        'default' => 12,
                        'min' => '1',
                        'step' => '1',
                        'max' => '100',
                        'type' => 'slider'
                    ),
                    array(
                        'id' => 'product_columns',
                        'type' => 'select',
                        'title' => esc_html__('Product Columns', 'univero'),
                        'options' => $columns,
                        'default' => 4
                    ),
                    array(
                        'id' => 'show_quickview',
                        'type' => 'switch',
                        'title' => esc_html__('Show Quick View', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'show_swap_image',
                        'type' => 'switch',
                        'title' => esc_html__('Show Second Image (Hover)', 'univero'),
                        'default' => 1
                    ),
                    array (
                        'id' => 'products_sidebar_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Sidebar Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'product_archive_layout',
                        'type' => 'image_select',
                        'compiler' => true,
                        'title' => esc_html__('Sidebar position', 'univero'),
                        'subtitle' => esc_html__('Select the layout you want to apply on your archive product page.', 'univero'),
                        'options' => array(
                            'main' => array(
                                'title' => esc_html__('Main Content', 'univero'),
                                'alt' => esc_html__('Main Content', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen1.png'
                            ),
                            'left-main' => array(
                                'title' => esc_html__('Left Sidebar - Main Content', 'univero'),
                                'alt' => esc_html__('Left Sidebar - Main Content', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen2.png'
                            ),
                            'main-right' => array(
                                'title' => esc_html__('Main Content - Right Sidebar', 'univero'),
                                'alt' => esc_html__('Main Content - Right Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen3.png'
                            ),
                            'left-main-right' => array(
                                'title' => esc_html__('Left Sidebar - Main Content - Right Sidebar', 'univero'),
                                'alt' => esc_html__('Left Sidebar - Main Content - Right Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen4.png'
                            ),
                        ),
                        'default' => 'left-main'
                    ),
                    array(
                        'id' => 'product_archive_fullwidth',
                        'type' => 'switch',
                        'title' => esc_html__('Is Full Width?', 'univero'),
                        'default' => false
                    ),
                    array(
                        'id' => 'product_archive_left_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Archive Left Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for left sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    array(
                        'id' => 'product_archive_right_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Archive Right Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for right sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    
                )
            );
            // Product Page
            $this->sections[] = array(
                'subsection' => true,
                'title' => esc_html__('Single Product', 'univero'),
                'fields' => array(
                    array (
                        'id' => 'product_breadcrumbs_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Breadcrumbs Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'show_product_breadcrumbs',
                        'type' => 'switch',
                        'title' => esc_html__('Breadcrumbs', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'product_breadcrumb_title',
                        'type' => 'text',
                        'title' => esc_html__('Breadcrumbs Title', 'univero'),
                        'default' => 'Shop'
                    ),
                    array(
                        'id' => 'product_breadcrumbs_layout',
                        'type' => 'select',
                        'title' => esc_html__('Breadcrumbs Layout', 'univero'),
                        'options' => array(
                            'layout1' => esc_html__('Layout 1', 'univero'),
                            'layout2' => esc_html__('Layout 2', 'univero'),
                        ),
                        'default' => 'layout1'
                    ),
                    array (
                        'title' => esc_html__('Breadcrumbs Background Color', 'univero'),
                        'subtitle' => '<em>'.esc_html__('The breadcrumbs background color of the site.', 'univero').'</em>',
                        'id' => 'product_breadcrumb_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),
                    array(
                        'id' => 'product_breadcrumb_image',
                        'type' => 'media',
                        'title' => esc_html__('Breadcrumbs Background', 'univero'),
                        'subtitle' => esc_html__('Upload a .jpg or .png image that will be your breadcrumbs.', 'univero'),
                    ),
                    array (
                        'id' => 'product_general_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('General Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'show_product_social_share',
                        'type' => 'switch',
                        'title' => esc_html__('Show Social Share', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'show_product_review_tab',
                        'type' => 'switch',
                        'title' => esc_html__('Show Product Review Tab', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'show_product_releated',
                        'type' => 'switch',
                        'title' => esc_html__('Show Products Releated', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'show_product_upsells',
                        'type' => 'switch',
                        'title' => esc_html__('Show Products upsells', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'number_product_releated',
                        'title' => esc_html__('Number of related/upsells products to show', 'univero'),
                        'default' => 4,
                        'min' => '1',
                        'step' => '1',
                        'max' => '20',
                        'type' => 'slider'
                    ),
                    array(
                        'id' => 'releated_product_columns',
                        'type' => 'select',
                        'title' => esc_html__('Releated Products Columns', 'univero'),
                        'options' => $columns,
                        'default' => 4
                    ),
                    array (
                        'id' => 'product_sidebar_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('General Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'product_single_layout',
                        'type' => 'image_select',
                        'compiler' => true,
                        'title' => esc_html__('Sidebar position', 'univero'),
                        'subtitle' => esc_html__('Select the layout you want to apply on your Single Product Page.', 'univero'),
                        'options' => array(
                            'main' => array(
                                'title' => esc_html__('Main Only', 'univero'),
                                'alt' => esc_html__('Main Only', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen1.png'
                            ),
                            'left-main' => array(
                                'title' => esc_html__('Left - Main Sidebar', 'univero'),
                                'alt' => esc_html__('Left - Main Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen2.png'
                            ),
                            'main-right' => array(
                                'title' => esc_html__('Main - Right Sidebar', 'univero'),
                                'alt' => esc_html__('Main - Right Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen3.png'
                            ),
                            'left-main-right' => array(
                                'title' => esc_html__('Left - Main - Right Sidebar', 'univero'),
                                'alt' => esc_html__('Left - Main - Right Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen4.png'
                            ),
                        ),
                        'default' => 'left-main'
                    ),
                    array(
                        'id' => 'product_single_fullwidth',
                        'type' => 'switch',
                        'title' => esc_html__('Is Full Width?', 'univero'),
                        'default' => false
                    ),
                    array(
                        'id' => 'product_single_left_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Single Product Left Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for left sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    array(
                        'id' => 'product_single_right_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Single Product Right Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for right sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    
                )
            );
            
            // Course
            $this->sections[] = array(
                'icon' => 'el el-pencil',
                'title' => esc_html__('Course', 'univero'),
                'fields' => array(
                    
                )
            );
            // Course Archive settings
            $this->sections[] = array(
                'subsection' => true,
                'title' => esc_html__('Course Archives', 'univero'),
                'fields' => array(
                    array (
                        'id' => 'courses_breadcrumbs_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Breadcrumbs Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'show_courses_breadcrumbs',
                        'type' => 'switch',
                        'title' => esc_html__('Breadcrumbs', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'courses_breadcrumb_title',
                        'type' => 'text',
                        'title' => esc_html__('Breadcrumbs Title', 'univero'),
                        'default' => 'Courses'
                    ),
                    array(
                        'id' => 'courses_breadcrumbs_layout',
                        'type' => 'select',
                        'title' => esc_html__('Breadcrumbs Layout', 'univero'),
                        'options' => array(
                            'layout1' => esc_html__('Layout 1', 'univero'),
                            'layout2' => esc_html__('Layout 2', 'univero'),
                        ),
                        'default' => 'layout1'
                    ),
                    array (
                        'title' => esc_html__('Breadcrumbs Background Color', 'univero'),
                        'subtitle' => '<em>'.esc_html__('The breadcrumbs background color of the site.', 'univero').'</em>',
                        'id' => 'courses_breadcrumb_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),
                    array(
                        'id' => 'courses_breadcrumb_image',
                        'type' => 'media',
                        'title' => esc_html__('Breadcrumbs Background', 'univero'),
                        'subtitle' => esc_html__('Upload a .jpg or .png image that will be your breadcrumbs.', 'univero'),
                    ),
                    array (
                        'id' => 'courses_general_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('General Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'course_archive_display_mode',
                        'type' => 'select',
                        'title' => esc_html__('Display Mode', 'univero'),
                        'options' => array(
                            'grid' => esc_html__('Grid', 'univero'),
                            'list' => esc_html__('List', 'univero'),
                        ),
                        'default' => 'grid'
                    ),
                    array(
                        'id' => 'number_courses_per_page',
                        'type' => 'text',
                        'title' => esc_html__('Number of Courses Per Page', 'univero'),
                        'default' => 12,
                        'min' => '1',
                        'step' => '1',
                        'max' => '100',
                        'type' => 'slider'
                    ),
                    array(
                        'id' => 'course_columns',
                        'type' => 'select',
                        'title' => esc_html__('Course Columns', 'univero'),
                        'options' => $columns,
                        'default' => 4
                    ),
                    array(
                        'id' => 'show_course_categories_top',
                        'type' => 'switch',
                        'title' => esc_html__('Show Top Course Categories', 'univero'),
                        'default' => true
                    ),
                    array (
                        'id' => 'courses_sidebar_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('General Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'course_archive_layout',
                        'type' => 'image_select',
                        'compiler' => true,
                        'title' => esc_html__('Sidebar position', 'univero'),
                        'subtitle' => esc_html__('Select the layout you want to apply on your archive course page.', 'univero'),
                        'options' => array(
                            'main' => array(
                                'title' => esc_html__('Main Content', 'univero'),
                                'alt' => esc_html__('Main Content', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen1.png'
                            ),
                            'left-main' => array(
                                'title' => esc_html__('Left Sidebar - Main Content', 'univero'),
                                'alt' => esc_html__('Left Sidebar - Main Content', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen2.png'
                            ),
                            'main-right' => array(
                                'title' => esc_html__('Main Content - Right Sidebar', 'univero'),
                                'alt' => esc_html__('Main Content - Right Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen3.png'
                            ),
                            'left-main-right' => array(
                                'title' => esc_html__('Left Sidebar - Main Content - Right Sidebar', 'univero'),
                                'alt' => esc_html__('Left Sidebar - Main Content - Right Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen4.png'
                            ),
                        ),
                        'default' => 'left-main'
                    ),
                    array(
                        'id' => 'course_archive_fullwidth',
                        'type' => 'switch',
                        'title' => esc_html__('Is Full Width?', 'univero'),
                        'default' => false
                    ),
                    array(
                        'id' => 'course_archive_left_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Archive Left Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for left sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    array(
                        'id' => 'course_archive_right_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Archive Right Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for right sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    
                )
            );
            // Course Page
            $this->sections[] = array(
                'subsection' => true,
                'title' => esc_html__('Single Course', 'univero'),
                'fields' => array(
                    array (
                        'id' => 'course_breadcrumbs_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Breadcrumbs Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'show_course_breadcrumbs',
                        'type' => 'switch',
                        'title' => esc_html__('Breadcrumbs', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'course_breadcrumb_title',
                        'type' => 'text',
                        'title' => esc_html__('Breadcrumbs Title', 'univero'),
                        'default' => 'Courses'
                    ),
                    array(
                        'id' => 'course_breadcrumbs_layout',
                        'type' => 'select',
                        'title' => esc_html__('Breadcrumbs Layout', 'univero'),
                        'options' => array(
                            'layout1' => esc_html__('Layout 1', 'univero'),
                            'layout2' => esc_html__('Layout 2', 'univero'),
                        ),
                        'default' => 'layout1'
                    ),
                    array (
                        'title' => esc_html__('Breadcrumbs Background Color', 'univero'),
                        'subtitle' => '<em>'.esc_html__('The breadcrumbs background color of the site.', 'univero').'</em>',
                        'id' => 'course_breadcrumb_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),
                    array(
                        'id' => 'course_breadcrumb_image',
                        'type' => 'media',
                        'title' => esc_html__('Breadcrumbs Background', 'univero'),
                        'subtitle' => esc_html__('Upload a .jpg or .png image that will be your breadcrumbs.', 'univero'),
                    ),
                    array (
                        'id' => 'course_general_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('General Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'show_course_social_share',
                        'type' => 'switch',
                        'title' => esc_html__('Show Social Share', 'univero'),
                        'default' => 1
                    ),
                    array (
                        'id' => 'course_sidebar_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('General Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'course_single_layout',
                        'type' => 'image_select',
                        'compiler' => true,
                        'title' => esc_html__('Sidebar position', 'univero'),
                        'subtitle' => esc_html__('Select the layout you want to apply on your Single Course Page.', 'univero'),
                        'options' => array(
                            'main' => array(
                                'title' => esc_html__('Main Only', 'univero'),
                                'alt' => esc_html__('Main Only', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen1.png'
                            ),
                            'left-main' => array(
                                'title' => esc_html__('Left - Main Sidebar', 'univero'),
                                'alt' => esc_html__('Left - Main Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen2.png'
                            ),
                            'main-right' => array(
                                'title' => esc_html__('Main - Right Sidebar', 'univero'),
                                'alt' => esc_html__('Main - Right Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen3.png'
                            ),
                            'left-main-right' => array(
                                'title' => esc_html__('Left - Main - Right Sidebar', 'univero'),
                                'alt' => esc_html__('Left - Main - Right Sidebar', 'univero'),
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen4.png'
                            ),
                        ),
                        'default' => 'left-main'
                    ),
                    array(
                        'id' => 'course_single_fullwidth',
                        'type' => 'switch',
                        'title' => esc_html__('Is Full Width?', 'univero'),
                        'default' => false
                    ),
                    array(
                        'id' => 'course_single_left_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Single Course Left Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for left sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    array(
                        'id' => 'course_single_right_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Single Course Right Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for right sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    
                )
            );
            
            // Event
            $this->sections[] = array(
                'icon' => 'el el-pencil',
                'title' => esc_html__('Event', 'univero'),
                'fields' => array(
                )
            );
            // Archive Event Settings
            $this->sections[] = array(
                'subsection' => true,
                'title' => esc_html__('Event Archives', 'univero'),
                'fields' => array(
                    array (
                        'id' => 'events_breadcrumbs_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Breadcrumbs Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'show_events_breadcrumbs',
                        'type' => 'switch',
                        'title' => esc_html__('Breadcrumbs', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'events_breadcrumb_title',
                        'type' => 'text',
                        'title' => esc_html__('Breadcrumbs Title', 'univero'),
                        'default' => 'Events'
                    ),
                    array(
                        'id' => 'events_breadcrumbs_layout',
                        'type' => 'select',
                        'title' => esc_html__('Breadcrumbs Layout', 'univero'),
                        'options' => array(
                            'layout1' => esc_html__('Layout 1', 'univero'),
                            'layout2' => esc_html__('Layout 2', 'univero'),
                        ),
                        'default' => 'layout1'
                    ),
                    array (
                        'title' => esc_html__('Breadcrumbs Background Color', 'univero'),
                        'subtitle' => '<em>'.esc_html__('The breadcrumbs background color of the site.', 'univero').'</em>',
                        'id' => 'events_breadcrumb_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),
                    array(
                        'id' => 'events_breadcrumb_image',
                        'type' => 'media',
                        'title' => esc_html__('Breadcrumbs Background', 'univero'),
                        'subtitle' => esc_html__('Upload a .jpg or .png image that will be your breadcrumbs.', 'univero'),
                    ),
                    array (
                        'id' => 'events_general_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('General Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'show_top_event_categories',
                        'type' => 'switch',
                        'title' => esc_html__('Show Top Event Categories', 'univero'),
                        'default' => true
                    ),
                    array(
                        'id' => 'event_archive_display_mode',
                        'type' => 'select',
                        'title' => esc_html__('Display Mode', 'univero'),
                        'options' => array(
                            'grid' => esc_html__('Grid', 'univero'),
                            'list' => esc_html__('List', 'univero'),
                        ),
                        'default' => 'grid'
                    ),
                    array(
                        'id' => 'event_archive_columns',
                        'type' => 'select',
                        'title' => esc_html__('Events Columns', 'univero'),
                        'options' => $columns,
                        'default' => 3,
                        'required' => array('event_archive_display_mode', 'equals', 'grid'),
                    ),
                    array (
                        'id' => 'events_sidebar_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Sidebar Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'event_archive_layout',
                        'type' => 'image_select',
                        'compiler' => true,
                        'title' => esc_html__('Archive Event Layout', 'univero'),
                        'subtitle' => esc_html__('Select the layout you want to apply on your archive event page.', 'univero'),
                        'options' => array(
                            'main' => array(
                                'title' => 'Main Content',
                                'alt' => 'Main Content',
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen1.png'
                            ),
                            'left-main' => array(
                                'title' => 'Left Sidebar - Main Content',
                                'alt' => 'Left Sidebar - Main Content',
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen2.png'
                            ),
                            'main-right' => array(
                                'title' => 'Main Content - Right Sidebar',
                                'alt' => 'Main Content - Right Sidebar',
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen3.png'
                            ),
                            'left-main-right' => array(
                                'title' => 'Left Sidebar - Main Content - Right Sidebar',
                                'alt' => 'Left Sidebar - Main Content - Right Sidebar',
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen4.png'
                            ),
                        ),
                        'default' => 'left-main'
                    ),
                    array(
                        'id' => 'event_archive_fullwidth',
                        'type' => 'switch',
                        'title' => esc_html__('Is Full Width?', 'univero'),
                        'default' => false
                    ),
                    array(
                        'id' => 'event_archive_left_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Archive Left Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for left sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    array(
                        'id' => 'event_archive_right_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Archive Right Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for right sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    
                )
            );
            // Event Detail Page
            $this->sections[] = array(
                'subsection' => true,
                'title' => esc_html__('Single Event', 'univero'),
                'fields' => array(
                    array (
                        'id' => 'event_breadcrumbs_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Breadcrumbs Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'show_event_breadcrumbs',
                        'type' => 'switch',
                        'title' => esc_html__('Breadcrumbs', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'event_breadcrumb_title',
                        'type' => 'text',
                        'title' => esc_html__('Breadcrumbs Title', 'univero'),
                        'default' => 'Events'
                    ),
                    array(
                        'id' => 'event_breadcrumbs_layout',
                        'type' => 'select',
                        'title' => esc_html__('Breadcrumbs Layout', 'univero'),
                        'options' => array(
                            'layout1' => esc_html__('Layout 1', 'univero'),
                            'layout2' => esc_html__('Layout 2', 'univero'),
                        ),
                        'default' => 'layout1'
                    ),
                    array (
                        'title' => esc_html__('Breadcrumbs Background Color', 'univero'),
                        'subtitle' => '<em>'.esc_html__('The breadcrumbs background color of the site.', 'univero').'</em>',
                        'id' => 'event_breadcrumb_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),
                    array(
                        'id' => 'event_breadcrumb_image',
                        'type' => 'media',
                        'title' => esc_html__('Breadcrumbs Background', 'univero'),
                        'subtitle' => esc_html__('Upload a .jpg or .png image that will be your breadcrumbs.', 'univero'),
                    ),
                    array (
                        'id' => 'event_general_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('General Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'show_event_social_share',
                        'type' => 'switch',
                        'title' => esc_html__('Show Social Share', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'show_event_social_share',
                        'type' => 'switch',
                        'title' => esc_html__('Show Social Share', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'show_event_releated',
                        'type' => 'switch',
                        'title' => esc_html__('Show Releated Events', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'number_event_releated',
                        'type' => 'text',
                        'title' => esc_html__('Number of related events to show', 'univero'),
                        'required' => array('show_event_releated', '=', '1'),
                        'default' => 4,
                        'min' => '1',
                        'step' => '1',
                        'max' => '20',
                        'type' => 'slider'
                    ),
                    array(
                        'id' => 'releated_event_columns',
                        'type' => 'select',
                        'title' => esc_html__('Releated Events Columns', 'univero'),
                        'required' => array('show_event_releated', '=', '1'),
                        'options' => $columns,
                        'default' => 4
                    ),
                    array (
                        'id' => 'event_sidebar_settings',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Sidebar Settings', 'univero').'</h3>',
                    ),
                    array(
                        'id' => 'event_single_layout',
                        'type' => 'image_select',
                        'compiler' => true,
                        'title' => esc_html__('Single Event Layout', 'univero'),
                        'subtitle' => esc_html__('Select the layout you want to apply on your Single Event Page.', 'univero'),
                        'options' => array(
                            'main' => array(
                                'title' => 'Main Only',
                                'alt' => 'Main Only',
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen1.png'
                            ),
                            'left-main' => array(
                                'title' => 'Left - Main Sidebar',
                                'alt' => 'Left - Main Sidebar',
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen2.png'
                            ),
                            'main-right' => array(
                                'title' => 'Main - Right Sidebar',
                                'alt' => 'Main - Right Sidebar',
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen3.png'
                            ),
                            'left-main-right' => array(
                                'title' => 'Left - Main - Right Sidebar',
                                'alt' => 'Left - Main - Right Sidebar',
                                'img' => get_template_directory_uri() . '/inc/assets/images/screen4.png'
                            ),
                        ),
                        'default' => 'left-main'
                    ),
                    array(
                        'id' => 'event_single_fullwidth',
                        'type' => 'switch',
                        'title' => esc_html__('Is Full Width?', 'univero'),
                        'default' => false
                    ),
                    array(
                        'id' => 'event_single_left_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Single Event Left Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for left sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    array(
                        'id' => 'event_single_right_sidebar',
                        'type' => 'select',
                        'title' => esc_html__('Single Event Right Sidebar', 'univero'),
                        'subtitle' => esc_html__('Choose a sidebar for right sidebar.', 'univero'),
                        'options' => $sidebars
                    ),
                    
                )
            );
            // 404 Page
            $this->sections[] = array(
                'icon' => 'el el-pencil',
                'title' => esc_html__('404 Page', 'univero'),
                'fields' => array(
                    array(
                        'id' => '404_title',
                        'type' => 'text',
                        'title' => esc_html__('Title', 'univero'),
                        'default' => 'Page not found'
                    ),
                    array(
                        'id' => '404_description',
                        'type' => 'textarea',
                        'title' => esc_html__('Desciption', 'univero'),
                        'default' => 'We are sorry, but we can not find the page you were looking for'
                    ),
                )
            );
            // Style
            $this->sections[] = array(
                'icon' => 'el el-icon-css',
                'title' => esc_html__('Style', 'univero'),
                'fields' => array(
                    array (
                        'id' => 'main_font_info',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Content', 'univero').'</h3>',
                    ),
                    array (
                        'title' => esc_html__('Main Theme Color', 'univero'),
                        'subtitle' => esc_html__('The main color of the site.', 'univero'),
                        'id' => 'main_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),
                    array (
                        'title' => esc_html__('Second Theme Color', 'univero'),
                        'subtitle' => esc_html__('The Second color of the site.', 'univero'),
                        'id' => 'second_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),
                    array (
                        'title' => esc_html__('Accent Theme Color', 'univero'),
                        'subtitle' => esc_html__('The Accent color of the site.', 'univero'),
                        'id' => 'accent_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),                    

                    array (
                        'title' => esc_html__('Button Theme Background Color', 'univero'),
                        'subtitle' => esc_html__('The Button color of the site.', 'univero'),
                        'id' => 'button_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),
                    array (
                        'title' => esc_html__('Button Theme Color', 'univero'),
                        'subtitle' => esc_html__('The Button color of the site.', 'univero'),
                        'id' => 'button_text_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),
                    array (
                        'title' => esc_html__('Button Theme Background Hover Color', 'univero'),
                        'subtitle' => esc_html__('The Button color of the site.', 'univero'),
                        'id' => 'button_hover_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),
                    array (
                        'title' => esc_html__('Button Theme Hover Color', 'univero'),
                        'subtitle' => esc_html__('The Button color of the site.', 'univero'),
                        'id' => 'button_hover_text_color',
                        'type' => 'color',
                        'transparent' => false,
                    ),
                    array (
                        'id' => 'site_background',
                        'type' => 'background',
                        'title' => esc_html__('Site Background', 'univero'),
                        'output' => 'body'
                    ),
                    array (
                        'id' => 'container_bg',
                        'type' => 'color_rgba',
                        'title' => esc_html__('Container Background Color', 'univero'),
                        'output' => array(
                            'background-color' =>'#ninzio-main-content,.wrapper-shop,.single-product .wrapper-shop, .detail-post #comments::before,.detail-post #comments::after,.detail-post #comments
                            .widget.upsells::before, .widget.upsells::after, .widget.related::before, .widget.related::after,.widget.related
                            '
                        )
                    ),
                    array (
                        'id' => 'forms_inputs_bg',
                        'type' => 'color_rgba',
                        'title' => esc_html__('Forms inputs Color', 'univero'),
                        'output' => array(
                            'background-color' =>'.form-control, select, input[type="text"], input[type="email"], input[type="password"], input[type="tel"], textarea, textarea.form-control'
                        )
                    ),
                )
            );
            $this->sections[] = array(
                'subsection' => true,
                'title' => esc_html__('Typography', 'univero'),
                'fields' => array(
                    
                    array (
                        'id' => 'main_font_info',
                        'icon' => true,
                        'type' => 'info',
                        'raw' => '<h3 style="margin: 0;"> '.esc_html__('Body Font', 'univero').'</h3>',
                    ),
                    // Standard + Google Webfonts
                    array (
                        'title' => esc_html__('Font Face', 'univero'),
                        'subtitle' => '<em>'.esc_html__('Pick the Main Font for your site.', 'univero').'</em>',
                        'id' => 'main_font',
                        'type' => 'typography',
                        'line-height' => false,
                        'text-align' => false,
                        'font-style' => false,
                        'font-weight' => false,
                        'all_styles'=> true,
                        'font-size' => false,
                        'color' => false
                    ),
                    array (
                        'title' => esc_html__('Font Face Second', 'univero'),
                        'subtitle' => '<em>'.esc_html__('Pick the Second Font for your site( Heading).', 'univero').'</em>',
                        'id' => 'second_font',
                        'type' => 'typography',
                        'line-height' => false,
                        'text-align' => false,
                        'font-style' => false,
                        'font-weight' => false,
                        'all_styles'=> true,
                        'font-size' => false,
                        'color' => false
                    ),
                    array (
                        'title' => esc_html__('Font Face Button', 'univero'),
                        'subtitle' => '<em>'.esc_html__('Pick the Second Font for Button.', 'univero').'</em>',
                        'id' => 'button_font',
                        'type' => 'typography',
                        'line-height' => false,
                        'text-align' => false,
                        'font-style' => false,
                        'font-weight' => false,
                        'all_styles'=> true,
                        'font-size' => false,
                        'color' => false
                    ),
                )
            );
            $this->sections[] = array(
                'subsection' => true,
                'title' => esc_html__('Top Bar', 'univero'),
                'fields' => array(
                    array(
                        'id'=>'topbar_bg',
                        'type' => 'background',
                        'title' => esc_html__('Background', 'univero'),
                        'output' => '#ninzio-topbar'
                    ),
                    array(
                        'title' => esc_html__('Text Color', 'univero'),
                        'id' => 'topbar_text_color',
                        'type' => 'color_rgba',
                        'output' => array(
                            'color' =>'#ninzio-topbar'
                        )
                    ),
                    array(
                        'title' => esc_html__('Link Color', 'univero'),
                        'id' => 'topbar_link_color',
                        'type' => 'color_rgba',
                        'output' => array(
                            'color' =>'#ninzio-header #ninzio-topbar a,#ninzio-header #ninzio-topbar .list-social > li > a'                            
                        )
                    ),
                    array(
                        'title' => esc_html__('Link Color When Hover', 'univero'),
                        'id' => 'topbar_link_color_hover',
                        'type' => 'color_rgba',
                        'output' => array(
                            'color' =>'#ninzio-header #ninzio-topbar a:hover,
#ninzio-header #ninzio-topbar a:active, 
#ninzio-header .ninzio-topbar a:focus,
#ninzio-header .ninzio-topbar a:hover,
#ninzio-header #ninzio-topbar .list-social > li > a:hover, 
#ninzio-header #ninzio-topbar .list-social > li > a:active,
.ninzio-header.header-v2 .ninzio-topbar a:hover,
.ninzio-header.header-v2 .ninzio-topbar a:focus,
.ninzio-header.header-v2 .ninzio-topbar a:active,
.ninzio-header.header-v4 .ninzio-topbar a:hover,
.ninzio-header.header-v4 .ninzio-topbar a:active,
.ninzio-header.header-v4 .ninzio-topbar a:focus'
                        )
                    ),
                )
            );
            $this->sections[] = array(
                'subsection' => true,
                'title' => esc_html__('Header', 'univero'),
                'fields' => array(
                    array(
                        'id'=>'header_bg',
                        'type' => 'background',
                        'title' => esc_html__('Background', 'univero'),
                        'output' => '#ninzio-header, #ninzio-header-mobile'
                    ),
                    array(
                        'title' => esc_html__('Text Color', 'univero'),
                        'id' => 'header_text_color',
                        'type' => 'color',
                        'output' => array(
                            'color' =>'#ninzio-header, #ninzio-header-mobile, #ninzio-header .header-meta .media-heading'
                        )
                    ),
                    array(
                        'title' => esc_html__('Link Color', 'univero'),
                        'id' => 'header_link_color',
                        'type' => 'color',
                        'output' => array(
                            'color' =>'#ninzio-header a, #ninzio-header-mobile a'
                        )
                    ),
                    array(
                        'title' => esc_html__('Link Color Active', 'univero'),
                        'id' => 'header_link_color_active',
                        'type' => 'color',
                        'output' => array(
                            'color' =>'#ninzio-header .active>a,
#ninzio-header a:active,
#ninzio-header a:hover,
#ninzio-header-mobile .active>a,
#ninzio-header-mobile a:active,
#ninzio-header-mobile a:hover,
.ninzio-header .header-meta a:hover,
.ninzio-header .header-meta a:active,
.ninzio-header .header-meta a:focus'                               
                        )
                    ),
                )
            );
            $this->sections[] = array(
                'subsection' => true,
                'title' => esc_html__('Main Menu', 'univero'),
                'fields' => array(
                     array(
                        'id'=>'main_menu_bg',
                        'type' => 'background',
                        'title' => esc_html__('Background', 'univero'),
                        'output' => '
                            .ninzio-megamenu:before,
                            .ninzio-megamenu:after,
                            .navbar-nav .dropdown-menu>li.open>a:before,
                            .navbar-nav .dropdown-menu>li.active>a:before,
                            .header-v1 .ninzio-megamenu,
                            .navbar-nav>li>ul>li>a:before',                                                        

                    ),
                    array(
                        'title' => esc_html__('Link Color', 'univero'),
                        'id' => 'main_menu_link_color',
                        'type' => 'color',
                        'output' => array(
                            'color' =>'#ninzio-header .navbar-nav.megamenu > li > a,
.header-v1 .navbar-nav > li > a
                            '
                        )
                    ),
                    array(
                        'title' => esc_html__('Link Color Active', 'univero'),
                        'id' => 'main_menu_link_color_active',
                        'type' => 'color',
                        'output' => array(
                            'color' =>'#ninzio-header .navbar-nav.megamenu > li.active > a,
#ninzio-header .navbar-nav.megamenu > li:hover > a,
#ninzio-header .navbar-nav.megamenu > li:active > a,
.header-v1 .navbar-nav > li > a:hover,
.header-v1 .navbar-nav > li > a:focus,
.header-v1 .navbar-nav > li > a:active,                            
#ninzio-header .navbar-nav > li.active > a,                            
#ninzio-header .navbar-nav .dropdown-menu > li:hover > a, 
#ninzio-header .navbar-nav .dropdown-menu > li:focus > a, 
#ninzio-header .navbar-nav .dropdown-menu > li:active > a,
#ninzio-header .navbar-nav .dropdown-menu > li.active > a,  
#ninzio-header .navbar-nav .dropdown-menu > li > a:hover, 
#ninzio-header .navbar-nav .dropdown-menu > li > a:focus, 
#ninzio-header .navbar-nav .dropdown-menu > li > a:active
#ninzio-header .navbar-nav .dropdown-menu ul > li:hover > a, 
#ninzio-header .navbar-nav .dropdown-menu ul > li:active > a, 
#ninzio-header .navbar-nav .dropdown-menu ul > li:focus > a, 
#ninzio-header .navbar-nav .dropdown-menu .menu-megamenu-container ul > li:hover > a, 
#ninzio-header .navbar-nav .dropdown-menu .menu-megamenu-container ul > li:active > a, 
#ninzio-header .navbar-nav .dropdown-menu .menu-megamenu-container ul > li:focus > a'                            
                        )
                    ),
                )
            );
            $this->sections[] = array(
                'subsection' => true,
                'title' => esc_html__('Footer', 'univero'),
                'fields' => array(
                    array(
                        'title' => esc_html__('Heading Color', 'univero'),
                        'id' => 'footer_heading_color',
                        'type' => 'color',
                        'output' => array(
                            'color' => '#ninzio-footer .widgettitle ,#ninzio-footer .widget-title,#ninzio-footer .title,
                            #ninzio-footer h1,#ninzio-footer h2,#ninzio-footer h3,#ninzio-footer h4,#ninzio-footer h5,#ninzio-footer h6'
                        )
                    ),
                    array(
                        'title' => esc_html__('Text Color', 'univero'),
                        'id' => 'footer_text_color',
                        'type' => 'color',
                        'output' => array(
                            'color' => '#ninzio-footer, .ninzio-footer .contact-info, .ninzio-copyright'
                        )
                    ),
                    array(
                        'title' => esc_html__('Link Color', 'univero'),
                        'id' => 'footer_link_color',
                        'type' => 'color',
                        'output' => array(
                            'color' => '#ninzio-footer a'
                        )
                    ),
                    array(
                        'title' => esc_html__('Link Color Hover', 'univero'),
                        'id' => 'footer_link_color_hover',
                        'type' => 'color',
                        'output' => array(
                            'color' => '#ninzio-footer a:hover,#ninzio-footer a:active'
                        )
                    ),
                )
            );
            
            $this->sections[] = array(
                'subsection' => true,
                'title' => esc_html__('Copyright', 'univero'),
                'fields' => array(
                    array(
                        'title' => esc_html__('Text Color', 'univero'),
                        'id' => 'copyright_text_color',
                        'type' => 'color',
                        'output' => array(
                            'color' => '.ninzio-copyright'
                        )
                    ),
                    array(
                        'title' => esc_html__('Link Color', 'univero'),
                        'id' => 'copyright_link_color',
                        'type' => 'color',
                        'output' => array(
                            'color' => '.ninzio-copyright a, .ninzio-copyright a i,
                            .ninzio-footer .ninzio_custom_menu .menu > li > a'
                        )
                    ),
                    array(
                        'title' => esc_html__('Link Color Hover', 'univero'),
                        'id' => 'copyright_link_color_hover',
                        'type' => 'color',
                        'output' => array(
                            'color' => '.ninzio-copyright a:hover .ninzio-copyright a i:hover
                            .ninzio-footer .ninzio_custom_menu .menu > li > a:hover'
                        )
                    ),
                )
            );

            // Social Media
            $this->sections[] = array(
                'icon' => 'el el-file',
                'title' => esc_html__('Social Media', 'univero'),
                'fields' => array(
                    array(
                        'id' => 'facebook_share',
                        'type' => 'switch',
                        'title' => esc_html__('Enable Facebook Share', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'twitter_share',
                        'type' => 'switch',
                        'title' => esc_html__('Enable twitter Share', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'linkedin_share',
                        'type' => 'switch',
                        'title' => esc_html__('Enable linkedin Share', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'google_share',
                        'type' => 'switch',
                        'title' => esc_html__('Enable google plus Share', 'univero'),
                        'default' => 1
                    ),
                    array(
                        'id' => 'pinterest_share',
                        'type' => 'switch',
                        'title' => esc_html__('Enable pinterest Share', 'univero'),
                        'default' => 1
                    )
                )
            );
           
            $this->sections[] = array(
                'title' => esc_html__('Import / Export', 'univero'),
                'desc' => esc_html__('Import and Export your Redux Framework settings from file, text or URL.', 'univero'),
                'icon' => 'el-icon-refresh',
                'fields' => array(
                    array(
                        'id' => 'opt-import-export',
                        'type' => 'import_export',
                        'title' => esc_html__('Import Export', 'univero'),
                        'subtitle' => esc_html__('Save and restore your Redux options', 'univero'),
                        'full_width' => false,
                    ),
                ),
            );

            $this->sections[] = array(
                'type' => 'divide',
            );
        }

        /**
         * All the possible arguments for Redux.
         * For full documentation on arguments, please refer to: https://github.com/ReduxFramework/ReduxFramework/wiki/Arguments
         * */
        public function setArguments()
        {
            $theme = wp_get_theme(); // For use with some settings. Not necessary.

            $preset = univero_get_demo_preset();
            $this->args = array(
                // TYPICAL -> Change these values as you need/desire
                'opt_name' => 'univero_theme_options'.$preset,
                // This is where your data is stored in the database and also becomes your global variable name.
                'display_name' => $theme->get('Name'),
                // Name that appears at the top of your panel
                'display_version' => $theme->get('Version'),
                // Version that appears at the top of your panel
                'menu_type' => 'menu',
                //Specify if the admin menu should appear or not. Options: menu or submenu (Under appearance only)
                'allow_sub_menu' => true,
                // Show the sections below the admin menu item or not
                'menu_title' => esc_html__('Theme Options', 'univero'),
                'page_title' => esc_html__('Theme Options', 'univero'),

                // You will need to generate a Google API key to use this feature.
                // Please visit: https://developers.google.com/fonts/docs/developer_api#Auth
                'google_api_key' => '',
                // Set it you want google fonts to update weekly. A google_api_key value is required.
                'google_update_weekly' => false,
                // Must be defined to add google fonts to the typography module
                'async_typography' => true,
                // Use a asynchronous font on the front end or font string
                //'disable_google_fonts_link' => true,                    // Disable this in case you want to create your own google fonts loader
                'admin_bar' => true,
                // Show the panel pages on the admin bar
                'admin_bar_icon' => 'dashicons-portfolio',
                // Choose an icon for the admin bar menu
                'admin_bar_priority' => 50,
                // Choose an priority for the admin bar menu
                'global_variable' => 'univero_options',
                // Set a different name for your global variable other than the opt_name
                'dev_mode' => false,
                // Show the time the page took to load, etc
                'update_notice' => true,
                // If dev_mode is enabled, will notify developer of updated versions available in the GitHub Repo
                'customizer' => true,
                // Enable basic customizer support
                //'open_expanded'     => true,                    // Allow you to start the panel in an expanded way initially.
                //'disable_save_warn' => true,                    // Disable the save warning when a user changes a field

                // OPTIONAL -> Give you extra features
                'page_priority' => null,
                // Order where the menu appears in the admin area. If there is any conflict, something will not show. Warning.
                'page_parent' => 'themes.php',
                // For a full list of options, visit: //codex.wordpress.org/Function_Reference/add_submenu_page#Parameters
                'page_permissions' => 'manage_options',
                // Permissions needed to access the options panel.
                'menu_icon' => '',
                // Specify a custom URL to an icon
                'last_tab' => '',
                // Force your panel to always open to a specific tab (by id)
                'page_icon' => 'icon-themes',
                // Icon displayed in the admin panel next to your menu_title
                'page_slug' => '_options',
                // Page slug used to denote the panel
                'save_defaults' => true,
                // On load save the defaults to DB before user clicks save or not
                'default_show' => false,
                // If true, shows the default value next to each field that is not the default value.
                'default_mark' => '',
                // What to print by the field's title if the value shown is default. Suggested: *
                'show_import_export' => true,
                // Shows the Import/Export panel when not used as a field.

                // CAREFUL -> These options are for advanced use only
                'transient_time' => 60 * MINUTE_IN_SECONDS,
                'output' => true,
                // Global shut-off for dynamic CSS output by the framework. Will also disable google fonts output
                'output_tag' => true,
                // Allows dynamic CSS to be generated for customizer and google fonts, but stops the dynamic CSS from going to the head
                // 'footer_credit'     => '',                   // Disable the footer credit of Redux. Please leave if you can help it.

                // FUTURE -> Not in use yet, but reserved or partially implemented. Use at your own risk.
                'database' => '',
                // possible: options, theme_mods, theme_mods_expanded, transient. Not fully functional, warning!
                'system_info' => false,
                // REMOVE
                'use_cdn' => true
            );

            return $this->args;
        }

    }

    global $reduxConfig;
    $reduxConfig = new Univero_Redux_Framework_Config();
}

if ( function_exists('ninzio_framework_redux_register_custom_extension_loader') ) {
    $preset = univero_get_demo_preset();
    $opt_name = 'univero_theme_options'.$preset;
    add_action("redux/extensions/{$opt_name}/before", 'ninzio_framework_redux_register_custom_extension_loader', 0);
}