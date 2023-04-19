<?php
/*---------------------------------------------------
/* THEME ADD ACTION
/*---------------------------------------------------*/
remove_action('g5plus_after_single_post_content','g5plus_share',15);

/*---------------------------------------------------
/* CUSTOM HEADER CSS
/*---------------------------------------------------*/
if (!function_exists('g5plus_custom_header_css')) {
	function g5plus_custom_header_css() {
		$page_id = '0';
		if (isset($_REQUEST['current_page_id'])) {
			$page_id = $_REQUEST['current_page_id'];
		}

		$css_variable = g5plus_custom_css_variable($page_id);

		if (!class_exists('Less_Parser')) {
			require_once G5PLUS_THEME_DIR . 'g5plus-framework/less/Less.php';
		}
		$parser = new Less_Parser(array( 'compress'=>true ));

		$parser->parse($css_variable, G5PLUS_THEME_URL);
		$parser->parseFile( G5PLUS_THEME_DIR . 'assets/css/less/variable.less', G5PLUS_THEME_URL );
		$parser->parseFile( G5PLUS_THEME_DIR . 'assets/css/less/header-customize.less', G5PLUS_THEME_URL );
		$parser->parseFile( G5PLUS_THEME_DIR . 'assets/css/less/footer-customize.less', G5PLUS_THEME_URL );

		$prefix = 'g5plus_';
		$enable_page_color = rwmb_meta($prefix . 'enable_page_color', array(), $page_id);
		if ($enable_page_color == '1') {
			$parser->parseFile( G5PLUS_THEME_DIR . 'assets/css/less/color.less' );
		}

		$css = $parser->getCss();
		header("Content-type: text/css; charset:" . get_bloginfo( 'charset', 'display' ));
		echo sprintf('%s', $css);
	}
	add_action('custom-page/header-custom-css', 'g5plus_custom_header_css');
}

if (!function_exists('g5plus_get_font_family')) {
    function g5plus_get_font_family($name) {
        if ((strpos($name, ',') === false) || (strpos($name, ' ') === false)) {
            return $name;
        }
        return "'{$name}'";
    }
}

if (!function_exists('g5plus_process_font')) {
    function g5plus_process_font($fonts) {
        if (isset($fonts['font-weight']) && (($fonts['font-weight'] === '') || ($fonts['font-weight'] === 'regular')) ) {
            $fonts['font-weight'] = '400';
        }

        if (isset($fonts['font-style']) && ($fonts['font-style'] === '') ) {
            $fonts['font-style'] = 'normal';
        }
        return $fonts;
    }
}

if (!function_exists('g5plus_custom_css_editor_callback')) {
    function g5plus_custom_css_editor_callback() {
        $custom_css = g5plus_custom_css_editor();

        /**
         * Make sure we set the correct MIME type
         */
        header( 'Content-Type: text/css' );
        /**
         * Render RTL CSS
         */
        echo sprintf('%s',$custom_css);
        die();
    }
    add_action( 'wp_ajax_gsf_custom_css_editor', 'g5plus_custom_css_editor_callback');
    add_action( 'wp_ajax_nopriv_gsf_custom_css_editor', 'g5plus_custom_css_editor_callback');
}

if (!function_exists('g5plus_custom_css_editor')) {
    function g5plus_custom_css_editor() {
        $custom_css =<<<CSS
        body {
              margin: 9px 10px;
            }
CSS;
        $post_id = isset($_GET['post_id']) ? $_GET['post_id'] : '';

        if (!empty($post_id)) {


            $g5plus_options = &G5Plus_Global::get_options();
            $prefix = 'g5plus_';
            $sidebar_layout = rwmb_meta($prefix.'page_sidebar', array(), $post_id);
            if (($sidebar_layout === '') || ($sidebar_layout == '-1')) {
                $sidebar_layout = $g5plus_options['single_blog_sidebar'];
            }

            $sidebar_width = rwmb_meta($prefix.'sidebar_width', array(), $post_id);
            if (($sidebar_width === '') || ($sidebar_width == '-1')) {
                $sidebar_width = $g5plus_options['single_blog_sidebar_width'];
            }

            $left_sidebar = rwmb_meta($prefix.'page_left_sidebar', array(), $post_id);
            if (($left_sidebar === '') || ($left_sidebar == '-1')) {
                $left_sidebar = $g5plus_options['single_blog_left_sidebar'];
            }

            $right_sidebar = rwmb_meta($prefix.'page_right_sidebar', array(), $post_id);
            if (($right_sidebar === '') || ($right_sidebar == '-1')) {
                $right_sidebar = $g5plus_options['single_blog_right_sidebar'];
            }




            $content_width = 1170;
            $sidebar_text = esc_html__('Sidebar', 'g5plus-academia');
            if ($sidebar_width === 'large') {
                $sidebar_width = 770;
            } else {
                $sidebar_width = 870;
            }

            $custom_css .= <<<CSS
            
            .mceContentBody::after {
              display: block;
              position: absolute;
              top: 0;
              left: 102%;
              width: 10px;
              -ms-word-break: break-all;
              word-break: break-all;
              font-size: 14px;
              color: #d8d8d8;
              text-align: center;
              height: 100%;
              max-width: 330px;
              z-index: 1;
              text-transform: uppercase;
              font-family: sans-serif;
              font-weight: 600;
              line-height: 26px;
              pointer-events: none;
            }
            
            .mceContentBody.mceContentBody {
              padding-right: 25px !important;
              padding-left: 15px !important;
              border-right: 1px solid #eee;
              position: relative;
              
            }
            .mceContentBody.mceContentBody[data-site_layout="none"] {
                max-width: 1170px;
                
              }
            .mceContentBody.mceContentBody[data-site_layout="none"]:after {
                  content: '';
             }
CSS;


            if (($sidebar_layout == 'left' && is_active_sidebar($left_sidebar))
            || ($sidebar_layout == 'right' && is_active_sidebar($right_sidebar))
            ) {
                $content_width = $sidebar_width;

                $custom_css .= <<<CSS
				.mceContentBody::after {
				    content: '{$sidebar_text}';
				}
CSS;
            }


            $custom_css .= <<<CSS
            

			.mceContentBody[data-site_layout="left"],
			.mceContentBody[data-site_layout="right"]{
			    max-width: {$sidebar_width}px;
			}
			
			.mceContentBody[data-site_layout="left"]::after,
			 .mceContentBody[data-site_layout="right"]::after{
				    content: '{$sidebar_text}';
				}

			.mceContentBody {
				max-width: {$content_width}px;
			}
			
CSS;
        }


        $custom_fonts = array(
            'body_font' => array(
                'default' => array(
                    'font-size'=>'13px',
                    'font-family'=>'Roboto',
                    'font-weight'=>'400',
                    'line-height'=>'24px',
                ),
                'selector' => array('body')
            ) ,
            'h1_font' => array(
                'default' =>  array(
                    'font-size'=>'36px',
                    'font-family' => 'Oswald',
                    'font-weight'=>'700',
                    'line-height'=>'48px',
                ),
                'selector' => array('h1')
            ),
            'h2_font' => array(
                'default' =>  array(
                    'font-size'=>'29px',
                    'font-family' => 'Oswald',
                    'font-weight'=>'700',
                    'line-height'=>'39px',
                ),
                'selector' => array('h2')
            ),
            'h3_font' => array(
                'default' =>  array(
                    'font-size'=>'22px',
                    'font-family' => 'Oswald',
                    'font-weight'=>'400',
                    'line-height'=>'29px',
                ),
                'selector' => array('h3')
            ),
            'h4_font' => array(
                'default' =>  array(
                    'font-size'=>'17px',
                    'font-family' => 'Oswald',
                    'font-weight'=>'400',
                    'line-height'=>'23px',
                ),
                'selector' => array('h4')
            ),
            'h5_font' => array(
                'default' =>  array(
                    'font-size'=>'13px',
                    'font-family' => 'Oswald',
                    'font-weight'=>'400',
                    'line-height'=>'17px',
                ),
                'selector' => array('h5')
            ),
            'h6_font'  => array(
                'default' =>  array(
                    'font-size'     =>'11px',
                    'font-family'   => 'Oswald',
                    'font-weight'   =>'400',
                    'line-height'   => '15px'
                ),
                'selector' => array('h6')
            )
        );

        foreach ($custom_fonts as $optionKey => $v) {
            $fonts = isset($g5plus_options[$optionKey]) ? $g5plus_options[$optionKey] : $v;
            if ($fonts) {
                $selector = implode(',', $v['selector']);
                $fonts = g5plus_process_font($fonts);
                $fonts_attributes = array();
                if (isset($fonts['font-family'])) {
                    $fonts['font-family'] = g5plus_get_font_family($fonts['font-family']);
                    $fonts_attributes[] = "font-family: '{$fonts['font-family']}'";
                }

                if (isset($fonts['font-size'])) {
                    $fonts_attributes[] = "font-size: {$fonts['font-size']}";
                }

                if (isset($fonts['font-weight'])) {
                    $fonts_attributes[] = "font-weight: {$fonts['font-weight']}";
                }

                if (isset($fonts['font-style'])) {
                    $fonts_attributes[] = "font-style: {$fonts['font-style']}";
                }

                if (sizeof($fonts_attributes) > 0) {
                    $fonts_css = implode(';', $fonts_attributes);

                    $custom_css .= <<<CSS
                {$selector} {
                    {$fonts_css}
                }
CSS;
                }
            }
        }

        // Remove comments
        $custom_css = preg_replace('!/\*[^*]*\*+([^/][^*]*\*+)*/!', '', $custom_css);
        // Remove space after colons
        $custom_css = str_replace(': ', ':', $custom_css);
        // Remove whitespace
        $custom_css = str_replace(array("\r\n", "\r", "\n", "\t", '  ', '    ', '    '), '', $custom_css);
        return $custom_css;
    }
}

if (!function_exists('g5plus_enqueue_block_editor_assets')) {
    function g5plus_enqueue_block_editor_assets() {
        wp_enqueue_style('font-awesome', G5PLUS_THEME_URL . 'assets/plugins/fonts-awesome/css/font-awesome.min.css', array(),'4.5.0');

        wp_enqueue_style('block-editor', G5PLUS_THEME_URL . '/assets/css/editor-blocks.css');

        $screen = get_current_screen();
        $post_id = '';
        if ( is_admin() && ($screen->id == 'post') ) {
            global $post;
            $post_id = $post->ID;
        }

        wp_enqueue_style('gsf_custom_css_block_editor', admin_url('admin-ajax.php') . '?action=gsf_custom_css_block_editor&post_id=' . $post_id);

        $g5plus_options = &G5Plus_Global::get_options();
        $custom_fonts_variable = array(
            'body_font' => array(
                'font-size'=>'13px',
                'font-family'=>'Roboto',
                'font-weight'=>'400',
                'line-height'=>'24px',
            ),
            'h1_font' => array(
                'font-size'=>'36px',
                'font-family' => 'Oswald',
                'font-weight'=>'700',
                'line-height'=>'48px',
            ),
            'h2_font' => array(
                'font-size'=>'29px',
                'font-family' => 'Oswald',
                'font-weight'=>'700',
                'line-height'=>'39px',
            ),
            'h3_font' => array(
                'font-size'=>'22px',
                'font-family' => 'Oswald',
                'font-weight'=>'400',
                'line-height'=>'29px',
            ),
            'h4_font' => array(
                'font-size'=>'17px',
                'font-family' => 'Oswald',
                'font-weight'=>'400',
                'line-height'=>'23px',
            ),
            'h5_font' => array(
                'font-size'=>'13px',
                'font-family' => 'Oswald',
                'font-weight'=>'400',
                'line-height'=>'17px',
            ),
            'h6_font'  => array(
                'font-size'     =>'11px',
                'font-family'   => 'Oswald',
                'font-weight'   =>'400',
                'line-height'   => '15px'
            ),
            'primary_font' => array(
                'font-family'=>'Oswald',
            ),
            'secondary_font' => array(
                'font-family'=>'Roboto',
            )
        );
        $google_fonts = array();
        foreach ($custom_fonts_variable as $k => $v) {
            $custom_fonts = isset($g5plus_options[$k]) ? $g5plus_options[$k] : $v;
            if ($custom_fonts && isset($custom_fonts['font-family']) && !in_array($custom_fonts['font-family'],$google_fonts)) {
                $google_fonts[] = $custom_fonts['font-family'];
            }
        }
        $fonts = '';
        foreach($google_fonts as $google_font)
        {
            $fonts .= str_replace('','+',$google_font) . ':100,300,400,600,700,900,100italic,300italic,400italic,600italic,700italic,900italic|';
        }
        if ($fonts != '')
        {
            $protocol = is_ssl() ? 'https' : 'http';
            wp_enqueue_style('google-fonts',$protocol . '://fonts.googleapis.com/css?family=' . substr_replace( $fonts, "", - 1 ));
        }

    }
    add_action('enqueue_block_editor_assets','g5plus_enqueue_block_editor_assets');
}

if (!function_exists('g5plus_custom_css_block_editor_callback')) {
    function g5plus_custom_css_block_editor_callback() {
        $custom_css = g5plus_custom_css_block_editor();

        /**
         * Make sure we set the correct MIME type
         */
        header( 'Content-Type: text/css' );
        /**
         * Render RTL CSS
         */
        echo sprintf('%s',$custom_css);
        die();
    }
    add_action( 'wp_ajax_gsf_custom_css_block_editor', 'g5plus_custom_css_block_editor_callback');
    add_action( 'wp_ajax_nopriv_gsf_custom_css_block_editor', 'g5plus_custom_css_block_editor_callback');
}

if (!function_exists('g5plus_custom_css_block_editor')) {
    function g5plus_custom_css_block_editor() {
        $post_id = isset($_GET['post_id']) ? $_GET['post_id'] : '';



        $g5plus_options = &G5Plus_Global::get_options();
        $prefix = 'g5plus_';
        $sidebar_layout = rwmb_meta($prefix.'page_sidebar', array(), $post_id);
        if (($sidebar_layout === '') || ($sidebar_layout == '-1')) {
            $sidebar_layout = $g5plus_options['single_blog_sidebar'];
        }

        $sidebar_width = rwmb_meta($prefix.'sidebar_width', array(), $post_id);
        if (($sidebar_width === '') || ($sidebar_width == '-1')) {
            $sidebar_width = $g5plus_options['single_blog_sidebar_width'];
        }

        $left_sidebar = rwmb_meta($prefix.'page_left_sidebar', array(), $post_id);
        if (($left_sidebar === '') || ($left_sidebar == '-1')) {
            $left_sidebar = $g5plus_options['single_blog_left_sidebar'];
        }

        $right_sidebar = rwmb_meta($prefix.'page_right_sidebar', array(), $post_id);
        if (($right_sidebar === '') || ($right_sidebar == '-1')) {
            $right_sidebar = $g5plus_options['single_blog_right_sidebar'];
        }


        $content_width = 1170;
        if ($sidebar_width === 'large') {
            $sidebar_width = 770;
        } else {
            $sidebar_width = 870;
        }

        $custom_css = '';
        if (($sidebar_layout == 'left' && is_active_sidebar($left_sidebar))
            || ($sidebar_layout == 'right' && is_active_sidebar($right_sidebar))
        ) {
            $content_width = $sidebar_width;
        }
        $custom_css .= <<<CSS
            
            .edit-post-layout__content[data-site_layout="left"] .wp-block,
			.edit-post-layout__content[data-site_layout="right"] .wp-block,
			.edit-post-layout__content[data-site_layout="left"] .wp-block[data-align="wide"],
			.edit-post-layout__content[data-site_layout="right"] .wp-block[data-align="wide"],
			.edit-post-layout__content[data-site_layout="left"] .wp-block[data-align="full"],
			.edit-post-layout__content[data-site_layout="right"] .wp-block[data-align="full"]{
			    max-width: {$sidebar_width}px;
			}
			
			.wp-block[data-align="full"] {
			    margin-left: auto;
			    margin-right: auto;
			}
			
            
            .wp-block,
            .wp-block[data-align="wide"],
             .wp-block[data-align="full"]{
                max-width: {$content_width}px;
            }
			
CSS;


        $custom_fonts = array(
            'body_font' => array(
                'default' => array(
                    'font-size'=>'13px',
                    'font-family'=>'Roboto',
                    'font-weight'=>'400',
                    'line-height'=>'24px',
                ),
                'selector' => array('.editor-styles-wrapper.editor-styles-wrapper')
            ) ,
            'h1_font' => array(
                'default' =>  array(
                    'font-size'=>'36px',
                    'font-family' => 'Oswald',
                    'font-weight'=>'700',
                    'line-height'=>'48px',
                ),
                'selector' => array('.editor-styles-wrapper.editor-styles-wrapper h1')
            ),
            'h2_font' => array(
                'default' =>  array(
                    'font-size'=>'29px',
                    'font-family' => 'Oswald',
                    'font-weight'=>'700',
                    'line-height'=>'39px',
                ),
                'selector' => array('.editor-styles-wrapper.editor-styles-wrapper h2')
            ),
            'h3_font' => array(
                'default' =>  array(
                    'font-size'=>'22px',
                    'font-family' => 'Oswald',
                    'font-weight'=>'400',
                    'line-height'=>'29px',
                ),
                'selector' => array('.editor-styles-wrapper.editor-styles-wrapper h3')
            ),
            'h4_font' => array(
                'default' =>  array(
                    'font-size'=>'17px',
                    'font-family' => 'Oswald',
                    'font-weight'=>'400',
                    'line-height'=>'23px',
                ),
                'selector' => array('.editor-styles-wrapper.editor-styles-wrapper h4')
            ),
            'h5_font' => array(
                'default' =>  array(
                    'font-size'=>'13px',
                    'font-family' => 'Oswald',
                    'font-weight'=>'400',
                    'line-height'=>'17px',
                ),
                'selector' => array('.editor-styles-wrapper.editor-styles-wrapper h5')
            ),
            'h6_font'  => array(
                'default' =>  array(
                    'font-size'     =>'11px',
                    'font-family'   => 'Oswald',
                    'font-weight'   =>'400',
                    'line-height'   => '15px'
                ),
                'selector' => array('.editor-styles-wrapper.editor-styles-wrapper h6')
            ),
        );

        foreach ($custom_fonts as $optionKey => $v) {
            $fonts = isset($g5plus_options[$optionKey]) ? $g5plus_options[$optionKey] : $v;
            if ($fonts) {
                $selector = implode(',', $v['selector']);
                $fonts = g5plus_process_font($fonts);
                $fonts_attributes = array();
                if (isset($fonts['font-family'])) {
                    $fonts['font-family'] = g5plus_get_font_family($fonts['font-family']);
                    $fonts_attributes[] = "font-family: '{$fonts['font-family']}'";
                }

                if (isset($fonts['font-size'])) {
                    $fonts_attributes[] = "font-size: {$fonts['font-size']}";
                }

                if (isset($fonts['font-weight'])) {
                    $fonts_attributes[] = "font-weight: {$fonts['font-weight']}";
                }

                if (isset($fonts['font-style'])) {
                    $fonts_attributes[] = "font-style: {$fonts['font-style']}";
                }

                if (sizeof($fonts_attributes) > 0) {
                    $fonts_css = implode(';', $fonts_attributes);

                    $custom_css .= <<<CSS
                {$selector} {
                    {$fonts_css}
                }
CSS;
                }
            }
        }

        // Remove comments
        $custom_css = preg_replace('!/\*[^*]*\*+([^/][^*]*\*+)*/!', '', $custom_css);
        // Remove space after colons
        $custom_css = str_replace(': ', ':', $custom_css);
        // Remove whitespace
        $custom_css = str_replace(array("\r\n", "\r", "\n", "\t", '  ', '    ', '    '), '', $custom_css);

        return $custom_css;
    }
}



