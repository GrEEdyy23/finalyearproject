<?php

function univero_woocommerce_setup() {
    global $pagenow;
    if ( is_admin() && isset($_GET['activated'] ) && $pagenow == 'themes.php' ) {
        $catalog = array(
            'width'     => '330',   // px
            'height'    => '330',   // px
            'crop'      => 1        // true
        );

        $single = array(
            'width'     => '660',   // px
            'height'    => '660',   // px
            'crop'      => 1        // true
        );

        $thumbnail = array(
            'width'     => '130',    // px
            'height'    => '130',   // px
            'crop'      => 1        // true
        );

        // Image sizes
        update_option( 'shop_catalog_image_size', $catalog );       // Product category thumbs
        update_option( 'shop_single_image_size', $single );         // Single product image
        update_option( 'shop_thumbnail_image_size', $thumbnail );   // Image gallery thumbs
    }
}

add_action( 'init', 'univero_woocommerce_setup');


if ( !function_exists('univero_get_products') ) {
    function univero_get_products($categories = array(), $product_type = 'featured_product', $paged = 1, $post_per_page = -1, $orderby = '', $order = '') {
        global $woocommerce, $wp_query;
        $args = array(
            'post_type' => 'product',
            'posts_per_page' => $post_per_page,
            'post_status' => 'publish',
            'paged' => $paged,
            'orderby'   => $orderby,
            'order' => $order
        );

        if ( isset( $args['orderby'] ) ) {
            if ( 'price' == $args['orderby'] ) {
                $args = array_merge( $args, array(
                    'meta_key'  => '_price',
                    'orderby'   => 'meta_value_num'
                ) );
            }
            if ( 'featured' == $args['orderby'] ) {
                $args = array_merge( $args, array(
                    'meta_key'  => '_featured',
                    'orderby'   => 'meta_value'
                ) );
            }
            if ( 'sku' == $args['orderby'] ) {
                $args = array_merge( $args, array(
                    'meta_key'  => '_sku',
                    'orderby'   => 'meta_value'
                ) );
            }
        }

        switch ($product_type) {
            case 'best_selling':
                $args['meta_key']='total_sales';
                $args['orderby']='meta_value_num';
                $args['ignore_sticky_posts']   = 1;
                $args['meta_query'] = array();
                $args['meta_query'][] = $woocommerce->query->stock_status_meta_query();
                $args['meta_query'][] = $woocommerce->query->visibility_meta_query();
                break;
            case 'featured_product':
                $product_visibility_term_ids = wc_get_product_visibility_term_ids();
                $args['tax_query'][] = array(
                    'taxonomy' => 'product_visibility',
                    'field'    => 'term_taxonomy_id',
                    'terms'    => $product_visibility_term_ids['featured'],
                );
                break;
            case 'top_rate':
                add_filter( 'posts_clauses',  array( $woocommerce->query, 'order_by_rating_post_clauses' ) );
                $args['meta_query'] = array();
                $args['meta_query'][] = $woocommerce->query->stock_status_meta_query();
                $args['meta_query'][] = $woocommerce->query->visibility_meta_query();
                break;
            case 'recent_product':
                $args['meta_query'] = array();
                $args['meta_query'][] = $woocommerce->query->stock_status_meta_query();
                break;
            case 'deals':
                $args['meta_query'] = array();
                $args['meta_query'][] = $woocommerce->query->stock_status_meta_query();
                $args['meta_query'][] = $woocommerce->query->visibility_meta_query();
                $args['meta_query'][] =  array(
                    array(
                        'key'           => '_sale_price_dates_to',
                        'value'         => time(),
                        'compare'       => '>',
                        'type'          => 'numeric'
                    )
                );
                break;     
            case 'on_sale':
                $product_ids_on_sale    = wc_get_product_ids_on_sale();
                $product_ids_on_sale[]  = 0;
                $args['post__in'] = $product_ids_on_sale;
                break;
            case 'recent_review':
                if($post_per_page == -1) $_limit = 4;
                else $_limit = $post_per_page;
                global $wpdb;
                $query = "SELECT c.comment_post_ID FROM {$wpdb->prefix}posts p, {$wpdb->prefix}comments c
                        WHERE p.ID = c.comment_post_ID AND c.comment_approved > 0 AND p.post_type = 'product' AND p.post_status = 'publish' AND p.comment_count > 0
                        ORDER BY c.comment_date ASC";
                $results = $wpdb->get_results($query, OBJECT);
                $_pids = array();
                foreach ($results as $re) {
                    if(!in_array($re->comment_post_ID, $_pids))
                        $_pids[] = $re->comment_post_ID;
                    if(count($_pids) == $_limit)
                        break;
                }

                $args['meta_query'] = array();
                $args['meta_query'][] = $woocommerce->query->stock_status_meta_query();
                $args['meta_query'][] = $woocommerce->query->visibility_meta_query();
                $args['post__in'] = $_pids;

                break;
        }

        if ( !empty($categories) && is_array($categories) ) {
            $args['tax_query']    = array(
                array(
                    'taxonomy'      => 'product_cat',
                    'field'         => 'slug',
                    'terms'         => implode(",", $categories ),
                    'operator'      => 'IN'
                )
            );
        }
        
        return new WP_Query($args);
    }
}

// cart modal
if ( !function_exists('univero_woocommerce_cart_modal') ) {
    function univero_woocommerce_cart_modal() {
        wc_get_template( 'content-product-cart-modal.php' , array( 'current_product_id' => (int)$_GET['product_id'] ) );
        die;
    }
}

add_action( 'wp_ajax_univero_add_to_cart_product', 'univero_woocommerce_cart_modal' );
add_action( 'wp_ajax_nopriv_univero_add_to_cart_product', 'univero_woocommerce_cart_modal' );


// hooks
if ( !function_exists('univero_woocommerce_enqueue_styles') ) {
    function univero_woocommerce_enqueue_styles() {
        wp_enqueue_style( 'univero-woocommerce', get_template_directory_uri() . '/css/woocommerce.css' , 'univero-woocommerce-front' , UNIVERO_THEME_VERSION, 'all' );
        wp_register_script( 'univero-woocommerce', get_template_directory_uri() . '/js/woocommerce.js', array( 'jquery' ), '20150330', true );
        wp_enqueue_script( 'univero-woocommerce' );

        wp_enqueue_script( 'wc-add-to-cart-variation' );
    }
}
add_action( 'wp_enqueue_scripts', 'univero_woocommerce_enqueue_styles', 99 );

// cart
if ( !function_exists('univero_woocommerce_header_add_to_cart_fragment') ) {
    function univero_woocommerce_header_add_to_cart_fragment( $fragments ){
        global $woocommerce;
        $fragments['#cart .count'] =  sprintf(_n(' <span class="count"> %d  </span> ', ' <span class="count"> %d </span> ', $woocommerce->cart->cart_contents_count, 'univero'), $woocommerce->cart->cart_contents_count);
        $fragments['#cart .total-price'] = trim( $woocommerce->cart->get_cart_total() );
        return $fragments;
    }
}
add_filter('woocommerce_add_to_cart_fragments', 'univero_woocommerce_header_add_to_cart_fragment' );

// breadcrumb for woocommerce page
if ( !function_exists('univero_woocommerce_breadcrumb_defaults') ) {
    function univero_woocommerce_breadcrumb_defaults( $args ) {
        $key = 'products';
        if ( is_singular('product') ) {
            $key = 'product';
        }
        $layout = univero_get_config($key.'_breadcrumbs_layout', 'layout1');
        $breadcrumb_img = univero_get_config($key.'_breadcrumb_image');
        $breadcrumb_color = univero_get_config($key.'_breadcrumb_color');
        $style = array();
        $breadcrumb_enable = univero_get_config('show_'.$key.'_breadcrumbs');
        $archive = '';
        if ( !$breadcrumb_enable ) {
            $style[] = 'display:none';
        }
        if( $breadcrumb_color  ){
            $style[] = 'background-color:'.$breadcrumb_color;
        }
        if ( isset($breadcrumb_img['url']) && !empty($breadcrumb_img['url']) ) {
            $style[] = 'background-image:url(\''.esc_url($breadcrumb_img['url']).'\')';
        }
        $estyle = !empty($style)? ' style="'.implode(";", $style).'"':"";

        $title = univero_get_config($key.'_breadcrumb_title', 'Shop');

        if ( $layout == 'layout1' ) {
            $args['wrap_before'] = '<section id="ninzio-breadscrumb" class="ninzio-breadscrumb '.$layout.'">
                <div class="ninzio-breadscrumb-top" '.$estyle.'>
                    <div class="breadscrumb-title">
                        <h2 class="bread-title">'.$title.'</h2>
                    </div>
                    <div class="ninzio-breadscrumb-bottom">
                        <div class="container">
                            <div class="breadscrumb-inner">
                                <ol class="ninzio-woocommerce-breadcrumb breadcrumb" ' . ( is_single() ? 'itemprop="breadcrumb"' : '' ) . '>';
                $args['wrap_after'] = '</ol>
                            </div>
                        </div>
                    </div>
                </div>
            </section>';

        } else {
            $args['wrap_before'] = '<section id="ninzio-breadscrumb" class="ninzio-breadscrumb '.$layout.'"'.$estyle.'>
                <div class="container">
                    <div class="wrapper-breads">
                        <div class="breadscrumb-inner">
                            <div class="breadscrumb-title pull-left"><h2 class="bread-title">'.$title.'</h2></div>
                            <div class="pull-right">
                                <ol class="ninzio-woocommerce-breadcrumb breadcrumb" ' . ( is_single() ? 'itemprop="breadcrumb"' : '' ) . '>';
            $args['wrap_after'] = '</ol>
                            </div>
                        </div>
                    </div>
                </div>
            </section>';
        }
        

        return $args;
    }
}
add_filter( 'woocommerce_breadcrumb_defaults', 'univero_woocommerce_breadcrumb_defaults' );
add_action( 'univero_woo_template_main_before', 'woocommerce_breadcrumb', 30, 0 );



if(!function_exists('univero_filter_before')){
    function univero_filter_before(){
        echo '<div class="ninzio-filter clearfix">';
    }
}
if(!function_exists('univero_filter_after')){
    function univero_filter_after(){
        echo '</div>';
    }
}
add_action( 'woocommerce_before_shop_loop', 'univero_filter_before' , 1 );
add_action( 'woocommerce_before_shop_loop', 'univero_filter_after' , 40 );

// set display mode to cookie
if ( !function_exists('univero_before_woocommerce_init') ) {
    function univero_before_woocommerce_init() {
        if( isset($_GET['display_mode']) && ($_GET['display_mode']=='list' || $_GET['display_mode']=='grid') ){  
            setcookie( 'univero_woo_mode', trim($_GET['display_mode']) , time()+3600*24*100,'/' );
            $_COOKIE['univero_woo_mode'] = trim($_GET['display_mode']);
        }
    }
}
add_action( 'init', 'univero_before_woocommerce_init' );

// Number of products per page
if ( !function_exists('univero_woocommerce_shop_per_page') ) {
    function univero_woocommerce_shop_per_page($number) {
        $value = univero_get_config('number_products_per_page');
        if ( is_numeric( $value ) && $value ) {
            $number = absint( $value );
        }
        return $number;
    }
}
add_filter( 'loop_shop_per_page', 'univero_woocommerce_shop_per_page' );

// Number of products per row
if ( !function_exists('univero_woocommerce_shop_columns') ) {
    function univero_woocommerce_shop_columns($number) {
        $value = univero_get_config('product_columns');
        if ( in_array( $value, array(2, 3, 4, 6) ) ) {
            $number = $value;
        }
        return $number;
    }
}
add_filter( 'loop_shop_columns', 'univero_woocommerce_shop_columns' );

// share box
if ( !function_exists('univero_woocommerce_share_box') ) {
    function univero_woocommerce_share_box() {
        if ( univero_get_config('show_product_social_share') ) {
            get_template_part( 'template-parts/sharebox-product' );
        }
    }
}
add_filter( 'woocommerce_single_product_summary', 'univero_woocommerce_share_box', 100 );

// quickview
if ( !function_exists('univero_woocommerce_quickview') ) {
    function univero_woocommerce_quickview() {
        $args = array(
            'post_type'=>'product',
            'product' => $_GET['productslug']
        );
        $query = new WP_Query($args);
        if ( $query->have_posts() ) {
            while ($query->have_posts()): $query->the_post(); global $product;
                wc_get_template_part( 'content', 'product-quickview' );
            endwhile;
        }
        wp_reset_postdata();
        die;
    }
}

function univero_woocommerce_quickview_init() {
    if ( univero_get_config('show_quickview') ) {
        add_action( 'wp_ajax_univero_quickview_product', 'univero_woocommerce_quickview' );
        add_action( 'wp_ajax_nopriv_univero_quickview_product', 'univero_woocommerce_quickview' );
    }
}
add_action( 'init', 'univero_woocommerce_quickview_init' );


// swap effect
if ( !function_exists('univero_swap_images') ) {
    function univero_swap_images($size = 'shop_catalog') {
        global $post, $product, $woocommerce;
        
        $output = '';
        $class = 'image-no-effect unveil-image';
        if (has_post_thumbnail()) {
            $product_thumbnail_id = get_post_thumbnail_id();
            $product_thumbnail_title = get_the_title( $product_thumbnail_id );
            $product_thumbnail = wp_get_attachment_image_src( $product_thumbnail_id, $size );
            $placeholder_image = univero_create_placeholder(array($product_thumbnail[1],$product_thumbnail[2]));

            if ( univero_get_config('show_swap_image') ) {
                $attachment_ids = $product->get_gallery_image_ids();
                if ($attachment_ids && isset($attachment_ids[0])) {
                    $class = 'image-hover';
                    $product_thumbnail_hover_title = get_the_title( $attachment_ids[0] );
                    $product_thumbnail_hover = wp_get_attachment_image_src( $attachment_ids[0], $size );
                    
                    if ( univero_get_config('image_lazy_loading') ) {
                        echo '<img src="' . trim( $placeholder_image ) . '" data-src="' . esc_url( $product_thumbnail_hover[0] ) . '" width="' . esc_attr( $product_thumbnail_hover[1] ) . '" height="' . esc_attr( $product_thumbnail_hover[2] ) . '" alt="' . esc_attr( $product_thumbnail_hover_title ) . '" class="attachment-shop-catalog unveil-image image-effect" />';
                    } else {
                        echo '<img src="' . esc_url( $product_thumbnail_hover[0] ) . '" width="' . esc_attr( $product_thumbnail_hover[1] ) . '" height="' . esc_attr( $product_thumbnail_hover[2] ) . '" alt="' . esc_attr( $product_thumbnail_hover_title ) . '" class="attachment-shop-catalog image-effect" />';
                    }
                }
            }
            
            if ( univero_get_config('image_lazy_loading') ) {
                echo '<img src="' . trim( $placeholder_image ) . '" data-src="' . esc_url( $product_thumbnail[0] ) . '" width="' . esc_attr( $product_thumbnail[1] ) . '" height="' . esc_attr( $product_thumbnail[2] ) . '" alt="' . esc_attr( $product_thumbnail_title ) . '" class="attachment-shop-catalog unveil-image '.esc_attr($class).'" />';
            } else {
                echo '<img src="' . esc_url( $product_thumbnail[0] ) . '" width="' . esc_attr( $product_thumbnail[1] ) . '" height="' . esc_attr( $product_thumbnail[2] ) . '" alt="' . esc_attr( $product_thumbnail_title ) . '" class="attachment-shop-catalog '.esc_attr($class).'" />';
            }
        } else {
            $image_sizes = get_option('shop_catalog_image_size');
            $placeholder_width = $image_sizes['width'];
            $placeholder_height = $image_sizes['height'];

            $output .= '<img src="'.wc_placeholder_img_src().'" alt="'.esc_attr__('Placeholder' , 'univero').'" class="'.$class.'" width="'.$placeholder_width.'" height="'.$placeholder_height.'" />';
        }
        echo wp_kses_post($output);
    }
}


// get image
if ( !function_exists('univero_product_get_image') ) {
    function univero_product_get_image($thumb = 'shop_thumbnail') {
        global $product;

        $product_thumbnail_id = get_post_thumbnail_id();
        $product_thumbnail_title = get_the_title( $product_thumbnail_id );
        $product_thumbnail = wp_get_attachment_image_src( $product_thumbnail_id, $thumb );
        
        $placeholder_image = univero_create_placeholder(array($product_thumbnail[1],$product_thumbnail[2]));

        echo '<div class="product-image">';
        if ( univero_get_config('image_lazy_loading') ) {
            echo '<img src="' . trim( $placeholder_image ) . '" data-src="' . esc_url( $product_thumbnail[0] ) . '" width="' . esc_attr( $product_thumbnail[1] ) . '" height="' . esc_attr( $product_thumbnail[2] ) . '" alt="' . esc_attr( $product_thumbnail_title ) . '" class="attachment-'.esc_attr($thumb).' size-'.esc_attr($thumb).' wp-post-image unveil-image" />';
        } else {
            echo '<img src="' . esc_url( $product_thumbnail[0] ) . '" width="' . esc_attr( $product_thumbnail[1] ) . '" height="' . esc_attr( $product_thumbnail[2] ) . '" alt="' . esc_attr( $product_thumbnail_title ) . '" class="attachment-'.esc_attr($thumb).' size-'.esc_attr($thumb).' wp-post-image" />';
        }
        echo '</div>';
    }
}

// layout class for woo page
if ( !function_exists('univero_woocommerce_content_class') ) {
    function univero_woocommerce_content_class( $class ) {
        $page = 'archive';
        if ( is_singular( 'product' ) ) {
            $page = 'single';
        }
        if( univero_get_config('product_'.$page.'_fullwidth') ) {
            return 'container-fluid';
        }
        return $class;
    }
}
add_filter( 'univero_woocommerce_content_class', 'univero_woocommerce_content_class' );

// get layout configs
if ( !function_exists('univero_get_woocommerce_layout_configs') ) {
    function univero_get_woocommerce_layout_configs() {
        $page = 'archive';
        if ( is_singular( 'product' ) ) {
            $page = 'single';
        }
        $left = univero_get_config('product_'.$page.'_left_sidebar');
        $right = univero_get_config('product_'.$page.'_right_sidebar');

        switch ( univero_get_config('product_'.$page.'_layout') ) {
            case 'left-main':
                $configs['left'] = array( 'sidebar' => $left, 'class' => 'col-md-3'  );
                $configs['main'] = array( 'class' => 'col-md-9 ' );
                break;
            case 'main-right':
                $configs['right'] = array( 'sidebar' => $right,  'class' => 'col-md-3' ); 
                $configs['main'] = array( 'class' => 'col-md-9 ' );
                break;
            case 'main':
                $configs['main'] = array( 'class' => 'col-md-12' );
                break;
            case 'left-main-right':
                $configs['left'] = array( 'sidebar' => $left,  'class' => 'col-md-3'  );
                $configs['right'] = array( 'sidebar' => $right, 'class' => 'col-md-3' ); 
                $configs['main'] = array( 'class' => 'col-md-6 ' );
                break;
            default:
                $configs['main'] = array( 'class' => 'col-md-12' );
                break;
        }

        return $configs; 
    }
}

// Show/Hide related, upsells products
if ( !function_exists('univero_woocommerce_related_upsells_products') ) {
    function univero_woocommerce_related_upsells_products($located, $template_name) {
        $content_none = get_template_directory() . '/woocommerce/content-none.php';
        $show_product_releated = univero_get_config('show_product_releated');
        if ( 'single-product/related.php' == $template_name ) {
            if ( !$show_product_releated  ) {
                $located = $content_none;
            }
        } elseif ( 'single-product/up-sells.php' == $template_name ) {
            $show_product_upsells = univero_get_config('show_product_upsells');
            if ( !$show_product_upsells ) {
                $located = $content_none;
            }
        }

        return apply_filters( 'univero_woocommerce_related_upsells_products', $located, $template_name );
    }
}
add_filter( 'wc_get_template', 'univero_woocommerce_related_upsells_products', 10, 2 );

if ( !function_exists( 'univero_product_tabs' ) ) {
    function univero_product_tabs($tabs) {
        global $product, $post;
        
        if ( !univero_get_config('show_product_review_tab') && isset($tabs['reviews']) ) {
            unset( $tabs['reviews'] ); 
        }

        return $tabs;
    }
}
add_filter( 'woocommerce_product_tabs', 'univero_product_tabs', 90 );


function univero_woocommerce_get_ajax_products() {
    $categories = isset($_POST['categories']) ? $_POST['categories'] : '';
    $columns = isset($_POST['columns']) ? $_POST['columns'] : 4;
    $number = isset($_POST['number']) ? $_POST['number'] : 4;
    $product_type = isset($_POST['product_type']) ? $_POST['product_type'] : '';
    $layout_type = isset($_POST['layout_type']) ? $_POST['layout_type'] : '';

    $categories_id = !empty($categories) ? array($categories) : array();
    $loop = ninzio_themer_get_products( $categories_id, $product_type, 1, $number );
    if ( $loop->have_posts()) {
        wc_get_template( 'layout-products/'.$layout_type.'.php' , array( 'loop' => $loop, 'columns' => $columns, 'number' => $number ) );
    }
    exit();
}
add_action( 'wp_ajax_univero_get_products', 'univero_woocommerce_get_ajax_products' );
add_action( 'wp_ajax_nopriv_univero_get_products', 'univero_woocommerce_get_ajax_products' );


function univero_show_percent_disount() {
    global $product;
    $regular_price = $product->get_regular_price();
    $sale_price = $product->get_sale_price();
    
    if ( !empty($sale_price) ) {
        $percentage = round( ( ( $regular_price - $sale_price ) / $regular_price ) * 100 );

        return $percentage.esc_html__('%', 'univero');
    } else {
        return '';
    }
}

