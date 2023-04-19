<?php

function univero_get_events( $type, $number = -1 ) {
    switch ( $type ) {
        case 'most_recent' : 
           $args = array( 
                'posts_per_page' => $number, 
                'orderby' => 'date', 
                'order' => 'DESC',
                'post_type' => 'tribe_events'
            );
            break;

        case 'featured' :
            $args = array( 
                'posts_per_page' => $number, 
                'orderby' => 'date', 
                'order' => 'DESC',
                'post_type' => 'tribe_events',
                'meta_query' => array( array(
                    'key'     => '_tribe_featured',
                    'value' => '1'
                ) ),
            );
            break;
        case 'random' : 
            $args = array(
                'post_type' => 'tribe_events',
                'posts_per_page' => $number, 
                'orderby' => 'rand'
            );
            break;
        default : 
            $args = array(
                'post_type' => 'tribe_events',
                'posts_per_page' => $number, 
                'orderby' => 'rand'
            );
            break;
    }
    $wp_query = new WP_Query( $args );
    return $wp_query;
}

// layout class for event page
if ( !function_exists('univero_event_content_class') ) {
    function univero_event_content_class( $class ) {
        $page = 'archive';
        if ( is_singular( 'tribe_events' ) ) {
            $page = 'single';
        }
        if( univero_get_config('event_'.$page.'_fullwidth') ) {
            return 'container-fluid';
        }
        return $class;
    }
}
add_filter( 'univero_event_content_class', 'univero_event_content_class' );

// get layout event
if ( !function_exists('univero_get_event_layout_configs') ) {
    function univero_get_event_layout_configs() {
        $page = 'archive';
        if ( is_singular( 'tribe_events' ) ) {
            $page = 'single';
        }
        $left = univero_get_config('event_'.$page.'_left_sidebar');
        $right = univero_get_config('event_'.$page.'_right_sidebar');

        switch ( univero_get_config('event_'.$page.'_layout') ) {
            case 'left-main':
                $configs['left'] = array( 'sidebar' => $left, 'class' => 'col-md-3 col-sm-12 col-xs-12'  );
                $configs['main'] = array( 'class' => 'col-md-9 col-sm-12 col-xs-12' );
                break;
            case 'main-right':
                $configs['right'] = array( 'sidebar' => $right,  'class' => 'col-md-3 col-sm-12 col-xs-12' ); 
                $configs['main'] = array( 'class' => 'col-md-9 col-sm-12 col-xs-12' );
                break;
            case 'main':
                $configs['main'] = array( 'class' => 'col-md-12 col-sm-12 col-xs-12' );
                break;
            case 'left-main-right':
                $configs['left'] = array( 'sidebar' => $left,  'class' => 'col-md-3 col-sm-12 col-xs-12'  );
                $configs['right'] = array( 'sidebar' => $right, 'class' => 'col-md-3 col-sm-12 col-xs-12' ); 
                $configs['main'] = array( 'class' => 'col-md-6 col-sm-12 col-xs-12' );
                break;
            default:
                $configs['main'] = array( 'class' => 'col-md-12 col-sm-12 col-xs-12' );
                break;
        }

        return $configs; 
    }
}

function univero_tribe_events_event_schedule_start( $event = null, $before = '', $after = '' ) {
    if ( is_null( $event ) ) {
        global $post;
        $event = $post;
    }

    if ( is_numeric( $event ) ) {
        $event = get_post( $event );
    }

    $inner                    = '<span class="tribe-event-date-start">';
    $format                   = '';
    $date_without_year_format = tribe_get_date_format();
    $date_with_year_format    = tribe_get_date_format( true );
    $time_format              = get_option( 'time_format' );
    $datetime_separator       = tribe_get_option( 'dateTimeSeparator', ' @ ' );
    $time_range_separator     = tribe_get_option( 'timeRangeSeparator', ' - ' );

    $settings = array(
        'show_end_time' => true,
        'time'          => true,
    );

    $settings = wp_parse_args( apply_filters( 'tribe_events_event_schedule_details_formatting', $settings ), $settings );
    if ( ! $settings['time'] ) {
        $settings['show_end_time'] = false;
    }

    /**
     * @var $show_end_time
     * @var $time
     */
    extract( $settings );

    $format = $date_with_year_format;

    // if it starts and ends in the current year then there is no need to display the year
    if ( tribe_get_start_date( $event, false, 'Y' ) === date( 'Y' ) && tribe_get_end_date( $event, false, 'Y' ) === date( 'Y' ) ) {
        $format = $date_without_year_format;
    }

    if ( tribe_event_is_multiday( $event ) ) { // multi-date event

        $format2ndday = apply_filters( 'tribe_format_second_date_in_range', $format, $event );

        if ( tribe_event_is_all_day( $event ) ) {
            $inner .= tribe_get_start_date( $event, true, $format );
            $inner .= '</span>';
        } else {
            $inner .= tribe_get_start_date( $event, false, $format ) . ( $time ? $datetime_separator . tribe_get_start_date( $event, false, $time_format ) : '' );
            $inner .= '</span>';
            
        }
    } elseif ( tribe_event_is_all_day( $event ) ) { // all day event
        $inner .= tribe_get_start_date( $event, true, $format );
    } else { // single day event
        if ( tribe_get_start_date( $event, false, 'g:i A' ) === tribe_get_end_date( $event, false, 'g:i A' ) ) { // Same start/end time
            $inner .= tribe_get_start_date( $event, false, $format ) . ( $time ? $datetime_separator . tribe_get_start_date( $event, false, $time_format ) : '' );
        } else { // defined start/end time
            $inner .= tribe_get_start_date( $event, false, $format ) . ( $time ? $datetime_separator . tribe_get_start_date( $event, false, $time_format ) : '' );
            $inner .= '</span>';
        }
    }

    $inner .= '</span>';

    return $inner;
}

function univero_tribe_events_event_schedule_finish( $event = null, $before = '', $after = '' ) {
    if ( is_null( $event ) ) {
        global $post;
        $event = $post;
    }

    if ( is_numeric( $event ) ) {
        $event = get_post( $event );
    }

    $inner                    = '';
    $format                   = '';
    $date_without_year_format = tribe_get_date_format();
    $date_with_year_format    = tribe_get_date_format( true );
    $time_format              = get_option( 'time_format' );
    $datetime_separator       = tribe_get_option( 'dateTimeSeparator', ' @ ' );
    $time_range_separator     = tribe_get_option( 'timeRangeSeparator', ' - ' );

    $settings = array(
        'show_end_time' => true,
        'time'          => true,
    );

    $settings = wp_parse_args( apply_filters( 'tribe_events_event_schedule_details_formatting', $settings ), $settings );
    if ( ! $settings['time'] ) {
        $settings['show_end_time'] = false;
    }

    /**
     * @var $show_end_time
     * @var $time
     */
    extract( $settings );

    $format = $date_with_year_format;

    // if it starts and ends in the current year then there is no need to display the year
    if ( tribe_get_start_date( $event, false, 'Y' ) === date( 'Y' ) && tribe_get_end_date( $event, false, 'Y' ) === date( 'Y' ) ) {
        $format = $date_without_year_format;
    }

    if ( tribe_event_is_multiday( $event ) ) { // multi-date event

        $format2ndday = apply_filters( 'tribe_format_second_date_in_range', $format, $event );

        if ( tribe_event_is_all_day( $event ) ) {
            $inner .= '<span class="tribe-event-date-end">';

            $end_date_full = tribe_get_end_date( $event, true, Tribe__Date_Utils::DBDATETIMEFORMAT );
            $end_date_full_timestamp = strtotime( $end_date_full );

            // if the end date is <= the beginning of the day, consider it the previous day
            if ( $end_date_full_timestamp <= strtotime( tribe_beginning_of_day( $end_date_full ) ) ) {
                $end_date = tribe_format_date( $end_date_full_timestamp - DAY_IN_SECONDS, false, $format2ndday );
            } else {
                $end_date = tribe_get_end_date( $event, false, $format2ndday );
            }

            $inner .= $end_date;
        } else {
            $inner .= '<span class="tribe-event-date-end">';
            $inner .= tribe_get_end_date( $event, false, $format2ndday ) . ( $time ? $datetime_separator . tribe_get_end_date( $event, false, $time_format ) : '' );
        }
    } elseif ( tribe_event_is_all_day( $event ) ) { // all day event
        $inner .= tribe_get_start_date( $event, true, $format );
    } else { // single day event
        if ( tribe_get_start_date( $event, false, 'g:i A' ) === tribe_get_end_date( $event, false, 'g:i A' ) ) { // Same start/end time
            $inner .= tribe_get_start_date( $event, false, $format ) . ( $time ? $datetime_separator . tribe_get_start_date( $event, false, $time_format ) : '' );
        } else { // defined start/end time
            $inner .= '<span class="tribe-event-time">';
            $inner .= ( $show_end_time ? tribe_get_end_date( $event, false, $time_format ) : '' );
            $inner .= '</span>';
        }
    }

    $inner .= '';

    return $inner;
}