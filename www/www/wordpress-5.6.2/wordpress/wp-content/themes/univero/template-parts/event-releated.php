<?php
    $relate_count = univero_get_config('number_event_releated', 3);
    $relate_columns = univero_get_config('releated_event_columns', 3);
    $terms = get_the_terms( get_the_ID(), 'tribe_events_cat' );
    $termids =array();

    if ($terms) {
        foreach($terms as $term) {
            $termids[] = $term->term_id;
        }
    }

    $args = array(
        'post_type' => 'tribe_events',
        'posts_per_page' => $relate_count,
        'post__not_in' => array( get_the_ID() ),
        'tax_query' => array(
            'relation' => 'AND',
            array(
                'taxonomy' => 'tribe_events_cat',
                'field' => 'id',
                'terms' => $termids,
                'operator' => 'IN'
            )
        )
    );

    $relates = new WP_Query( $args );
    if( $relates->have_posts() ):

?>
    <div class="widget no-margin">
        <h4 class="title">
            <span><?php esc_html_e( 'Releated Events', 'univero' ); ?></span>
        </h4>

        <div class="related-events-content  widget-content">
            <div class="owl-carousel" data-smallmedium="2" data-extrasmall="1" data-items="<?php echo esc_attr($relate_columns); ?>" data-carousel="owl" data-pagination="true" data-nav="true">
                <?php while ( $relates->have_posts() ) : $relates->the_post(); ?>
                    <?php tribe_get_template_part( 'list/single', 'event' ) ?>
                <?php endwhile; ?>
                <?php wp_reset_postdata(); ?>
            </div>
        </div>
        
    </div>
<?php endif; ?>