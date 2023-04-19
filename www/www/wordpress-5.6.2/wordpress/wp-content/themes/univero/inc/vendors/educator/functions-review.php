<?php

// add comment for room type

function univero_add_comments_support_for_course() {
    add_post_type_support( EDR_PT_COURSE, 'comments' );
}
add_action( 'init', 'univero_add_comments_support_for_course', 1 );

// comment template
function univero_room_comments_template_loader($template) {
    if ( get_post_type() !== EDR_PT_COURSE ) {
        return $template;
    }
    return get_template_directory() . '/educator/single/reviews.php';
}
add_filter( 'comments_template', 'univero_room_comments_template_loader', 1000 );

// comment list
function univero_room_comments( $comment, $args, $depth ) {
    $GLOBALS['comment'] = $comment;
    set_query_var( 'comment', $comment );
    set_query_var( 'args', $args );
    set_query_var( 'depth', $depth );
    get_template_part( 'educator/single/review' );
}
// add comment meta
function univero_add_custom_comment_field( $comment_id, $comment_approved, $commentdata ) {
    $post_id = $commentdata['comment_post_ID'];
    $post = get_post($post_id);
    if ( $post->post_type == EDR_PT_COURSE ) {
        add_comment_meta( $comment_id, '_ninzio_rating', $_POST['rating'] );
    }
}
add_action( 'comment_post', 'univero_add_custom_comment_field', 10, 3 );

function univero_get_total_reviews( $post_id ) {
    $comments = get_comments( array('post_id' => $post_id, 'status' => 'approve') );
    if (empty($comments)) {
        return 0;
    }
    return count($comments);
}

function univero_get_total_rating( $post_id ) {
    $comments = get_comments( array('post_id' => $post_id, 'status' => 'approve') );
    if (empty($comments)) {
        return 0;
    }
    $total_review = 0;
    foreach ($comments as $comment) {
        $rating = intval( get_comment_meta( $comment->comment_ID, '_ninzio_rating', true ) );
        if ($rating) {
            $total_review += (int)$rating;
        }
    }
    return $total_review/count($comments);
}

function univero_print_review( $rate ) {
    ?>
        <div class="review-stars-rated list-rating">
            <div class="rating-print-wrapper">
                <ul class="review-stars">
                    <li><span class="fa fa-star-o"></span></li>
                    <li><span class="fa fa-star-o"></span></li>
                    <li><span class="fa fa-star-o"></span></li>
                    <li><span class="fa fa-star-o"></span></li>
                    <li><span class="fa fa-star-o"></span></li>
                </ul>
                
                <ul class="review-stars filled"  style="<?php echo esc_attr( 'width: ' . ( $rate * 20 ) . '%' ) ?>" >
                    <li><span class="fa fa-star"></span></li>
                    <li><span class="fa fa-star"></span></li>
                    <li><span class="fa fa-star"></span></li>
                    <li><span class="fa fa-star"></span></li>
                    <li><span class="fa fa-star"></span></li>
                </ul>
            </div>
        </div>
    <?php
}

function univero_print_single_review( $rate ) {
    ?>
        <div class="review-stars-rated list-rating">
            <div class="rating-print-wrapper">
                <ul class="review-stars">
                    <li><span class="fa fa-star-o"></span></li>
                    <li><span class="fa fa-star-o"></span></li>
                    <li><span class="fa fa-star-o"></span></li>
                    <li><span class="fa fa-star-o"></span></li>
                    <li><span class="fa fa-star-o"></span></li>
                </ul>
                
                <ul class="review-stars filled"  style="<?php echo esc_attr( 'width: ' . ( $rate * 20 ) . '%' ) ?>" >
                    <li><span class="fa fa-star"></span></li>
                    <li><span class="fa fa-star"></span></li>
                    <li><span class="fa fa-star"></span></li>
                    <li><span class="fa fa-star"></span></li>
                    <li><span class="fa fa-star"></span></li>
                </ul>
            </div>
        </div>
    <?php
}

function univero_get_detail_ratings( $post_id ) {
    global $wpdb;
    $comment_ratings = $wpdb->get_results( $wpdb->prepare(
        "
            SELECT cm2.meta_value AS rating, COUNT(*) AS quantity FROM $wpdb->posts AS p
            INNER JOIN $wpdb->comments AS c ON p.ID = c.comment_post_ID
            INNER JOIN $wpdb->commentmeta AS cm2 ON cm2.comment_id = c.comment_ID AND cm2.meta_key=%s
            WHERE p.ID=%d AND c.comment_approved=%d
            GROUP BY cm2.meta_value",
            '_ninzio_rating',
            $post_id,
            1
        ), OBJECT_K
    );
    return $comment_ratings;
}
