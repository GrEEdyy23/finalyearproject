<?php

global $post;

if ( ! defined( 'ABSPATH' ) ) {
	exit; // Exit if accessed directly
}

if ( ! comments_open() ) {
	return;
}
$total_rating = univero_get_total_rating( get_the_ID() );
$comment_ratings = univero_get_detail_ratings( get_the_ID() );
$total = univero_get_total_reviews( get_the_ID() );
?>
<div id="course-review">
	<div id="reviews">
		<h3 class="title-tab"><?php esc_html_e( 'Reviews', 'univero' ); ?></h3>
		<div class="course-rating clearfix">
			<div class="average-rating">
				<div class="rating-overall">
					<div class="average-value"><?php echo wp_kses_post( $total_rating  ? esc_html( round( $total_rating, 1 ) ) : 0); ?></div>
					<?php univero_print_review($total_rating); ?>
					<div class="count-rating"><?php echo sprintf(_n('%d Rating', '%d Ratings', $total, 'univero'), $total) ?></div>
				</div>

				<div class="rating-detailed">
					<div class="detailed-rating-inner">
						<?php for ( $i = 5; $i >= 1; $i -- ) : ?>
							<div class="skill skil-course">
								<div class="key"><?php printf( esc_html__( 'stars %d', 'univero' ), $i ); ?></div>
								<div class="progress">
									<div class="value-percent"><?php echo wp_kses_post(( $total && !empty( $comment_ratings[$i]->quantity ) ) ? esc_attr(  ( $comment_ratings[$i]->quantity / $total * 100 ) . '%' ) : '0%'); ?></div>
									<div class="progress-bar progress-bar-success progress-bar-striped " style="<?php echo trim(( $total && !empty( $comment_ratings[$i]->quantity ) ) ? esc_attr( 'width: ' . ( $comment_ratings[$i]->quantity / $total * 100 ) . '%' ) : 'width: 0%'); ?>">
									</div>
								</div>
								<div class="value"><?php echo empty( $comment_ratings[$i]->quantity ) ? '0' : esc_html( $comment_ratings[$i]->quantity ); ?></div>
							</div>
						<?php endfor; ?>
					</div>
				</div>
			</div>
		</div>
		<div id="comments">
			<?php
			if ( have_comments() ) : ?>
				<h3 class="title-tab"><?php echo sprintf(_n('%d Review', '%d Reviews', $total, 'univero'), $total) ?></h3>

				<ol class="commentlist">
					<?php wp_list_comments( array( 'callback' => 'univero_room_comments' ) ); ?>
				</ol>
				
				<?php if ( get_comment_pages_count() > 1 && get_option( 'page_comments' ) ) :
					echo '<nav class="ninzio-pagination">';
					paginate_comments_links( apply_filters( 'ninzio_comment_pagination_args', array(
						'prev_text' => '&larr;',
						'next_text' => '&rarr;',
						'type'      => 'list',
					) ) );
					echo '</nav>';
				endif; ?>
				
			<?php else : ?>
				<p class="ninzio-noreviews"><?php esc_html_e( 'There are no reviews yet.', 'univero' ); ?></p>
			<?php endif; ?>
		</div>
			
		<div class="clear"></div>

		<h3 class="title-tab"><?php esc_html_e( 'Leave a Review', 'univero' ); ?></h3>
		<div id="review_form_wrapper">
			<div id="review_form">
				<?php
					$commenter = wp_get_current_commenter();
					$comment_form = array(
						'title_reply'          => have_comments() ? '': sprintf( esc_html__( 'Be the first to review &ldquo;%s&rdquo;', 'univero' ), get_the_title() ),
						'title_reply_to'       => esc_html__( 'Leave a Reply to %s', 'univero' ),
						'comment_notes_before' => '',
						'comment_notes_after'  => '',
						'fields'               => array(
							'author' => '<div class="comment-form-author">' . '<label for="author">' . esc_html__( 'Name', 'univero' ) . ' <span class="required">*</span></label> ' .
							            '<input id="author" name="author" type="text" value="' . esc_attr( $commenter['comment_author'] ) . '" size="30" aria-required="true" /></div>',
							'email'  => '<div class="comment-form-email"><label for="email">' . esc_html__( 'Email', 'univero' ) . ' <span class="required">*</span></label> ' .
							            '<input id="email" name="email" type="text" value="' . esc_attr(  $commenter['comment_author_email'] ) . '" size="30" aria-required="true" /></div>',
						),
						'label_submit'  => esc_html__( 'Submit', 'univero' ),
						'logged_in_as'  => '',
						'comment_field' => ''
					);

					
					$comment_form['comment_field'] = '<div class="forrating"><label for="rating">' . esc_html__( 'Your Rating', 'univero' ) .'</label><div><div class="comment-form-rating rating-print-wrapper">
						<ul class="review-stars">
							<li><span class="univero-star-empty"></span></li>
							<li><span class="univero-star-empty"></span></li>
							<li><span class="univero-star-empty"></span></li>
							<li><span class="univero-star-empty"></span></li>
							<li><span class="univero-star-empty"></span></li>
						</ul>
						<ul class="review-stars filled" style="width: 100%">
							<li><span class="univero-star"></span></li>
							<li><span class="univero-star"></span></li>
							<li><span class="univero-star"></span></li>
							<li><span class="univero-star"></span></li>
							<li><span class="univero-star"></span></li>
						</ul>
						<input type="hidden" value="5" name="rating" id="ninzio_input_rating"></div></div></div>';
					

					$comment_form['comment_field'] .= '<div class="comment-form-comment"><label for="comment">' . esc_html__( 'Your Review', 'univero' ) . '</label><textarea id="comment" name="comment" cols="45" rows="8" aria-required="true"></textarea></div>';
 					
 					univero_comment_form($comment_form);
				?>
			</div>
		</div>

		
	</div>
</div>