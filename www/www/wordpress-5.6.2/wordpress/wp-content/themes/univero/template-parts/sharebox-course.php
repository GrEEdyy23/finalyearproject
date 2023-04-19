<?php

global $post;
$args = array( 'position' => 'top', 'animation' => 'true' );
?>
<div class="ninzio-course-social-share">
	<ul class="list-social bo-social-icons bo-sicolor list-unstyled">
		<?php if ( univero_get_config('facebook_share', 1) ): ?>
 
			<li>
				<a class="bo-social-facebook facebook" data-toggle="tooltip" data-placement="<?php echo esc_attr($args['position']); ?>" data-animation="<?php echo esc_attr($args['animation']); ?>"  data-original-title="Facebook" href="//www.facebook.com/sharer.php?s=100&p&#91;url&#93;=<?php the_permalink(); ?>&p&#91;title&#93;=<?php the_title(); ?>" target="_blank" title="<?php echo esc_attr__('Share on facebook', 'univero'); ?>">
					<i class="fa fa-facebook"></i>
				</a>
			</li>
 
		<?php endif; ?>
		<?php if ( univero_get_config('twitter_share', 1) ): ?>
 
			<li>
				<a class="bo-social-facebook twitter"  data-toggle="tooltip" data-placement="<?php echo esc_attr($args['position']); ?>" data-animation="<?php echo esc_attr($args['animation']); ?>"  data-original-title="Twitter" href="//twitter.com/home?status=<?php the_title(); ?> <?php the_permalink(); ?>" target="_blank" title="<?php echo esc_attr__('Share on Twitter', 'univero'); ?>">
				<i class="fa fa-twitter"></i>
			</a>
			</li>
 
		<?php endif; ?>
		<?php if ( univero_get_config('linkedin_share', 1) ): ?>
 
			<li>
				<a class="bo-social-facebook linkedin"  data-toggle="tooltip" data-placement="<?php echo esc_attr($args['position']); ?>" data-animation="<?php echo esc_attr($args['animation']); ?>"  data-original-title="LinkedIn" href="//linkedin.com/shareArticle?mini=true&amp;url=<?php the_permalink(); ?>&amp;title=<?php the_title(); ?>" target="_blank" title="<?php echo esc_attr__('Share on LinkedIn', 'univero'); ?>">
				<i class="fa fa-linkedin"></i>
			</a>
			</li>
 
		<?php endif; ?>
		
		<?php if ( univero_get_config('google_share', 1) ): ?>
 
			<li>
				<a class="bo-social-facebook google" data-toggle="tooltip" data-placement="<?php echo esc_attr($args['position']); ?>" data-animation="<?php echo esc_attr($args['animation']); ?>"  data-original-title="Google plus" href="https://plus.google.com/share?url=<?php the_permalink(); ?>" onclick="javascript:window.open(this.href, '', 'menubar=no,toolbar=no,resizable=yes,scrollbars=yes,height=600,width=600');return false;" target="_blank" title="<?php echo esc_attr__('Share on Google plus', 'univero'); ?>"> <i class="fa fa-google-plus"></i> </a>
			</li>

		<?php endif; ?>
		<?php if ( univero_get_config('pinterest_share', 1) ): ?>
 
			<?php $full_image = wp_get_attachment_image_src(get_post_thumbnail_id(), 'full'); ?>
			<li>
				<a class="bo-social-pinterest pinterest" data-toggle="tooltip" data-placement="<?php echo esc_attr($args['position']); ?>" data-animation="<?php echo esc_attr($args['animation']); ?>"  data-original-title="Pinterest" href="//pinterest.com/pin/create/button/?url=<?php echo urlencode(get_permalink()); ?>&amp;description=<?php echo urlencode($post->post_title); ?>&amp;media=<?php echo urlencode($full_image[0]); ?>" target="_blank" title="<?php echo esc_attr__('Share on Pinterest', 'univero'); ?>">
				<i class="fa fa-pinterest"></i>
			</a>
			</li>
 
		<?php endif; ?>
	</ul>
</div>	