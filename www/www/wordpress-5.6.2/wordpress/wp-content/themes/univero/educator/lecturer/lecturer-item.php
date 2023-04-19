<?php

	$author_id = $lecturer->ID;
	$author_info = get_the_author_meta( 'ninzio_edr_info', $author_id );
?>
<div class="ninzio-teacher-inner style1 clearfix ">
	<div class="author-avatar">
		<a href="<?php echo esc_url( get_author_posts_url( $author_id ) ); ?>">
			<?php echo get_avatar( $author_id, 370); ?>
		</a>
	</div>
	<div class="infor">
		<h3 class="name">
			<a href="<?php echo esc_url( get_author_posts_url( $author_id ) ); ?>">
				<?php echo get_the_author_meta('display_name', $author_id ); ?>
			</a>
		</h3>
		<?php if ( isset($author_info['job']) ): ?>
			<div class="job"><?php echo wp_kses_post($author_info['job']); ?></div>
		<?php endif; ?>
		<div class="description">
			<?php echo wp_kses_post(univero_substring(get_the_author_meta('description', $author_id ), 12, '...')); ?>
		</div>
		<div class="socials">
			<?php if ( isset($author_info['facebook']) && $author_info['facebook'] ): ?>
				<a href="<?php echo esc_url($author_info['facebook']); ?>" class="facebook"><i class="fa fa-facebook"></i></a>
			<?php endif; ?>
			<?php if ( isset($author_info['twitter']) && $author_info['twitter']): ?>
				<a href="<?php echo esc_url($author_info['twitter']); ?>" class="twitter"><i class="fa fa-twitter"></i></a>
			<?php endif; ?>
			<?php if ( isset($author_info['google']) && $author_info['google'] ): ?>
				<a href="<?php echo esc_url($author_info['google']); ?>" class="google"><i class="fa fa-google-plus"></i></a>
			<?php endif; ?>
			<?php if ( isset($author_info['linkedin']) && $author_info['linkedin'] ): ?>
				<a href="<?php echo esc_url($author_info['linkedin']); ?>" class="linkedin"><i class="fa fa-linkedin"></i></a>
			<?php endif; ?>
			<?php if ( isset($author_info['instagram']) && $author_info['instagram']): ?>
				<a href="<?php echo esc_url($author_info['instagram']); ?>" class="instagram"><i class="fa fa-instagram"></i></a>
			<?php endif; ?>
			<?php if ( isset($author_info['youtube']) && $author_info['youtube']): ?>
				<a href="<?php echo esc_url($author_info['youtube']); ?>" class="youtube"><i class="fa fa-youtube"></i></a>
			<?php endif; ?>
		</div>		
	</div>	
</div>