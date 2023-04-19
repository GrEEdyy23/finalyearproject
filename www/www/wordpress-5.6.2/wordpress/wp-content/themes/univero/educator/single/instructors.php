<?php
global $post;
$author_id = $post->post_author;

$instructors = array($author_id);

$users = univero_get_lecturers_by_ids( $instructors );
if ( !empty($users) ):
?>
	<div class="course-instructors">
		<h3 class="title-tab"><?php esc_html_e('About Instructor', 'univero'); ?></h3>
	<ul class="list-instructors">
		<?php
			foreach ($users as $user) {
				$author_info = get_the_author_meta( 'ninzio_edr_info', $user->ID );
				?>
				<li>
					<div class="about-container media">
						<div class="avatar-img media-left">
							<?php echo get_avatar( get_the_author_meta( 'user_email' ), 200 ); ?>
						</div>
						<!-- .author-avatar -->
						<div class="description media-body">
							<h4 class="author-title">
								<?php echo get_the_author_meta('display_name', $user->ID); ?>
							</h4>
							<div class="description">
								<?php echo get_the_author_meta('description', $user->ID); ?>
							</div>
							<div class="socials">
								<?php if ( isset($author_info['facebook']) && $author_info['facebook'] ): ?>
									<a href="<?php echo esc_url($author_info['facebook']); ?>" class="facebook icon-theme icon-theme--small icon-theme--gray"><i class="univero-facebook"></i></a>
								<?php endif; ?>
								<?php if ( isset($author_info['twitter']) && $author_info['twitter']): ?>
									<a href="<?php echo esc_url($author_info['twitter']); ?>" class="twitter icon-theme icon-theme--small icon-theme--gray"><i class="univero-twitter"></i></a>
								<?php endif; ?>
								<?php if ( isset($author_info['google']) && $author_info['google'] ): ?>
									<a href="<?php echo esc_url($author_info['google']); ?>" class="google icon-theme icon-theme--small icon-theme--gray"><i class="univero-google-plus"></i></a>
								<?php endif; ?>
								<?php if ( isset($author_info['linkedin']) && $author_info['linkedin'] ): ?>
									<a href="<?php echo esc_url($author_info['linkedin']); ?>" class="linkedin icon-theme icon-theme--small icon-theme--gray"><i class="univero-linkedin"></i></a>
								<?php endif; ?>
								<?php if ( isset($author_info['instagram']) && $author_info['instagram']): ?>
									<a href="<?php echo esc_url($author_info['instagram']); ?>" class="instagram icon-theme icon-theme--small icon-theme--gray"><i class="univero-instagram"></i></a>
								<?php endif; ?>
								<?php if ( isset($author_info['youtube']) && $author_info['youtube']): ?>
									<a href="<?php echo esc_url($author_info['youtube']); ?>" class="youtube icon-theme icon-theme--small icon-theme--gray"><i class="univero-youtube"></i></a>
								<?php endif; ?>
								<?php if ( isset($author_info['pinterest']) && $author_info['pinterest']): ?>
									<a href="<?php echo esc_url($author_info['pinterest']); ?>" class="pinterest icon-theme icon-theme--small icon-theme--gray"><i class="univero-pinterest"></i></a>
								<?php endif; ?>
							</div>
						</div>
					</div>
				</li>
				<?php
			}
		?>
	</ul>
	</div>	
<?php endif; ?>