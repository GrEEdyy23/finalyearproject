<?php
global $post;
$current_lesson_id = get_the_ID();
?>
<?php if ( ! empty( $syllabus ) ) : ?>
	<div class="edr-syllabus syllabus-lesson">
		<?php foreach ( $syllabus as $group ) : ?>
			<?php if ( ! empty( $group['lessons'] ) ) : ?>
					<h3 class="group-title"><?php echo esc_html( $group['title'] ); ?></h3>
					<div class="group-body">
						<ul class="edr-lessons">
							<?php
								foreach ( $group['lessons'] as $lesson_id ) {
									if ( isset( $lessons[ $lesson_id ] ) ) {
										$post = $lessons[ $lesson_id ];
										setup_postdata( $post );
										$obj = Edr_Access::get_instance();
										$access = $obj->can_study_lesson($lesson_id);
										?>
											<li class="lesson <?php echo esc_attr($current_lesson_id == $lesson_id ? 'active' : ''); ?> <?php echo esc_attr($access ? 'can-access' : ''); ?>">
												<div class="lesson-header">
													<?php if ($access) { ?>
														<a class="lesson-title" href="<?php the_permalink(); ?>">
													<?php } ?>
														<?php the_title(); ?>
														
													<?php if ($access) { ?>
														</a>
													<?php } ?>
												</div>
											</li>
										<?php
									}
								}
								wp_reset_postdata();
							?>
						</ul>
					</div>
			<?php endif; ?>
		<?php endforeach; ?>
	</div>
<?php endif; ?>