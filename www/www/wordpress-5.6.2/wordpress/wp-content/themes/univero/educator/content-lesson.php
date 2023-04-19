<article <?php post_class('lesson-detail'); ?>>
    <?php $course_id = get_post_meta( get_the_ID(), '_edr_course_id', true ); ?>
    <div class="sticky-v-wrapper clearfix">
        <div class="course-lesson-sidebar-wrapper sticky-this">
            <?php if ($course_id) : ?>
                <div class="course-lesson-sidebar">
                    <a href="#course-lesson-sidebar" class="course-lesson-sidebar-btn show-lesson hidden-lg hidden-md" title="<?php esc_attr_e('Show Lessons', 'univero'); ?>"><i class="fa fa-hand-o-left" aria-hidden="true"></i></a>
                    <div class="widget">
                        <h3 class="forward"><a href="<?php echo esc_url(get_permalink($course_id)); ?>"><i class="univero-arrow-left" aria-hidden="true"></i> <?php esc_html_e('Back To The Course', 'univero'); ?></a></h3>
                        <div class="widget-content">
                            <?php univero_educator_display_lessons($course_id); ?>
                        </div>
                    </div>
                </div>
            <?php endif; ?>
        </div>
        <div class="course-lesson-content-wrapper">
            <div class="course-lesson-content">
                <div class="clearfix entry-content <?php echo !empty($thumb) ? '' : 'no-thumb'; ?>">
                    <div class="entry-info">
                        <?php if (get_the_title()) { ?>
                            <h4 class="entry-title">
                                <?php the_title(); ?>
                            </h4>
                        <?php } ?>
                    </div>
                    <div class="entry-thumb <?php echo (!has_post_thumbnail() ? 'no-thumb' : ''); ?>">
                        <?php
                            $thumb = univero_post_thumbnail();
                            echo wp_kses_post($thumb);
                        ?>
                    </div>
                    <div class="info-bottom">
                        <?php the_content(); ?>
                    </div>
                
                    <?php 
                    the_post_navigation( array(
                        'next_text' => '<span class="meta-nav" aria-hidden="true">' . esc_html__( 'Next', 'univero' ) . '</span> ' .
                            '<span class="pull-right navi">' . esc_html__( 'Next post:', 'univero' ) . '<i class="fa fa-long-arrow-right" aria-hidden="true"></i></span> ' .
                            '<span class="post-title">%title</span>',
                        'prev_text' => '<span class="meta-nav" aria-hidden="true">' . esc_html__( 'Previous', 'univero' ) . '</span> ' .
                            '<span class="pull-left navi"><i class="fa fa-long-arrow-left" aria-hidden="true"></i>' . esc_html__( 'Previous post:', 'univero' ) . '</span> ' .
                            '<span class="post-title">%title</span>',
                    ) );
                    ?>
                </div>
            </div>
        </div>
    </div>
</article>