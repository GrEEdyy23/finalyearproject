<?php
global $post;
$course_id = $post->ID;
?>
<article id="post-<?php the_ID(); ?>" <?php post_class(); ?>>
    
    <div class="entry-head">
        <div class="info-left">
            <?php if (get_the_title()) { ?>
                <h4 class="entry-title">
                    <?php the_title(); ?>
                </h4>
            <?php } ?>
        </div>
    </div>
    <div class="info-meta edr-course clearfix">

        <?php
            $terms = get_the_terms( $course_id, 'edr_course_category' );
            if ( !empty($terms) ) {
                ?>
                <div class="category">
                    <div class="left-icon">
                        <i class="univero-bookmark"></i>
                    </div>
                    <div class="right-categor">
                        <div class="title">
                            <?php echo esc_html__('Category','univero'); ?>
                        </div>
                        <?php 
                        foreach ( $terms as $term ) {
                            echo '<a href="' . get_term_link( $term->term_id, 'edr_course_category' ) . '">' . $term->name . '</a>';
                            break;
                        }
                        ?>
                    </div>
                </div>
                <?php
            }
        ?>

        <div class="edr-teacher">
            <div class="left-icon">
                 <i class="univero-user3"></i>
            </div>
            <!-- .author-avatar -->
            <div class="description">
                <div class="title">
                    <?php echo esc_html__('Instructor','univero'); ?>
                </div>
                <div class="author-title">
                    <?php echo get_the_author(); ?>
                </div>
            </div>
        </div>
       
        <?php $buy_html = edr_get_buy_widget( array( 'object_id' => $course_id, 'object_type' => EDR_PT_COURSE, 'label' => esc_html__( 'Take this course', 'univero' ) ) ); ?>
        <?php if (!empty($buy_html)) { ?>
            <div class="course-price">
                <?php echo wp_kses_post($buy_html); ?>
            </div>
        <?php } ?>

         <div class="course-review">
            <?php
                $total_rating = univero_get_total_rating( $course_id );
                $count_rating = univero_get_total_reviews( $course_id );
            ?>
            <?php univero_print_review($total_rating); ?>
            <div class="rating-count"><?php echo sprintf(_n('%d Review', '%d Reviews', $count_rating, 'univero'), $count_rating); ?></div>
        </div>

    </div>

	<div class="entry-thumb">
        <?php get_template_part( 'educator/single/gallery' ); ?>
	</div>
    
	<div class="detail-content">
        <?php
            remove_action( 'edr_before_single_course_content', 'edr_display_course_info' );
            remove_action( 'edr_after_single_course_content', 'edr_display_lessons' );
        ?>
        <div class="row">
            <div class="col-sm-8">
                <div id="course-description" class="entry-description">
                    <h3 class="title-tab"><?php echo esc_html__('Course Description', 'univero') ?></h3>
                    <?php the_content(); ?>
                </div><!-- /entry-content -->
            </div>
            <div class="col-sm-4">
                <div id="course-info" class="entry-info">
                    <h3 class="title-tab"><?php echo esc_html__('Course Info', 'univero') ?></h3>
                    <?php get_template_part( 'educator/single/course-features' ); ?>
                </div>
            </div>
        </div>
        
        <?php if ( univero_get_config('show_course_social_share', true) ) { ?>
            <div class="course-socials">
                <?php get_template_part( 'template-parts/sharebox-course' ); ?>
            </div>
        <?php } ?>

        <div id="course-program">
            <?php edr_display_lessons($course_id); ?>
        </div><!-- /entry-lesson -->


        <?php get_template_part( 'educator/single/instructors' ); ?>
        
        <?php
            if ( comments_open() || get_comments_number() ) {
                comments_template();
            }
        ?>
    </div>
</article>