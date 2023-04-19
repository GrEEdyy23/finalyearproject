<?php
$post_format = get_post_format();
global $post;
?>
<article id="post-<?php the_ID(); ?>" <?php post_class(); ?>>
    
    <?php if ( $post_format == 'gallery' ) {
        $gallery = univero_post_gallery( get_the_content(), array( 'size' => 'full' ) );
    ?>
        <div class="entry-thumb <?php echo  (empty($gallery) ? 'no-thumb' : ''); ?>">
            <?php
            if ( !empty($gallery) ) { ?>
                <div class="entry-image-wrapper">
                    <?php echo wp_kses_post($gallery); ?>                    
                    <div class="entry-date bg-theme">
                        <span class="day"><?php the_time( 'd' ); ?></span>
                        <span class="month"><?php the_time( 'M' ); ?></span>
                    </div>
                    <?php if ( is_sticky(get_the_ID()) ) { ?>
                        <span class="sticky-post"><i class="univero-pin"></i></span>
                    <?php } ?>
                </div>
                <?php    
            }
            ?>
        </div>
    <?php } elseif( $post_format == 'link' ) {
            $univero_format = univero_post_format_link_helper( get_the_content(), get_the_title() );
            $univero_title = $univero_format['title'];
            $univero_link = univero_get_link_attributes( $univero_title );
            $thumb = univero_post_thumbnail('', $univero_link);
            if ( !empty($thumb) ) { ?>
                <div class="entry-image-wrapper">
                    <?php echo wp_kses_post($thumb); ?>                    
                    <div class="entry-date bg-theme">
                        <span class="day"><?php the_time( 'd' ); ?></span>
                        <span class="month"><?php the_time( 'M' ); ?></span>
                    </div>
                    <?php if ( is_sticky(get_the_ID()) ) { ?>
                        <span class="sticky-post"><i class="univero-pin"></i></span>
                    <?php } ?>
                </div>
                <?php    
            }
            
        } else { ?>
    	<div class="entry-thumb <?php echo  (!has_post_thumbnail() ? 'no-thumb' : ''); ?>">
    		<?php
                $thumb = univero_post_thumbnail();
                if ( !empty($thumb) ) { ?>
                    <div class="entry-image-wrapper">
                        <?php echo wp_kses_post($thumb); ?>                        
                        <div class="entry-date bg-theme">
                            <span class="day"><?php the_time( 'd' ); ?></span>
                            <span class="month"><?php the_time( 'M' ); ?></span>
                        </div>
                        <?php if ( is_sticky(get_the_ID()) ) { ?>
                            <span class="sticky-post"><i class="univero-pin"></i></span>
                        <?php } ?>
                    </div>
                    <?php    
                }
            ?>
    	</div>
    <?php } ?>
    <div class="entry-info">
        <?php if (get_the_title()) { ?>
            <h4 class="entry-title">
                <?php the_title(); ?>
            </h4>
        <?php } ?>
        <div class="entry-meta">
            <span> <i class="univero-bookmark" aria-hidden="true"></i><?php echo esc_html__('In: ','univero'); univero_post_categories($post); ?></span> 
            <span class="author"><i class="univero-user3" aria-hidden="true"></i><?php echo esc_html__('By: ','univero'); the_author_posts_link(); ?></span>
            
            <span class="comments"><i class="univero-speech-bubble" aria-hidden="true"></i> <?php comments_number( '0 Comment', '1 Comment', '% Comments' ); ?></span>
            <?php $viewed = (int)get_post_meta(get_the_ID(), '_views_count', true); ?>
            <span class="views"><i class="univero-eye" aria-hidden="true"></i> <?php echo sprintf(_n('%d View', '%d Views', $viewed, 'univero'), $viewed); ?></span>  
        </div>
    </div>
	<div class="detail-content">

    	<div class="single-info info-bottom">
    		<?php
                if ( $post_format == 'gallery' ) {
                    $gallery_filter = univero_gallery_from_content( get_the_content() );
                    echo wp_kses_post($gallery_filter['filtered_content']);
                } else {
            ?>
                    <div class="entry-description"><?php the_content(); ?></div><!-- /entry-content -->
            <?php } ?>
    		<?php
    		wp_link_pages( array(
    			'before'      => '<div class="page-links"><span class="page-links-title">' . esc_html__( 'Pages:', 'univero' ) . '</span>',
    			'after'       => '</div>',
    			'link_before' => '<span>',
    			'link_after'  => '</span>',
    			'pagelink'    => '<span class="screen-reader-text">' . esc_html__( 'Page', 'univero' ) . ' </span>%',
    			'separator'   => '',
    		) );
    		?>

            <div class="social-share">
                    
               <?php if( univero_get_config('show_blog_social_share', false) ) {
                    get_template_part( 'template-parts/sharebox' );
                } ?>         
            </div>

    		<div class="entry-tag">
                <?php univero_post_tags(); ?>
    		</div>

            <?php get_template_part( 'template-parts/author-bio'); ?>
    	</div>
    </div>
</article>