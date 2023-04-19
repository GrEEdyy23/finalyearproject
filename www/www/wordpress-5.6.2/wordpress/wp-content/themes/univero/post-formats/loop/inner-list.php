<?php
    $thumbsize = isset($thumbsize) ? $thumbsize : univero_get_blog_thumbsize();
    $nb_word = isset($nb_word) ? $nb_word : 35;
?>
<article <?php post_class('post post-list'); ?>>
    <div class="row">
        <?php
        $thumb = univero_display_post_thumb($thumbsize);
        if (!empty($thumb)) {
            ?>
            <div class="col-md-5 col-sm-5 col-xs-12">
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
            </div>
            <?php
        }
        ?>
        <div class="infor-content col-md-<?php echo !empty($thumb) ? '7' : '12'; ?> col-sm-<?php echo !empty($thumb) ? '7' : '12'; ?> col-xs-12">
            <?php if (get_the_title()) { ?>
                <h4 class="entry-title">
                    <a href="<?php the_permalink(); ?>"><?php the_title(); ?></a>
                </h4>
            <?php } ?>
            <?php if ( has_excerpt() ) { ?>
                <div class="entry-description"><?php echo univero_substring( get_the_excerpt(), $nb_word, '...' ); ?></div>
            <?php } ?>

            <a href="<?php the_permalink(); ?>" class="readmore"><?php esc_html_e('Read More', 'univero'); ?></a>
        </div>
    </div>
</article>