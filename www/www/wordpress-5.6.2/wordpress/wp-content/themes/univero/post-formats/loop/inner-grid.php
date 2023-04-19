<?php
    $thumbsize = isset($thumbsize) ? $thumbsize : univero_get_blog_thumbsize();
    $nb_word = isset($nb_word) ? $nb_word : 15;
?>

<article <?php post_class('post post-grid'); ?>>
    <?php
    $thumb = univero_display_post_thumb($thumbsize);
    if ( !empty($thumb) ) {
    ?>
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
    <?php } ?>
    <div class="clearfix entry-content <?php echo !empty($thumb) ? '' : 'no-thumb'; ?>">
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
</article>