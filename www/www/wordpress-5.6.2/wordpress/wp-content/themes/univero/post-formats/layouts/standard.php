
<div class="style-standard">
    <?php while ( have_posts() ) : the_post(); ?>
        
        <?php get_template_part( 'post-formats/loop/inner-standard' ); ?>
        
    <?php endwhile; ?>
</div>
