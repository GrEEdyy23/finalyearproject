<div class="layout-blog style-list">
    <?php
        while ( have_posts() ) : the_post();
            get_template_part( 'post-formats/loop/inner-list' );
        endwhile;
    ?>
</div>