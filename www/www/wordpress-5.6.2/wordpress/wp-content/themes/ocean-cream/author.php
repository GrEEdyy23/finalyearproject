<!DOCTYPE html>
<html <?php language_attributes(); ?>>
<head>
	<meta charset="<?php bloginfo('charset'); ?>">

    <!-- Mobile zoom -->
    <meta name=viewport content="width=device-width, initial-scale=1">
    <link rel="profile" href="http://microformats.org/profile/hcard">

    <?php wp_head(); ?>
    <?php //PHP variables
    $ocean_cream_website = get_the_author_meta('url', $post->post_author);
    $ocean_cream_description = get_the_author_meta('description', $post->post_author);
    $ocean_cream_given_name = get_the_author_meta('first_name', $post->post_author);
    $ocean_cream_nickname = get_the_author_meta('nickname', $post->post_author);
    $ocean_cream_family_name = get_the_author_meta('last_name', $post->post_author);
    ?>
</head>
<body <?php body_class();?>>
    <!-- Create menu
    Show on mobile only when click on button -->
        <div id="ocean_cream_mobile_menu">
    <?php wp_nav_menu(
        array(
            'theme_location' => 'header-menu',
            'container_class' => 'menu',
            'container' => 'nav'  )
        );?>
        </div>
        <span id="ocean_cream_mobile_menu_icon">
<svg class="i-menu" viewBox="0 0 32 32" width="32" height="32" fill="none" stroke="currentcolor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2">
            <path d="M4 8 L28 8 M4 16 L28 16 M4 24 L28 24"></path>
        </svg>
    </span>

<main>
<!-- Widgets, default hide -->
    <div id="ocean_cream_sidebar">
    <aside class="ocean_cream_sidebar">
    <?php dynamic_sidebar('sidebar');?>
    </aside>
    </div>
    <span id="ocean_cream_sidebar_open" class="ocean_cream_button">
		<?php esc_html_e('Show Sidebar', 'ocean-cream'); ?>
	</span>
	<!-- Your post. Based on Codex example
https://developers.google.com/custom-search/docs/structured_data#viewing-extracted-structured-data
http://microformats.org/wiki/microformats2#authoring-->
<article <?php post_class('vcard');?> role="main">
    <h2>
    <?php esc_html_e('About:', 'ocean-cream');
        echo '<span class="fn n">';
    if (!empty($ocean_cream_given_name)) {
        echo '<span class="given-name">';
        the_author_meta('first_name');
        echo "</span> ";
    }
    if (!empty($ocean_cream_nickname)) {
        echo '<span class="nickname">';
        echo '"';
        the_author_meta('nickname');
        echo '"';
        echo "</span> ";
    }
    if (!empty($ocean_cream_family_name)) {
        echo  '<span class="family-name">';
        the_author_meta('last_name');
        echo "</span>";
        echo "</span>";
    }
    ?>
    </h2>
    <img style="float:left; margin-right:4px" src="<?php echo esc_url(get_avatar_url($user->ID)); ?>" alt="my avatar" class="h-card photo"/>
    <dl class="ocean_cream_author_profile">
        <!-- https://codex.wordpress.org/Template_Tags/the_author_meta -->
        <dt> <?php esc_html_e('From', 'ocean-cream'); ?>        </dt>
        <dd> <?php the_author_meta('user_registered'); ?>       </dd>
    <?php if (!empty($ocean_cream_website)) {
        echo "<dt>";
        esc_html_e('Website', 'ocean-cream');
        echo "</dt>";
        echo '<dd> <a class="url" href=';
        the_author_meta('user_url');
        echo " rel=me>";
        the_author_meta('user_url');
        echo "</a>              </dd>";
    };
    if (!empty($ocean_cream_description)) {
        echo "<dt>";
        esc_html_e('Biography', 'ocean-cream');
        echo "</dt>";
        echo '<dd class="note">';
        the_author_meta('description');
        echo "</dd>";
    } ?>
    </dl>
</br>
    <h2><?php esc_html_e('My posts:', 'ocean-cream'); ?></h2>

    <ul>
<!-- The Loop -->

    <?php if (have_posts()) : while (have_posts()) : the_post(); ?>
        <li style="list-style:none">
            <a href="<?php esc_url(the_permalink()) ?>" rel="bookmark">
            	<?php
			if (empty($post->post_title)){
			esc_html_e('Untitled', 'ocean-cream');
			 }
		else {
        the_title();}?></a>,
            <?php echo get_the_date(); ?>
			<?php esc_html_e('in', 'ocean-cream'); ?>
			<?php the_category('&');?>
        </li>

    <?php endwhile; else: ?>
        <p><?php esc_html_e('No posts by this author.', 'ocean-cream'); ?></p>

    <?php endif; ?>

<!-- End Loop -->

    </ul>
<?php the_posts_pagination(array( 'mid_size' => 2 )); ?>
</article>
</main>
    <?php wp_footer(); ?>
</body>
</html>