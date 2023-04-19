<?php
/**
 * The template for displaying search results pages.
 *
 * @package WordPress
 * @subpackage Univero
 * @since Univero 1.0
 */

get_header();

if ( isset($_GET['post_type']) && defined('EDR_PT_COURSE') && $_GET['post_type'] == EDR_PT_COURSE ) {
	get_template_part( 'search-course' );
} else {
	$sidebar_configs = univero_get_blog_layout_configs();

	$bscol = '12';
	$_count  = 0;

	univero_render_breadcrumbs();
	?>
	<section id="main-container" class="main-content  <?php echo apply_filters('univero_blog_content_class', 'container');?> inner">
		<div class="row">
			<?php if ( isset($sidebar_configs['left']) ) : ?>
				<div class="<?php echo esc_attr($sidebar_configs['left']['class']) ;?>">
				  	<aside class="sidebar sidebar-left" itemscope="itemscope" itemtype="//schema.org/WPSideBar">
				   		<?php if ( is_active_sidebar( $sidebar_configs['left']['sidebar'] ) ): ?>
					   		<?php dynamic_sidebar( $sidebar_configs['left']['sidebar'] ); ?>
					   	<?php endif; ?>
				  	</aside>
				</div>
			<?php endif; ?>

			<div id="main-content" class="col-sm-12 <?php echo esc_attr($sidebar_configs['main']['class']); ?>">
				<main id="main" class="site-main layout-blog" role="main">

				<?php if ( have_posts() ) : ?>

					<header class="page-header hidden">
						<?php
							the_archive_title( '<h1 class="page-title">', '</h1>' );
							the_archive_description( '<div class="taxonomy-description">', '</div>' );
						?>
					</header><!-- .page-header -->
					<div class="row">
					<?php
					// Start the Loop.
					while ( have_posts() ) : the_post();

						/*
						 * Include the Post-Format-specific template for the content.
						 * If you want to override this in a child theme, then include a file
						 * called content-___.php (where ___ is the Post Format name) and that will be used instead.
						 */
						?>
						
							<div class="col-sm-<?php echo esc_attr($bscol); ?>">
								<?php get_template_part( 'content', 'search' ); ?>
							</div>
						
						<?php
						$_count++;
					// End the loop.
					endwhile;
					?>
					</div>
					<?php
					// Previous/next page navigation.
					univero_paging_nav();

				// If no content, include the "No posts found" template.
				else :
					get_template_part( 'post-formats/content', 'none' );

				endif;
				?>

				</main><!-- .site-main -->
			</div><!-- .content-area -->
			<?php if ( isset($sidebar_configs['right']) ) : ?>
				<div class="<?php echo esc_attr($sidebar_configs['right']['class']) ;?>">
				  	<aside class="sidebar sidebar-right" itemscope="itemscope" itemtype="//schema.org/WPSideBar">
				   		<?php if ( is_active_sidebar( $sidebar_configs['right']['sidebar'] ) ): ?>
					   		<?php dynamic_sidebar( $sidebar_configs['right']['sidebar'] ); ?>
					   	<?php endif; ?>
				  	</aside>
				</div>
			<?php endif; ?>
			
		</div>
	</section>
<?php
}
get_footer();
