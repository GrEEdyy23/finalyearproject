<?php
/**
 * The template for displaying 404 pages (not found)
 *
 * @package WordPress
 * @subpackage Univero
 * @since Univero 1.0
 */
/*

*Template Name: 404 Page
*/
get_header();
$sidebar_configs = univero_get_page_layout_configs();

?>
<section class="page-404">
	<section id="main-container" class="inner">
		
		<div id="main-content" class="main-page">

			<section class="error-404 not-found text-center clearfix">
				<div class="inner-content">
					<h1 class="page-title"><?php print(univero_get_config( '404_title', 'Page Not Found!' )); ?></h1>
					<p class="sub-title"><?php print(univero_get_config( '404_description', 'We are sorry, but we can not find the page you were looking for' )); ?></p>
					<div class="big-font">4<span class="text-theme">0</span>4!</div>
				</div>
			</section><!-- .error-404 -->

		</div><!-- .content-area -->
		
	</section>
</section>
<?php get_footer(); ?>