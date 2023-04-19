<?php

get_header();

$sidebar_configs = univero_get_course_layout_configs();
?>

<section id="main-container" class="single-lesson">
		
		<div id="main-content">
			<div id="primary" class="content-area">
				<div id="content" class="site-content detail-post container" role="main">
					<?php
						// Start the Loop.
						while ( have_posts() ) : the_post();

							get_template_part( 'educator/content-lesson' );

							
						// End the loop.
						endwhile;
					?>
				</div><!-- #content -->
			</div><!-- #primary -->
		</div>
</section>
<?php get_footer(); ?>
