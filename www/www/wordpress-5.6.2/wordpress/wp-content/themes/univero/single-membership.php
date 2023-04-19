<?php

get_header();

$sidebar_configs = univero_get_course_layout_configs();

univero_render_breadcrumbs();
?>


<section id="main-container" class="main-content <?php echo apply_filters( 'univero_course_content_class', 'container' ); ?> inner">
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
		<div id="main-content" class="col-xs-12 <?php echo esc_attr($sidebar_configs['main']['class']); ?>">
			<div id="primary" class="content-area">
				<div id="content" class="site-content detail-post" role="main">
					<?php
						// Start the Loop.
						while ( have_posts() ) : the_post();
							?>
							<article <?php post_class('lesson-detail'); ?>>
							    
							    <div class="clearfix entry-content <?php echo !empty($thumb) ? '' : 'no-thumb'; ?>">
							        <div class="entry-info">
							            <?php if (get_the_title()) { ?>
							                <h4 class="entry-title">
							                    <a href="<?php the_permalink(); ?>"><?php the_title(); ?></a>
							                </h4>
							            <?php } ?>
							            
							        </div>
							        <div class="entry-thumb <?php echo  (!has_post_thumbnail() ? 'no-thumb' : ''); ?>">
							            <?php
							                $thumb = univero_post_thumbnail();
							                echo wp_kses_post($thumb);
							            ?>
							        </div>
							        <div class="info-bottom">
							            <?php the_content(); ?>
							        </div>
							    </div>
							</article>
							<?php
						// End the loop.
						endwhile;
					?>
				</div><!-- #content -->
			</div><!-- #primary -->
		</div>	
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
<?php get_footer(); ?>
