<?php

get_header();

$sidebar_configs = univero_get_event_layout_configs();

univero_render_breadcrumbs();

?>
<section id="main-container" class="main-content  <?php echo apply_filters('univero_event_content_class', 'container');?> inner">
	<div class="row">
		<?php if ( isset($sidebar_configs['left']) ) : ?>
			<div class="<?php echo esc_attr($sidebar_configs['left']['class']) ;?>">
			  	<aside class="sidebar sidebar-left" itemscope="itemscope" itemtype="//schema.org/WPSideBar">
			   		<?php dynamic_sidebar( $sidebar_configs['left']['sidebar'] ); ?>
			  	</aside>
			</div>
		<?php endif; ?>

		<div id="main-content" class="col-sm-12 <?php echo esc_attr($sidebar_configs['main']['class']); ?>">
			<main id="main" class="site-main layout-blog" role="main">
				<?php
				if ( !is_singular('tribe_events') && univero_get_config('show_top_event_categories', true) ) {
					global $wp_query;

					$term =	$wp_query->queried_object;
					$term_id_default = isset($term->term_id) ? $term->term_id : 0;
					$terms = get_terms(array(
					    'taxonomy' => 'tribe_events_cat',
					    'hide_empty' => false,
					));
					if (!empty($terms)) {
					?>
						<ul class="list-inline list-unstyled categories list-tab-v1 text-center">
							<?php foreach ($terms as $term) { ?>
								<li>
									<a class="<?php echo esc_attr($term->term_id == $term_id_default ? 'active' : ''); ?>" href="<?php echo esc_url(get_term_link($term->term_id, 'tribe_events_cat')); ?>"><?php echo wp_kses_post($term->name);?></a>
								</li>
							<?php } ?>
						</ul>
					<?php
					}
				}
				?>

				<?php tribe_events_before_html(); ?>
				<?php tribe_get_view(); ?>
				<?php tribe_events_after_html(); ?>
			</main><!-- .site-main -->
		</div><!-- .content-area -->
		<?php if ( isset($sidebar_configs['right']) ) : ?>
			<div class="<?php echo esc_attr($sidebar_configs['right']['class']) ;?>">
			  	<aside class="sidebar sidebar-right" itemscope="itemscope" itemtype="//schema.org/WPSideBar">
			   		<?php dynamic_sidebar( $sidebar_configs['right']['sidebar'] ); ?>
			  	</aside>
			</div>
		<?php endif; ?>
	</div>
</section>

<?php get_footer();