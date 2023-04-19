<?php 

$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );
if ( empty($post_name) ) {
	return;
}
$loop = new WP_Query( array(
	'post_type'      => EDR_PT_MEMBERSHIP,
	'post_status'    => 'publish',
	'order'          => 'ASC',
	'orderby'        => 'menu_order',
	'name' => $post_name
) );

if ( $loop->have_posts() ) :
	?>
	<div class="widget widget-membership <?php echo esc_attr($el_class); ?> <?php echo esc_attr($recommend ? 'active' : ''); ?>">
		<div class="edr-memberships">
			<?php
				while ( $loop->have_posts() ) {
					$loop->the_post();
					?>
						<?php Edr_View::template_part( 'content', 'membership' ); ?>
					<?php
				}
			?>
		</div>
	</div>
	<?php
	wp_reset_postdata();
endif;