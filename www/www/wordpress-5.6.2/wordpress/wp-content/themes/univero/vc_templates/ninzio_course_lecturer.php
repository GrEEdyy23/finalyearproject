<?php 

$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );

$bcol = 12/$columns;
if ($columns == 5) {
	$bcol = 'cus-5';
}
$users = univero_educator_get_lecturers($number);
if ( !empty($users) ) {
	?>
	<div class="widget widget-lecturer <?php echo esc_attr($el_class); ?>">
		<?php if ( $layout_type == 'grid' ) { ?>
			<div class="row">
				<?php
				foreach ($users as $user) {
					set_query_var( 'lecturer', $user );
					?>
					<div class="col-sm-<?php echo esc_attr($bcol); ?> col-xs-12">
						<?php get_template_part( 'educator/lecturer/lecturer-item' ); ?>
					</div>
					<?php
				}
				?>
			</div>
		<?php } else { ?>
			<div class="owl-carousel owl-carousel-top" data-items="<?php echo esc_attr($columns); ?>" data-carousel="owl" data-smallmedium="2" data-extrasmall="1" data-pagination="<?php echo esc_attr($show_pagination ? 'true' : 'false'); ?>" data-nav="<?php echo esc_attr($show_nav ? 'true' : 'false'); ?>">
			    <?php foreach ($users as $user) {
					set_query_var( 'lecturer', $user );
					?>
						<?php get_template_part( 'educator/lecturer/lecturer-item' ); ?>
				<?php } ?>
			</div> 
		<?php } ?>
	</div>
	<?php
}