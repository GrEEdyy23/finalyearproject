<?php 

$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );

$args = array(
    'type' => 'post',
    'child_of' => 0,
    'orderby' => 'name',
    'order' => 'ASC',
    'hide_empty' => true,
    'hierarchical' => 1,
    'taxonomy' => 'edr_course_category'
);
$categories = array( '' => esc_html__('All Categories', 'univero') );
$terms = get_categories( $args );
if ( ! empty( $terms ) && ! is_wp_error( $terms ) ){
	foreach ( $terms as $term ) {
 		$categories[$term->term_id] =  $term->name;  
	}
}
$instructors = univero_educator_get_lecturers();
?>
<div class="widget-search-form bg-theme <?php echo esc_attr($el_class); ?>">
	<form action="<?php echo esc_url( home_url( '/' ) ); ?>" method="get">
		
		<div class="clearfix">
			<div class="left-search">				
				<?php if ( $show_title ) { ?>
					<div class="search-form-item">
						<!-- keyword -->
						<input class="input_search form-control" name="s" value="" placeholder="<?php esc_html_e('Enter Keyword', 'univero'); ?>"/>
					</div>
				<?php } ?>
				<?php if ( $show_category ) { ?>
					<div class="search-form-item">
						<!-- categories -->
						<div class="inner-select">
							<select class="form-control" name="_category">
				               	<?php foreach ($categories as $key => $value) { ?>
			                        <option value="<?php echo esc_attr($key); ?>"><?php echo wp_kses_post($value); ?></option>
			                    <?php } ?>
			                </select>
		                </div>
					</div>
				<?php } ?>
				<?php if ( $show_instructor ) { ?>
					<div class="search-form-item">
						<!-- lecturer -->
						<div class="inner-select">
							<select class="form-control" name="_lecturer">
								<option value=""><?php esc_html_e( 'All Instructors', 'univero' ); ?></option>
				               	<?php foreach ($instructors as $key => $value) { ?>
			                        <option value="<?php echo esc_attr($value->ID); ?>"><?php echo get_the_author_meta('display_name', $value->ID ); ?></option>
			                    <?php } ?>
			                </select>
						</div>
					</div>
				<?php } ?>					
				<div class="search-form-item text-center submit">
					<button type="submit" class="btn btn-theme">					
						<i class="univero-magnifier3"></i>
					</button>
				</div>
			</div>
		</div>
		<input type="hidden" name="post_type" value="<?php echo defined('EDR_PT_COURSE') ? EDR_PT_COURSE : ''; ?>" class="post_type" />
	</form>
</div>