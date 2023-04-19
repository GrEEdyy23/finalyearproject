<?php
$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );

wp_enqueue_script( 'univero-counter', get_template_directory_uri().'/js/jquery.counterup.min.js', array( 'jquery' ) );
wp_enqueue_script( 'univero-waypoints', get_template_directory_uri().'/js/waypoints.min.js', array( 'jquery' ) );

$counters = (array) vc_param_group_parse_atts( $counters );
if ( !empty($counters) ):
	$bcol = 12/$columns;
?>
	<div class="widget-counters <?php echo esc_attr($el_class); ?>">
		<div class="row">
			<?php foreach ($counters as $item) { ?>
				<div class="col-sm-<?php echo esc_attr($bcol); ?> col-xs-<?php echo esc_attr($bcol); ?>">
					<div class="widget-counters-item <?php echo esc_attr(!empty($item['style']) ? $item['style'] : '');  echo  (!empty($item['featured']) && $item['featured'] == 1) ? ' featured' : ''; ?>">
						<div class="widget-counters-inner">
							<?php if ( !empty($item['icon_font']) ) { ?>
								<div class="font-icon">
									<i class="<?php echo esc_attr($item['icon_font']); ?>"></i>
								</div>
							<?php } ?>
							<div class="counter-wrap">
								<?php if ( !empty($item['number']) ) { ?>
							   		<span class="counter counterUp"><?php echo (int)$item['number']; ?></span>
							   	<?php } ?>
							   	<?php if ( !empty($item['suffix']) ) { ?>
							   		<span class="counter-suffix"><?php echo wp_kses_post($item['suffix']); ?></span>
							   	<?php } ?>
							</div>
							<?php if ( !empty($item['title']) ) { ?>
						   		<div class="title"><?php echo wp_kses_post($item['title']); ?></div>
						   	<?php } ?>
						</div>
					</div>
				</div>
			<?php } ?>
		</div>
	</div>
<?php endif; ?>