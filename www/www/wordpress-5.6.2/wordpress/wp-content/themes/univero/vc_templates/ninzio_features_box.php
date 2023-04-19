<?php
$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );
$items = (array) vc_param_group_parse_atts( $items );
if ( !empty($items) ):
$count = 0;
?>
	<div class="widget-features-box <?php echo esc_attr($el_class); ?> <?php echo esc_attr($layout_type); ?>  <?php echo esc_attr($style); ?>">
		
		<div class="widget-content">
			<?php foreach ($items as $item): ?>

				<?php $odd = ($count%2 == 1)?'odd':''; ?>
				<?php if ($count%$number == 0 ) {
					echo '<div class="row-item clearfix">';
				} ?>
				<?php if($number > 1) echo '<div class="'.$odd.' item col-xs-12 col-sm-12 col-lg-'.(12/$number).' col-md-'.(12/$number).'">'; ?>
					<?php if ( $layout_type == 'grid' ) { ?>
						<div class="feature-box clearfix <?php echo (!empty($item['featured'])) ? 'featured':''; ?>">
							<div class="feature-box-inner">
								<?php if ( !empty($item['icon_font']) ) { ?>
									<div class="fbox-icon">
										<div class="fbox-icon-inner">
											<i class="<?php echo esc_attr($item['icon_font']); ?>"></i>
										</div>
									</div>
								<?php } ?>
							    <div class="fbox-content ">  
							    	<?php if (isset($item['title']) && $item['title']!='') { ?>
							            <h3 class="ourservice-heading"><?php echo wp_kses_post($item['title']); ?></h3>
							        <?php } ?>
							         <?php if (isset($item['description']) && $item['description']!='') { ?>
							            <div class="description"><?php echo wp_kses_post( $item['description'] );?></div>  
							        <?php } ?>
							    </div> 
						    </div>
						</div>
					<?php } else { ?>
						<div class="feature-box clearfix">
							<div class="feature-box-inner media">
								<?php if ( !empty($item['icon_font']) ) { ?>
									<div class="fbox-icon media-left media-middle">
										<div class="fbox-icon-inner">
											<i class="<?php echo esc_attr($item['icon_font']); ?>"></i>
										</div>
									</div>
								<?php } ?>
							    <div class="fbox-content media-body media-middle">  
							    	<?php if (isset($item['title']) && $item['title']!='') { ?>
							            <h3 class="ourservice-heading"><?php echo wp_kses_post($item['title']); ?></h3>
							        <?php } ?>
							         <?php if (isset($item['description']) && $item['description']!='') { ?>
							            <div class="description"><?php echo wp_kses_post( $item['description'] );?></div>  
							        <?php } ?>
							    </div> 
						    </div>
						</div>
					<?php } ?>
				<?php if($number > 1) echo '</div>'; ?>
				<?php if ( ($count%$number == 1 && $count >= ($number - 1) ) || ($count == (count($items)-1) ) ) {
					echo '</div>';
				} ?>
				<?php $count++; ?>
			<?php endforeach; ?>
		</div>
	</div>
<?php endif; ?>