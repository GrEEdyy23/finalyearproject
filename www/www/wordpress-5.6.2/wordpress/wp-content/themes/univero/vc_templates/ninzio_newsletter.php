<?php
$atts = vc_map_get_attributes( $this->getShortcode(), $atts );
extract( $atts );
$img = wp_get_attachment_image_src($image_icon,'full');
?>
<div class="clearfix widget-newletter <?php echo esc_attr($el_class).' '.(isset($style) ? esc_attr($style) : ''); ?> <?php echo esc_attr($title != '' ? 'hastitle' : ''); ?>" >
    <div class="info-left">
	    <?php if ( !empty($img) && isset($img[0]) ): ?>
	    	<div class="icon">
	    		<img src="<?php echo esc_url_raw($img[0]); ?>" alt="<?php esc_attr_e('Image', 'univero'); ?>">
	    	</div>
	    <?php endif; ?>
	    <div class="info-right">
		    <?php if ($title!=''): ?>
		        <h3 class="widget-title">
		            <span><?php echo esc_attr( $title ); ?></span>
		        </h3>
		    <?php endif; ?>
			<?php if (!empty($description)) { ?>
				<p class="widget-description">
					<?php echo wp_kses_post( $description ); ?>
				</p>
			<?php } ?>	
		</div>
	</div>
    <div class="widget-content"> 
		<?php
			if ( function_exists( 'mc4wp_show_form' ) ) {
			  	try {
			  	    $form = mc4wp_get_form(); 
					mc4wp_show_form( $form->ID );
				} catch( Exception $e ) {
				 	esc_html_e( 'Please create a newsletter form from Mailchip plugins', 'univero' );	
				}
			}
		?>
	</div>
</div>