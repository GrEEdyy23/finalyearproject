<?php
/**
 * The template for displaying the footer
 *
 * Contains the closing of the "site-content" div and all content after.
 *
 * @package WordPress
 * @subpackage Univero
 * @since Univero 1.0
 */

$footer = apply_filters( 'univero_get_footer_layout', 'default' );

?>

	</div><!-- .site-content -->

	<footer id="ninzio-footer" class="ninzio-footer" role="contentinfo">
		<?php if ( !empty($footer) ): ?>
			<div class="container">
				<?php univero_display_footer_builder($footer); ?>
			</div>		
		<?php else: ?>			
			<div class="ninzio-copyright">
				<div class="container">
					<div class="copyright-content">
						<div class="text-copyright pull-left">
							<?php
								$allowed_html_array = array('strong' => array(),'a' => array('href' => array()));
								echo wp_kses(__('Copyright &copy; 2017 - Univero. All Rights Reserved. <br/> Powered by <a href="//ninzio.com">Ninzio</a>', 'univero'), $allowed_html_array);
							?>
						</div>
					</div>
				</div>
			</div>				
		<?php endif; ?>
		
	</footer><!-- .site-footer -->
	<?php
	if ( univero_get_config('back_to_top') ) { ?>
		<a href="#" id="back-to-top">
			<i class="univero-arrow-top"></i>
		</a>
	<?php
	}
	?>

</div><!-- .site -->

<?php wp_footer(); ?>
</body>
</html>