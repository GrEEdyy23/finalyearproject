<?php
/**
 *
 * Search form.
 * @since 1.0.0
 * @version 1.0.0
 *
 */
?>
<div class="search-form">
	<form action="<?php echo esc_url( home_url( '/' ) ); ?>" method="get">
		<div class="input-group">
			<input type="text" placeholder="<?php esc_html_e( 'Search', 'univero' ); ?>" name="s" class="input-lg form-control"/>
			<span class="input-group-btn"> <button type="submit" class="btn btn-theme btn-search btn-lg"><i class="fa fa-search"></i> </span>
		</div>
	</form>
</div>