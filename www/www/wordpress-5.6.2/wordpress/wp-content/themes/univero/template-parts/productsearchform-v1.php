<div class="dropdown search-form">
	<a href="#" class="dropdown-toggle icon-theme icon-theme--light icon-theme--small" data-toggle="dropdown">
		<i class="univero-magnifier1"></i>		
	</a>
	<div class="dropdown-menu">
		<form action="<?php echo esc_url( home_url( '/' ) ); ?>" method="get">
		  	<input type="text" placeholder="<?php esc_html_e( 'Search', 'univero' ); ?>" name="s" class="ninzio-search form-control"/>
			<button type="submit" class="button-search btn"><i class="univero-magnifier1"></i></button>
			
		</form>		
	</div>
</div>