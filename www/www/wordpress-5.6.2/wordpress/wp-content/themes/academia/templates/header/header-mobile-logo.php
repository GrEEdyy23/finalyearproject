<?php
$g5plus_options = &G5Plus_Global::get_options();
$prefix = 'g5plus_';

$mobile_logo_height = rwmb_meta($prefix . 'mobile_logo_height');
if ($mobile_logo_height === '') {
	if (isset($g5plus_options['mobile_logo_height']) && isset($g5plus_options['mobile_logo_height']['height']) && ! empty($g5plus_options['mobile_logo_height']['height'])) {
		$mobile_logo_height = $g5plus_options['mobile_logo_height']['height'];
	}
}
$mobile_logo_height = str_replace('px' , '', $mobile_logo_height);
if ($mobile_logo_height != '') {
	$mobile_logo_height .= 'px';
}

$mobile_logo = g5plus_get_logo_url('mobile_logo');
$mobile_logo_retina = g5plus_get_logo_url('mobile_logo_retina');

$logo_attr = array();

$logo_attr[] = sprintf('alt="%s - %s"',get_bloginfo( 'name', 'display' ) ,get_bloginfo( 'description' , 'display' ));

if (!empty($mobile_logo_height)) {
    $logo_attr[] = sprintf('style="height:%s"',$mobile_logo_height);
}

if (!empty($mobile_logo_retina)) {
    $logo_attr[] = 'class="has-retina"';
}

$logo_retina_attr = array();

$logo_retina_attr[] = sprintf('alt="%s - %s"',get_bloginfo( 'name', 'display' ) ,get_bloginfo( 'description', 'display'));

if (!empty($mobile_logo_height)) {
    $logo_retina_attr = sprintf('style="height:%s"',$mobile_logo_height);
}



?>
<div class="header-logo-mobile">
	<a href="<?php echo esc_url( home_url( '/' ) ); ?>" title="<?php echo esc_attr( get_bloginfo( 'name', 'display' ) ); ?> - <?php bloginfo( 'description' ); ?>">
		<img <?php echo join(' ', $logo_attr)?>  src="<?php echo esc_url($mobile_logo); ?>" />
		<?php if (!empty($mobile_logo_retina)): ?>
			<img class="retina-logo" <?php echo join(' ', $logo_retina_attr)?>  src="<?php echo esc_url($mobile_logo_retina); ?>" />
		<?php endif;?>
	</a>
</div>