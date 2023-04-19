<?php
$g5plus_options = &G5Plus_Global::get_options();

$prefix = 'g5plus_';
$logo_url = g5plus_get_logo_url('logo');
$logo_retina = g5plus_get_logo_url('logo_retina');

$sticky_logo = '';
$sticky_logo_retina = '';
if (!in_array(G5Plus_Global::get_header_layout(), array('header-4', 'header-5', 'header-6', 'header-7', 'header-8'))) {
	$sticky_logo = g5plus_get_logo_url('sticky_logo');
	$sticky_logo_retina = g5plus_get_logo_url('sticky_logo_retina');
}

$header_logo_class = array('header-logo');
if (!empty($sticky_logo) && ($sticky_logo != $logo_url)) {
	$header_logo_class[] = 'has-logo-sticky';
}

// Logo Height
$logo_height = rwmb_meta($prefix . 'logo_height');
if ($logo_height == '') {
	if (isset($g5plus_options['logo_height']) && isset($g5plus_options['logo_height']['height']) && ! empty($g5plus_options['logo_height']['height'])) {
		$logo_height = $g5plus_options['logo_height']['height'];
	}
}
$logo_height = str_replace('px' , '', $logo_height);
if ($logo_height != '') {
	$logo_height .= 'px';
}

$logo_attr = array();
$logo_attr[] = sprintf('alt="%s - %s"', get_bloginfo( 'name', 'display' ), get_bloginfo( 'description', 'display' ));
if (!empty($logo_height)) {
    $logo_attr[] = sprintf('style="height:%s"', $logo_height);
}

if (!empty($logo_retina)) {
    $logo_attr[] = 'class="has-retina"';
}

$logo_retina_attr = array();

if (!empty($logo_height)) {
    $logo_retina_attr[] = sprintf('style="height:%s"', $logo_height);
}
$logo_retina_attr[] = sprintf('alt="%s - %s"', get_bloginfo( 'name', 'display' ), get_bloginfo( 'description', 'display' ));


?>
<div class="<?php echo join(' ', $header_logo_class) ?>">
	<a href="<?php echo esc_url( home_url( '/' ) ); ?>" title="<?php echo esc_attr( get_bloginfo( 'name', 'display' ) ); ?> - <?php bloginfo( 'description' ); ?>">
		<img  <?php echo join(' ', $logo_attr)?> src="<?php echo esc_url($logo_url); ?>" />
		<?php if (!empty($logo_retina)): ?>
			<img class="retina-logo" <?php echo join(' ', $logo_retina_attr)?>  src="<?php echo esc_url($logo_retina) ?>"/>
		<?php endif;?>
	</a>
</div>
<?php if (!empty($sticky_logo) && ($sticky_logo != $logo_url)): ?>
	<div class="logo-sticky header-logo">
		<a  href="<?php echo esc_url( home_url( '/' ) ); ?>" title="<?php echo esc_attr( get_bloginfo( 'name', 'display' ) ); ?> - <?php bloginfo( 'description' ); ?>">
			<img class="<?php echo esc_attr(!empty($sticky_logo_retina) ? 'has-retina' : '' )?>" src="<?php echo esc_url($sticky_logo); ?>" alt="<?php echo esc_attr( get_bloginfo( 'name', 'display' ) ); ?> - <?php bloginfo( 'description' ); ?>" />
			<?php if (!empty($sticky_logo_retina)): ?>
				<img class="retina-logo" src="<?php echo esc_url($sticky_logo_retina); ?>" alt="<?php echo esc_attr( get_bloginfo( 'name', 'display' ) ); ?> - <?php bloginfo( 'description' ); ?>" />
			<?php endif;?>
		</a>
	</div>
<?php endif;?>