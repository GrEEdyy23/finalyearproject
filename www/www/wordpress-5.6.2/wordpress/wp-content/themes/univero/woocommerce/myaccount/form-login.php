<?php
/**
 * Login Form
 *
 * This template can be overridden by copying it to yourtheme/woocommerce/myaccount/form-login.php.
 *
 * HOWEVER, on occasion WooCommerce will need to update template files and you
 * (the theme developer) will need to copy the new files to your theme to
 * maintain compatibility. We try to do this as little as possible, but it does
 * happen. When this occurs the version of the template file will be bumped and
 * the readme will list any important changes.
 *
 * @see     https://docs.woocommerce.com/document/template-structure/
 * @package WooCommerce/Templates
 * @version 3.6.0
 */

if ( ! defined( 'ABSPATH' ) ) {
	exit; // Exit if accessed directly.
}

do_action( 'woocommerce_before_customer_login_form' ); ?>

<?php if ( get_option( 'woocommerce_enable_myaccount_registration' ) === 'yes' ) : ?>

<div class="row" id="customer_login">

	<div class="col-md-6 col-sm-12 col-xs-12">

<?php endif; ?>
		<div class="wrapper-account">
			<h2 class="title-login"><?php esc_html_e( 'Login Form', 'univero' ); ?></h2>

			<form method="post" class="login" role="form">

				<?php do_action( 'woocommerce_login_form_start' ); ?>

				<p class="form-group form-row form-row-wide">
					<input placeholder="<?php esc_html_e( 'Username or email address *', 'univero' ); ?>" type="text" class="input-text form-control" name="username" id="username" value="<?php if ( ! empty( $_POST['username'] ) ) echo esc_attr( $_POST['username'] ); ?>" />
				</p>
				<p class="form-group form-row form-row-wide">
					<input placeholder="<?php esc_html_e( 'Password *', 'univero' ); ?>" class="input-text form-control" type="password" name="password" id="password" />
				</p>

				<?php do_action( 'woocommerce_login_form' ); ?>

				<p class="form-group form-row">
					<?php wp_nonce_field( 'woocommerce-login', 'woocommerce-login-nonce' ); ?>
					<span class="pull-left input-submit">
						<input type="submit" class="button btn btn-theme radius-0" name="login" value="<?php esc_html_e( 'Login', 'univero' ); ?>" />
					</span>
					<span for="rememberme" class="pull-left">
							<input name="rememberme" type="checkbox" id="rememberme" value="forever" /> <?php esc_html_e( 'Remember me', 'univero' ); ?>
					</span>
					<span class="form-group lost_password pull-right">
						<a href="<?php echo esc_url( wp_lostpassword_url() ); ?>"><?php esc_html_e( 'Lost your password?', 'univero' ); ?></a>
					</span>
				</p>
				<?php do_action( 'woocommerce_login_form_end' ); ?>
			</form>
		</div>

<?php if ( get_option( 'woocommerce_enable_myaccount_registration' ) === 'yes' ) : ?>

	</div>

	<div class="col-md-6 col-sm-12 col-xs-12">
		<div class="wrapper-account">
		<h2 class="title-login"><?php esc_html_e( 'Register Form', 'univero' ); ?></h2>

		<form method="post" class="register widget" role="form">

			<?php do_action( 'woocommerce_register_form_start' ); ?>

			<?php if ( 'no' === get_option( 'woocommerce_registration_generate_username' ) ) : ?>

				<p class="form-group form-row form-row-wide">
					<label for="reg_username"><?php esc_html_e( 'Username', 'univero' ); ?> <span class="required">*</span></label>
					<input type="text" class="input-text form-control" name="username" id="reg_username" value="<?php if ( ! empty( $_POST['username'] ) ) echo esc_attr( $_POST['username'] ); ?>" />
				</p>

			<?php endif; ?>

			<p class="form-group form-row form-row-wide">
				<input type="email" placeholder="<?php esc_html_e( 'Email address *', 'univero' ); ?>" class="input-text form-control" name="email" id="reg_email" value="<?php if ( ! empty( $_POST['email'] ) ) echo esc_attr( $_POST['email'] ); ?>" />
			</p>

			<?php if ( 'no' === get_option( 'woocommerce_registration_generate_password' ) ) : ?>

				<p class="form-group form-row form-row-wide">
					<input type="password" placeholder="<?php esc_html_e( 'Password *', 'univero' ); ?>" class="input-text form-control" name="password" id="reg_password" />
				</p>

			<?php endif; ?>

			<!-- Spam Trap -->
			<div style="<?php echo ( ( is_rtl() ) ? 'right' : 'left' ); ?>: -999em; position: absolute;"><label for="trap"><?php esc_html_e( 'Anti-spam', 'univero' ); ?></label><input type="text" name="email_2" id="trap" tabindex="-1" /></div>

			<?php do_action( 'woocommerce_register_form' ); ?>
			

			<p class="form-group form-row">
				<?php wp_nonce_field( 'woocommerce-register', 'woocommerce-register-nonce' ); ?>
				<button type="submit" class="button btn btn-theme radius-0" name="register" value="<?php esc_attr_e( 'Register', 'univero' ); ?>"><?php esc_html_e( 'Register', 'univero' ); ?></button>
			</p>
			
			<?php do_action( 'woocommerce_register_form_end' ); ?>

		</form>
		</div>
	</div>

</div>
<?php endif; ?>

<?php do_action( 'woocommerce_after_customer_login_form' ); ?>