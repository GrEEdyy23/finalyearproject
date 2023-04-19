<?php

function univero_user_profile_fields( $user ) {
	$user_info = get_the_author_meta( 'ninzio_edr_info', $user->ID );
	?>
	<h3><?php esc_html_e( 'Lecturer Profile', 'univero' ); ?></h3>

	<table class="form-table">
		<tbody>
		<tr>
			<th>
				<label for="lecturer_mobile"><?php esc_html_e( 'Job', 'univero' ); ?></label>
			</th>
			<td>
				<input id="lecturer_mobile" class="regular-text" type="text" value="<?php echo isset( $user_info['job'] ) ? $user_info['job'] : ''; ?>" name="ninzio_edr_info[job]">
			</td>
		</tr>
		<tr>
			<th>
				<label for="lecturer_mobile"><?php esc_html_e( 'Mobile', 'univero' ); ?></label>
			</th>
			<td>
				<input id="lecturer_mobile" class="regular-text" type="text" value="<?php echo isset( $user_info['mobile'] ) ? $user_info['mobile'] : ''; ?>" name="ninzio_edr_info[mobile]">
			</td>
		</tr>
		<tr>
			<th>
				<label for="lecturer_facebook"><?php esc_html_e( 'Facebook Account', 'univero' ); ?></label>
			</th>
			<td>
				<input id="lecturer_facebook" class="regular-text" type="text" value="<?php echo isset( $user_info['facebook'] ) ? $user_info['facebook'] : ''; ?>" name="ninzio_edr_info[facebook]">
			</td>
		</tr>
		<tr>
			<th>
				<label for="lecturer_twitter"><?php esc_html_e( 'Twitter Account', 'univero' ); ?></label>
			</th>
			<td>
				<input id="lecturer_twitter" class="regular-text" type="text" value="<?php echo isset( $user_info['twitter'] ) ? $user_info['twitter'] : ''; ?>" name="ninzio_edr_info[twitter]">
			</td>
		</tr>
		<tr>
			<th>
				<label for="lecturer_google"><?php esc_html_e( 'Google Plus Account', 'univero' ); ?></label>
			</th>
			<td>
				<input id="lecturer_google" class="regular-text" type="text" value="<?php echo isset( $user_info['google'] ) ? $user_info['google'] : ''; ?>" name="ninzio_edr_info[google]">
			</td>
		</tr>
		<tr>
			<th>
				<label for="lecturer_linkedin"><?php esc_html_e( 'LinkedIn Plus Account', 'univero' ); ?></label>
			</th>
			<td>
				<input id="lecturer_linkedin" class="regular-text" type="text" value="<?php echo isset( $user_info['linkedin'] ) ? $user_info['linkedin'] : ''; ?>" name="ninzio_edr_info[linkedin]">
			</td>
		</tr>
		<tr>
			<th>
				<label for="lecturer_youtube"><?php esc_html_e( 'Youtube Account', 'univero' ); ?></label>
			</th>
			<td>
				<input id="lecturer_youtube" class="regular-text" type="text" value="<?php echo isset( $user_info['youtube'] ) ? $user_info['youtube'] : ''; ?>" name="ninzio_edr_info[youtube]">
			</td>
		</tr>
		<tr>
			<th>
				<label for="lecturer_pinterest"><?php esc_html_e( 'Pinterest Account', 'univero' ); ?></label>
			</th>
			<td>
				<input id="lecturer_pinterest" class="regular-text" type="text" value="<?php echo isset( $user_info['pinterest'] ) ? $user_info['pinterest'] : ''; ?>" name="ninzio_edr_info[pinterest]">
			</td>
		</tr>
		</tbody>
	</table>
	<?php
}
add_action( 'show_user_profile', 'univero_user_profile_fields' );
add_action( 'edit_user_profile', 'univero_user_profile_fields' );

function univero_save_user_profile_fields( $user_id ) {
	if ( !current_user_can( 'edit_user', $user_id ) ) {
		return false;
	}
	update_user_meta( $user_id, 'ninzio_edr_info', $_POST['ninzio_edr_info'] );
}

add_action( 'personal_options_update', 'univero_save_user_profile_fields' );
add_action( 'edit_user_profile_update', 'univero_save_user_profile_fields' );

function univero_educator_get_lecturers($number = -1) {
	$roles = array( 'administrator', 'lecturer' );
	$users_by_role = get_users( array( 'role__in' => $roles, 'number' => $number ) );
	return $users_by_role;
}

function univero_get_lecturers_by_ids( $args = array() ) {
	$wp_user_query = new WP_User_Query( array( 'include' => $args ) );
	return $wp_user_query->get_results();
}
