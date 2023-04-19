<?php
/**
 * Renders each membership in the [memberships_page] shortcode.
 *
 * @version 1.1.0
 */

$obj_memberships = Edr_Memberships::get_instance();
$membership_id = get_the_ID();
$classes = apply_filters( 'edr_membership_classes', array( 'edr-membership' ) );
?>
<article id="membership-<?php the_ID(); ?>" class="<?php echo esc_attr( implode( ' ', $classes ) ); ?>">
	<div class="edr-membership-wrapper">
		<div class="edr-membership__header">
			<h2 class="edr-membership__title bg-theme">
				<span><?php the_title(); ?></span>					
			</h2>
			<div class="edr-membership__price"><?php echo edr_get_the_membership_price( $membership_id ); ?></div>
		</div>
		<?php if (has_post_thumbnail()) { ?>
			<div class="entry-thumb ">
		        <?php
		            the_post_thumbnail();
		        ?>
		    </div>
	    <?php } ?>

	    <div class="edr-membership__content clearfix">    	
            <div class="entry-description"><?php the_content(); ?></div>
			<div class="edr-membership__footer">
				<?php echo edr_get_membership_buy_link( $membership_id ); ?>
			</div>
		</div>
	</div>
</article>