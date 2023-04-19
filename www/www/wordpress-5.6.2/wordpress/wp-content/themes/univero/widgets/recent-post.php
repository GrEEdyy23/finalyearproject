<?php
extract( $args );
extract( $instance );
$title = apply_filters('widget_title', $instance['title']);

if ( $title ) {
    echo wp_kses_post($before_title . $title . $after_title);
}

$args = array(
	'post_type' => 'post',
	'posts_per_page' => $number_post
);

$query = new WP_Query($args);
if($query->have_posts()):
?>
<div class="post-widget media-post-layout widget-content">
<ul class="posts-list">
<?php
	while($query->have_posts()):$query->the_post();
?>
	<li>
		<article class="post post-list">
		    <div class="entry-content media">
		        <?php
		        if ( has_post_thumbnail() ) {
		            ?>
		              <div class="media-left">
		                <figure class="entry-thumb effect-v6 dd">
		                    <a href="<?php the_permalink(); ?>" class="entry-image">
		                        <?php the_post_thumbnail( 'univero-course-thumb' ); ?>
		                    </a>  
		                </figure>
		              </div>
		            <?php
		        }
		        ?>
		        <div class="media-body">
		          <?php
		              if (get_the_title()) {
		              ?>
		                  <h4 class="entry-title">
		                      <a href="<?php the_permalink(); ?>"><?php the_title(); ?></a>
		                  </h4>
		              <?php
		          }
		          ?>
		          <span class="date"><?php the_time( get_option('date_format', 'd M, Y') ); ?></span>
		        </div>
		    </div>
		</article>
	</li>
<?php endwhile; ?>
<?php wp_reset_postdata(); ?>
</ul>
</div>
<?php endif; ?>
