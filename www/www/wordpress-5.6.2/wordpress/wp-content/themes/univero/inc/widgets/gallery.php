<?php

class Univero_Gallery_Widget extends Ninzio_Widget {
    public function __construct() {
        parent::__construct(
            'ninzio_gallery',
            esc_html__('Ninzio Gallery Widget', 'univero'),
            array( 'description' => esc_html__( 'Show list gallery', 'univero' ), )
        );
        $this->widgetName = 'gallery';
    }

    public function getTemplate() {
        $this->template = 'gallery.php';
    }

    public function widget( $args, $instance ) {
        $this->display($args, $instance);
    }
    
    public function form( $instance ) {
        $defaults = array(
            'title' => 'Photostream',
            'number_post' => '6',
        );
        $instance = wp_parse_args((array) $instance, $defaults);
        // Widget admin form
        $options = array(
            'most_recent' => esc_html__('Lastest Gallery', 'univero'),
            'random' => esc_html__('Random Gallery', 'univero'),
        );
        ?>
        <p>
            <label for="<?php echo esc_attr($this->get_field_id( 'title' )); ?>"><?php esc_html_e( 'Title:', 'univero' ); ?></label>
            <input class="widefat" id="<?php echo esc_attr($this->get_field_id( 'title' )); ?>" name="<?php echo esc_attr($this->get_field_name( 'title' )); ?>" type="text" value="<?php echo esc_attr( $instance['title'] ); ?>" />
        </p>
        <p>
            <label for="<?php echo esc_attr($this->get_field_id( 'number_post' )); ?>"><?php esc_html_e( 'Number Gallery:', 'univero' ); ?></label>
            <input class="widefat" id="<?php echo esc_attr($this->get_field_id( 'number_post' )); ?>" name="<?php echo esc_attr($this->get_field_name( 'number_post' )); ?>" type="text" value="<?php echo esc_attr($instance['number_post']); ?>" />
        </p>
<?php
    }

    public function update( $new_instance, $old_instance ) {
        $instance = array();
        $instance['title'] = ( ! empty( $new_instance['title'] ) ) ? strip_tags( $new_instance['title'] ) : '';
        $instance['number_post'] = ( ! empty( $new_instance['number_post'] ) ) ? strip_tags( $new_instance['number_post'] ) : '';
        return $instance;

    }
}

univero_reg_widget( 'Univero_Gallery_Widget' );