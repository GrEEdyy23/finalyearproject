<?php
require_once( __DIR__ . '/controls-manager.php' );

	class PAFE_Tooltip extends \Elementor\Widget_Base {

		public function __construct() {
			parent::__construct();
			$this->init();
		}

		public function get_name() {
			return 'pafe-tooltip';
		}

		public function register_controls( $element, $args ) {

			$element->start_controls_section(
				'pafe_tooltip_section',
				[
					'label' => __( 'PAFE Tooltip', 'pafe' ),
					'tab' => PAFE_Controls_Manager_Free::TAB_PAFE,
				]
			);

			$element->add_control(
				'pafe_tooltip',
				[
					'label' => __( 'Enable Tooltip', 'pafe' ),
					'type' => \Elementor\Controls_Manager::SWITCHER,
					'default' => '',
					'description' => __( 'This feature only works on the frontend.', 'pafe' ),
					'label_on' => 'Yes',
					'label_off' => 'No',
					'return_value' => 'yes',
				] 
			);

 
			$element->add_control(
				'pafe_tooltip_content',
				[
					'label' => __( 'Content', 'pafe' ),
					'type' => \Elementor\Controls_Manager::TEXTAREA,
					'condition' => [ 
						'pafe_tooltip' => 'yes',
					]
				]
				
			);

			$element->add_control(
				'pafe_tooltip_duration',
				[
					'label' => __( 'Duration (ms)', 'pafe' ),
					'type' => \Elementor\Controls_Manager::NUMBER,
					'default' => '300',
					'condition' => [ 
						'pafe_tooltip' => 'yes',
					]
				] 
			);

			$element->add_control(
				'pafe_tooltip_distance',
				[
					'label' => __( 'Distance', 'pafe' ),
					'type' => \Elementor\Controls_Manager::NUMBER,
					'size_units' => [ 'px' ],
					'default' => '5',
					'condition' => [ 
						'pafe_tooltip' => 'yes',
					]
				] 
			);

			$element->add_control(
				'pafe_tooltip_animation_type',
				[
					'label' => __( 'Animation Type', 'pafe' ),
					'type' => \Elementor\Controls_Manager::SELECT,
					'default' => 'scale',
					'options' => [
						'scale'  => __( 'Scale','pafe' ),
						'fade' => __( 'Fade','pafe' ),
					],
					'condition' => [ 
						'pafe_tooltip' => 'yes',
					]
				] 
			);

			$element->add_control(
				'pafe_tooltip_placement',
				[
					'label' => __( 'Placement', 'pafe' ),
					'type' => \Elementor\Controls_Manager::SELECT,
					'default' => 'top',
					'options' => [
						'top'  => __( 'Top','pafe' ),
						'right' => __( 'Right','pafe' ),
						'bottom' => __( 'Bottom','pafe' ),
						'left' => __( 'Left','pafe' ),
					],
					'condition' => [ 
						'pafe_tooltip' => 'yes',
					]
				] 
			);
 
			$element->add_control(
				'pafe_tooltip_content_color',
				[
					'label' => __( 'Content Color', 'pafe' ),
					'type' => \Elementor\Controls_Manager::COLOR,
					'scheme' => [
						'type' => \Elementor\Scheme_Color::get_type(),
						'value' => \Elementor\Scheme_Color::COLOR_1,
					],
					'default' => '#fff', 
					'selectors' => [
						'{{WRAPPER}} .tippy-tooltip' => 'color: {{VALUE}}',
					],
					'condition' => [ 
						'pafe_tooltip' => 'yes',  
					]
				]
			);

			$element->add_control(
				'pafe_tooltip_background_color',
				[
					'label' => __( 'Background Color', 'pafe' ),
					'type' => \Elementor\Controls_Manager::COLOR,
					'scheme' => [
						'type' => \Elementor\Scheme_Color::get_type(),
						'value' => \Elementor\Scheme_Color::COLOR_1,
					],
					'default' => '#000', 
					'selectors' => [
						'{{WRAPPER}} .tippy-tooltip' => 'background-color: {{VALUE}}',
						'{{WRAPPER}} .tippy-popper[x-placement^=top] .tippy-arrow' => 'border-top-color: {{VALUE}}',
						'{{WRAPPER}} .tippy-popper[x-placement^=right] .tippy-arrow' => 'border-right-color: {{VALUE}}',
						'{{WRAPPER}} .tippy-popper[x-placement^=bottom] .tippy-arrow' => 'border-bottom-color: {{VALUE}}',
						'{{WRAPPER}} .tippy-popper[x-placement^=left] .tippy-arrow' => 'border-left-color: {{VALUE}}',
					],
					'condition' => [ 
						'pafe_tooltip' => 'yes',  
					]
				]
			);

			$element->add_group_control(
				\Elementor\Group_Control_Typography::get_type(),
				[
					'name' => 'pafe_tooltip_label_typography',
					'selector' => '{{WRAPPER}} .tippy-tooltip',
					'scheme' => \Elementor\Scheme_Typography::TYPOGRAPHY_4,
					'condition' => [ 
						'pafe_tooltip' => 'yes',  
					]
				]
			);



			$element->end_controls_section();
		} 

		public function after_render_element($element) {
			$settings = $element->get_settings(); 	
			if ( !empty( $settings['pafe_tooltip'] ) && !empty($settings['pafe_tooltip_content']) ) {
				$tippy_options = [
					'duration' => absint( $settings['pafe_tooltip_duration'] ),
					'animation' => $settings['pafe_tooltip_animation_type'],
					'distance' => floatval( $settings['pafe_tooltip_distance'] ),
					'placement' => $settings['pafe_tooltip_placement'],
				];

				$element->add_render_attribute( '_wrapper', [
					'data-tippy-content' => $settings['pafe_tooltip_content'],
					'data-pafe-tippy-options' => json_encode( $tippy_options ),
				]);
			}
		}

		protected function init() {
			add_action( 'elementor/element/section/section_advanced/after_section_end', [ $this, 'register_controls' ], 10, 2 );
			add_action( 'elementor/element/column/section_advanced/after_section_end', [ $this, 'register_controls' ], 10, 2 );
			add_action( 'elementor/element/common/_section_background/after_section_end', [ $this, 'register_controls' ], 10, 2 );
			add_action( 'elementor/frontend/section/before_render', [ $this, 'after_render_element'], 10, 1 );
			add_action( 'elementor/frontend/column/before_render', [ $this, 'after_render_element'], 10, 1 );
			add_action( 'elementor/frontend/widget/before_render', [ $this, 'after_render_element'], 10, 1 );
		}

	}	