# Save this code as pendulum_rc_animation_final.py
# Run from terminal: manim -pql pendulum_rc_animation_final.py PendulumReservoirComputing

from manim import *
import numpy as np

class PendulumReservoirComputing(Scene):
    def construct(self):
        np.random.seed(42) # For consistency

        # --- Configuration ---
        input_color = BLUE_D
        pendulum_color = RED_C # Color for pendulum bob
        state_color = PURPLE_B # Color for sampled states
        readout_color = GREEN_D
        signal_color = YELLOW_D
        connection_color = WHITE
        trained_connection_color = ORANGE
        drive_color = GOLD_D
        equation_color = WHITE

        # --- Texts and Titles ---
        main_title = Text("Reservoir Computing with a Single Pendulum").scale(0.7)
        paper_ref = Text("Illustrating concept from: PhysRevE.105.054203", font_size=18).next_to(main_title, DOWN, buff=0.15)
        title_group = VGroup(main_title, paper_ref).to_edge(UP, buff=0.5)
        self.play(Write(title_group))
        self.wait(1)

        # --- 1. Introduce the Pendulum Reservoir ---
        section1_title = Text("1. The Reservoir: A Driven Pendulum", font_size=28).next_to(title_group, DOWN, buff=0.4).to_edge(LEFT)
        self.play(FadeIn(section1_title, shift=RIGHT))

        # Pendulum Setup
        pivot_point = UP * 1.75 + LEFT * 2.5 # Position pendulum more to the left
        pendulum_length = 2.0
        bob_radius = 0.18
        rod_width = 3

        # Create pendulum components
        pivot_dot = Dot(pivot_point, radius=0.05, color=DARK_GRAY)
        bob = Dot(pivot_point + DOWN * pendulum_length, radius=bob_radius, color=pendulum_color, fill_opacity=1).set_z_index(1)
        rod = Line(pivot_point, bob.get_center(), stroke_width=rod_width, color=LIGHT_GRAY).set_z_index(0)
        pendulum = VGroup(rod, bob) # Group rod and bob

        # Driving force indicator
        drive_arrow = Arrow(pivot_point + DOWN*pendulum_length*0.4 + LEFT*0.8,
                            pivot_point + DOWN*pendulum_length*0.4 + RIGHT*0.8,
                            buff=0, color=drive_color, stroke_width=4, max_tip_length_to_length_ratio=0.15)
        drive_label = Text("Periodic Driving", font_size=16, color=drive_color).next_to(drive_arrow, DOWN, buff=0.1)
        drive_visual = VGroup(drive_arrow, drive_label) # Removed shift, position relative to pendulum

        # Equation (Simplified representation)
        # Using x for theta, \alpha u(t) as the input perturbation term
        equation = MathTex(
            r"\ddot{x}", r"=", r"-\sin(x)", r"-", r"k\dot{x}", r"+", r"f \cdot \text{drive}(t)", r"\;", r"\mathbf{+ \alpha u(t)}"
        ).scale(0.7).next_to(pendulum, RIGHT, buff=1.2).shift(UP*0.8) # Position equation to the right
        equation.set_color(equation_color)
        equation.set_color_by_tex("k\dot{x}", YELLOW_E) # Damping
        equation.set_color_by_tex(r"f \cdot \text{drive}(t)", drive_color) # Driving
        equation.set_color_by_tex(r"\mathbf{+ \alpha u(t)}", signal_color) # Input term (will appear later)
        input_term_index = 8 # Index of the input term MathTex part

        # Updater for the rod
        def update_rod(mob):
            mob.become(Line(pivot_point, bob.get_center(), stroke_width=rod_width, color=LIGHT_GRAY).set_z_index(0))
        rod.add_updater(update_rod)

        # Animate pendulum creation and initial swing
        self.play(FadeIn(pivot_dot), Create(rod), Create(bob), Write(equation[:input_term_index]), FadeIn(drive_visual), run_time=1.5)
        self.wait(0.5)
        self.play(Flash(drive_arrow, color=WHITE, time_width=0.5, run_time=1), Indicate(equation[6], color=drive_color)) # Indicate driving term

        # Simulate some complex motion (using Rotate) - visually represents inherent dynamics
        self.play(Rotate(pendulum, angle=PI/3.5, about_point=pivot_point, rate_func=rate_functions.smooth, run_time=0.8))
        self.play(Rotate(pendulum, angle=-PI/2.2, about_point=pivot_point, rate_func=rate_functions.smooth, run_time=1.2))
        self.play(Rotate(pendulum, angle=PI/4.5, about_point=pivot_point, rate_func=rate_functions.smooth, run_time=0.7))
        self.play(Rotate(pendulum, angle=-PI/5, about_point=pivot_point, rate_func=rate_functions.smooth, run_time=0.6))
        self.wait(0.5)

        fixed_dynamics_label = Text("Fixed, Complex Dynamics", font_size=20).next_to(pendulum, DOWN, buff=0.6)
        self.play(FadeIn(fixed_dynamics_label))
        self.wait(1)
        pendulum_system = VGroup(pivot_dot, rod, bob, drive_visual, equation, fixed_dynamics_label) # Group all related elements

        # --- 2. Input Signal Injection ---
        self.play(FadeOut(section1_title))
        section2_title = Text("2. Input Signal Perturbs Dynamics", font_size=28).next_to(title_group, DOWN, buff=0.4).to_edge(LEFT)
        self.play(FadeIn(section2_title, shift=RIGHT))

        # Input signal representation
        input_group = VGroup( # Group label and pulse
             Text("Input Signal u(t)", font_size=20, color=signal_color),
             Dot(color=signal_color, radius=0.1)
        ).arrange(DOWN, buff=0.2).shift(LEFT * 5.5 + DOWN * 1.5) # Position input far left
        input_pulse = input_group[1]

        # Arrow showing input affects the equation
        input_effect_arrow = CurvedArrow(input_pulse.get_right(), equation[input_term_index].get_left() + DOWN*0.1, angle=-PI/4, color=signal_color, stroke_width=3)

        self.play(FadeIn(input_group, shift=RIGHT))
        self.wait(0.5)
        # Reveal the input term in the equation and draw the arrow using Create
        self.play(FadeIn(equation[input_term_index], shift=UP*0.2), Create(input_effect_arrow)) # *** Use Create instead of GrowArrow ***
        self.play(Indicate(equation[input_term_index], color=WHITE, scale_factor=1.2)) # Highlight input term
        self.wait(0.5)

        # Show input pulse effect: Flash pulse and make pendulum swing differently
        self.play(Flash(input_pulse, color=WHITE, flash_radius=0.3, time_width=0.5, run_time=1))
        # Different swing pattern due to input
        self.play(Rotate(pendulum, angle=-PI/1.8, about_point=pivot_point, rate_func=rate_functions.smooth, run_time=1.0)) # More drastic swing
        self.play(Rotate(pendulum, angle=PI/3, about_point=pivot_point, rate_func=rate_functions.smooth, run_time=0.8))
        self.play(Rotate(pendulum, angle=-PI/4, about_point=pivot_point, rate_func=rate_functions.smooth, run_time=0.7))
        self.wait(1)
        input_elements = VGroup(input_group, input_effect_arrow) # Group for fadeout

        # --- 3. State Sampling (Temporal Richness) ---
        self.play(FadeOut(section2_title))
        section3_title = Text("3. Sampling State Over Time", font_size=28).next_to(title_group, DOWN, buff=0.4).to_edge(LEFT)
        self.play(FadeIn(section3_title, shift=RIGHT))

        # Create axes for plotting state (angle x vs time)
        sampling_pos = RIGHT * 3.5 + DOWN * 1.0 # Position axes to the right
        time_axis_len = 4.5
        state_axis_height = 2.0
        state_axes = Axes(
            x_range=[0, 5.5, 1], # Time steps (adjusted range)
            y_range=[-PI/2, PI/2, PI/4], # Angle range
            x_length=time_axis_len,
            y_length=state_axis_height,
            tips=False,
            axis_config={"include_numbers": True, "font_size": 18, "color": LIGHT_GRAY},
            x_axis_config={"numbers_to_include": np.arange(1, 6)},
            y_axis_config={"numbers_to_include": [-PI/4, 0, PI/4], "decimal_number_config": {"num_decimal_places": 1}},
        ).shift(sampling_pos)
        state_axes_labels = state_axes.get_axis_labels(x_label=Tex("t", font_size=24), y_label=Tex("x(t)", font_size=24))

        sampling_label = Text("Sampled States [x(t), $\\dot{x}$(t), ...]", font_size=20).next_to(state_axes, UP, buff=0.2) # Added dot x notation

        self.play(FadeOut(input_elements), # Keep pendulum and equation
                  Create(state_axes), Write(state_axes_labels),
                  FadeIn(sampling_label))
        self.wait(0.5)

        # Simulate sampling states as pendulum moves
        sampled_state_dots = VGroup()
        n_samples = 5
        time_points = np.linspace(state_axes.x_range[0] + 0.5, state_axes.x_range[1] - 0.5, n_samples)

        # Define a simple function for visual angle sequence (NOT physics)
        def get_visual_angle(t_index):
            # Oscillates semi-randomly for visual effect
            return (PI/3) * np.sin(t_index * 1.5 + np.random.uniform(-0.5, 0.5)) * (0.6 + 0.4 * np.cos(t_index * 0.8))

        for i, t in enumerate(time_points):
            target_angle = get_visual_angle(i)
            # Rotate pendulum smoothly to the target angle visual
            self.play(Rotate(pendulum, angle=target_angle, about_point=pivot_point, run_time=0.6, rate_func=rate_functions.ease_in_out_sine))

            # Calculate plot coordinates based on the target_angle
            y_val = target_angle # Direct mapping for simplicity here
            state_coord = state_axes.coords_to_point(t, y_val)
            state_dot = Dot(state_coord, color=state_color, radius=0.07)

            # Animate dot appearance
            self.play(FadeIn(state_dot, scale=0.5), run_time=0.3)
            sampled_state_dots.add(state_dot)
            self.wait(0.1)

        # Highlight individual dots briefly before boxing
        self.play(LaggedStart(*[Indicate(dot, color=WHITE, scale_factor=1.5) for dot in sampled_state_dots], lag_ratio=0.15), run_time=1.0)

        # Box the sampled states to represent the "reservoir state vector"
        state_box = SurroundingRectangle(sampled_state_dots, buff=0.15, color=state_color, stroke_width=2)
        state_vector_label = Text("Effective State (History)", font_size=18, color=state_color).next_to(state_box, DOWN, buff=0.15)

        self.play(Create(state_box), Write(state_vector_label))
        self.wait(1.5)
        state_sampling_elements = VGroup(state_axes, state_axes_labels, sampling_label, sampled_state_dots, state_box, state_vector_label)

        # --- 4. Readout Layer & Training ---
        self.play(FadeOut(section3_title))
        section4_title = Text("4. Readout Layer Interprets State", font_size=28).next_to(title_group, DOWN, buff=0.4).to_edge(LEFT)
        self.play(FadeIn(section4_title, shift=RIGHT))

        # Create Readout Layer
        readout_nodes = VGroup(*[Dot(color=readout_color, radius=0.15) for _ in range(2)]).arrange(DOWN, buff=0.7)
        readout_label = Text("Readout", font_size=24).next_to(readout_nodes, DOWN, buff=0.2)
        trainable_label = Text("(Trainable)", font_size=18).next_to(readout_label, DOWN, buff=0.1)
        # Position readout to the far right
        readout_layer = VGroup(readout_nodes, readout_label, trainable_label).shift(RIGHT * 6.0 + UP * 0.5)

        self.play(FadeIn(readout_layer, shift=LEFT))
        self.wait(0.5)

        # Connections: State Box -> Readout
        output_connections = VGroup()
        connection_start_point = state_box.get_center() # Start from center of box for visual grouping

        for r_node in readout_nodes:
            arrow = Arrow(connection_start_point, r_node.get_center(), buff=0.2, color=connection_color,
                          stroke_width=2.5, max_tip_length_to_length_ratio=0.1) # Slightly thicker arrow
            output_connections.add(arrow)

        # Animate connection growth using Create
        self.play(LaggedStart(*[Create(a) for a in output_connections], lag_ratio=0.2, run_time=1.5)) # *** Use Create ***
        self.wait(0.5)

        # Emphasize Training Readout ONLY
        training_label_txt = Text("Training: Adjust Readout Weights Only", font_size=20).next_to(readout_layer, UP, buff=0.4)
        self.play(FadeIn(training_label_txt))

        # Animate readout connections changing color/thickness
        trained_connection_anims = [
            conn.animate.set_color(trained_connection_color).set_stroke(width=4.5) # Make trained arrows thicker
            for conn in output_connections
        ]
        # Indicate that pendulum dynamics remain fixed
        fixed_dynamics_indicator = Indicate(fixed_dynamics_label, color=BLUE, scale_factor=1.1)
        # Indicate the fixed part of the equation
        fixed_equation_indicator = Indicate(equation[:input_term_index], color=BLUE)

        self.play(
            AnimationGroup(*trained_connection_anims, lag_ratio=0.1),
            fixed_dynamics_indicator,
            fixed_equation_indicator,
            run_time=2.0
        )
        self.wait(1.5)

        # Revert colors/thickness
        revert_anims = [conn.animate.set_color(connection_color).set_stroke(width=2.5) for conn in output_connections]
        animations_to_play = [FadeOut(training_label_txt)]
        if revert_anims:
            animations_to_play.append(AnimationGroup(*revert_anims))

        self.play(*animations_to_play)
        self.wait(1)

        # --- Final Fade Out ---
        # Remove rod updater BEFORE fading out
        rod.remove_updater(update_rod)
        self.play(FadeOut(section4_title))
        self.play(
            FadeOut(pendulum_system),
            FadeOut(state_sampling_elements),
            FadeOut(readout_layer),
            FadeOut(output_connections),
            FadeOut(title_group)
        )
        self.wait(1)


# --- To run this: ---
# 1. Ensure Manim Community and NumPy are installed.
# 2. Save code as pendulum_rc_animation_final.py
# 3. Run: manim -pql pendulum_rc_animation_final.py PendulumReservoirComputing
#    (Use -pqm for medium quality, -pqh for high quality)
