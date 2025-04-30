from manim import *
import numpy as np
import time  # Add time module import

config.frame_height = 8
config.frame_width = 14
config.pixel_height = 1080
config.pixel_width = 1920

class ReservoirComputingIntro(Scene):
    def construct(self):
        # Title
        title = Text("Reservoir Computing with a Single Pendulum", font_size=48)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(1)
        
        # Basic intro explanation
        intro_text = Text(
            "A novel approach to machine learning using physical systems",
            font_size=28
        )
        intro_text.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(intro_text))
        self.wait(2)
        
        self.play(FadeOut(intro_text))
        
        # Traditional RC vs Single Pendulum
        compare = VGroup()
        traditional = Text("Traditional Reservoir Computing", font_size=32)
        traditional.to_edge(LEFT, buff=2)
        traditional.shift(UP * 1.5)
        
        single_pendulum = Text("Single Pendulum Reservoir", font_size=32)
        single_pendulum.to_edge(RIGHT, buff=2)
        single_pendulum.shift(UP * 1.5)
        
        vs_text = Text("VS", font_size=36)
        vs_text.move_to((traditional.get_right() + single_pendulum.get_left()) / 2)
        
        compare = VGroup(traditional, vs_text, single_pendulum)
        self.play(Write(compare))
        self.wait(1)
        
        # Traditional RC diagram
        trad_sys = VGroup()
        nodes = VGroup()
        
        for i in range(5):
            for j in range(4):
                node = Circle(radius=0.15, color=BLUE, fill_opacity=0.8)
                node.move_to([-4 + i * 0.8, 0 - j * 0.8, 0])
                nodes.add(node)
        
        # Add random connections
        edges = VGroup()
        for _ in range(30):
            n1, n2 = np.random.choice(range(len(nodes)), 2, replace=False)
            edge = Line(
                nodes[n1].get_center(), 
                nodes[n2].get_center(),
                stroke_opacity=0.5
            )
            edges.add(edge)
            
        trad_sys.add(nodes, edges)
        trad_sys.next_to(traditional, DOWN, buff=0.8)
        trad_sys.scale(0.8)
        
        # Single pendulum diagram
        pendulum_group = VGroup()
        pivot = Dot(color=WHITE)
        pivot.shift(RIGHT * 4 + UP * 0.5)
        
        rod = Line(pivot.get_center(), pivot.get_center() + DOWN * 2, color=WHITE)
        bob = Circle(radius=0.3, color=RED, fill_opacity=1).move_to(rod.get_end())
        
        pendulum_group.add(pivot, rod, bob)
        pendulum_group.next_to(single_pendulum, DOWN, buff=0.8)
        
        self.play(
            Create(trad_sys),
            Create(pendulum_group)
        )
        self.wait(2)
        
        # Clear the screen to move to the next part
        self.play(
            FadeOut(compare),
            FadeOut(trad_sys),
            FadeOut(pendulum_group),
            FadeOut(title)
        )
        self.wait(0.5)


class PendulumReservoir(Scene):
    def construct(self):
        # Pendulum physics explanation
        title = Text("Pendulum Dynamics", font_size=42)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        
        # Equation of motion
        eq_motion = MathTex(
            r"\frac{d^2x}{dt^2} = -\frac{g}{l}\sin(x) - k\frac{dx}{dt} + f \cdot \text{sign}[\sin(\omega t)]",
            font_size=36
        )
        eq_motion.next_to(title, DOWN, buff=0.5)
        
        # Explain the equation
        eq_labels = VGroup()
        
        gravity_term = Text("Gravity", font_size=24, color=YELLOW)
        gravity_term.next_to(eq_motion, DOWN, buff=0.8)
        gravity_term.shift(LEFT * 3)
        
        damping_term = Text("Damping", font_size=24, color=GREEN)
        damping_term.next_to(eq_motion, DOWN, buff=0.8)
        
        forcing_term = Text("Periodic forcing", font_size=24, color=RED)
        forcing_term.next_to(eq_motion, DOWN, buff=0.8)
        forcing_term.shift(RIGHT * 3.5)
        
        eq_labels.add(gravity_term, damping_term, forcing_term)
        
        self.play(Write(eq_motion))
        self.wait(1)
        
        # Arrows pointing to respective terms
        gravity_arrow = Arrow(
            gravity_term.get_top(), 
            eq_motion.get_part_by_tex(r"-\frac{g}{l}\sin").get_bottom(),
            buff=0.1,
            color=YELLOW
        )
        
        damping_arrow = Arrow(
            damping_term.get_top(), 
            eq_motion.get_part_by_tex(r"k\frac{dx}{dt}").get_bottom(),
            buff=0.1,
            color=GREEN
        )
        
        forcing_arrow = Arrow(
            forcing_term.get_top(), 
            eq_motion.get_part_by_tex(r"f \cdot \text{sign}").get_bottom(),
            buff=0.1,
            color=RED
        )
        
        arrows = VGroup(gravity_arrow, damping_arrow, forcing_arrow)
        
        self.play(
            Write(eq_labels),
            Create(arrows)
        )
        self.wait(2)
        
        # Clear for next animation
        self.play(
            FadeOut(eq_labels),
            FadeOut(arrows)
        )
        
        # Show pendulum in action
        # Move equation up to make room
        self.play(
            eq_motion.animate.scale(0.8).to_edge(UP, buff=1.5),
            FadeOut(title)
        )
        
        # Create pendulum animation
        pivot = Dot(color=WHITE)
        pendulum_length = 2
        
        # Setup pendulum
        def get_pendulum(theta):
            rod = Line(
                pivot.get_center(),
                pivot.get_center() + pendulum_length * np.array([np.sin(theta), -np.cos(theta), 0]),
                color=WHITE
            )
            bob = Circle(radius=0.3, color=RED, fill_opacity=1).move_to(rod.get_end())
            force_arrow = Arrow(
                bob.get_center(),
                bob.get_center() + RIGHT * 0.8 * np.sign(np.sin(time.time())),
                buff=0.1, color=RED, stroke_width=4
            )
            return VGroup(rod, bob, force_arrow)
        
        # Initial pendulum
        theta_0 = PI/6
        pendulum = get_pendulum(theta_0)
        self.play(Create(pivot), Create(pendulum))
        
        # Add a description of different dynamics
        dynamics_text = Text("Pendulum exhibits rich dynamics:", font_size=28)
        dynamics_text.to_edge(LEFT, buff=1)
        dynamics_text.shift(UP * 1)
        
        modes = VGroup(
            Text("• Periodic", font_size=24, color=BLUE),
            Text("• Quasiperiodic", font_size=24, color=GREEN),
            Text("• Chaotic", font_size=24, color=RED)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        modes.next_to(dynamics_text, DOWN, aligned_edge=LEFT, buff=0.3)
        
        self.play(Write(dynamics_text), Write(modes))
        
        # Simulate pendulum motion
        # Parameters
        g = 9.8
        l = 1.0
        k = 0.05  # damping
        f = 1.5   # forcing amplitude
        omega = 1.0  # forcing frequency
        
        dt = 0.05
        theta = theta_0
        omega_p = 0  # angular velocity
        
        # Pre-calculate some oscillation points for a smooth animation
        thetas = []
        times = []
        
        t = 0
        for _ in range(100):
            force = f * np.sign(np.sin(omega * t))
            omega_p += (-g/l * np.sin(theta) - k * omega_p + force) * dt
            theta += omega_p * dt
            thetas.append(theta)
            times.append(t)
            t += dt
        
        # Animate the pendulum
        self.remove(pendulum)
        trace = VMobject(color=BLUE_C, stroke_width=2, stroke_opacity=0.6)
        trace.set_points_as_corners([get_pendulum(thetas[0])[1].get_center()])
        # Animate the pendulum
        self.remove(pendulum)
        trace = VMobject(color=BLUE_C, stroke_width=2, stroke_opacity=0.6)
        # Access bob from the pendulum object - index 1 is the bob
        trace.set_points_as_corners([get_pendulum(thetas[0])[1].get_center()])
        for i in range(1, len(thetas)):
            new_pendulum = get_pendulum(thetas[i])
            trace.add_points_as_corners([new_pendulum[1].get_center()])
            self.remove(pendulum)
            self.add(new_pendulum)
            pendulum = new_pendulum
            self.wait(dt)
            
        self.wait(1)
        
        # Clear for next part
        self.play(
            FadeOut(pendulum),
            FadeOut(pivot),
            FadeOut(trace),
            FadeOut(dynamics_text),
            FadeOut(modes),
            FadeOut(eq_motion)
        )


class ReservoirArchitecture(Scene):
    def construct(self):
        # Show the full reservoir computing architecture
        title = Text("Reservoir Computing Architecture", font_size=42)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.5)
        
        # Create architecture diagram
        # Three main components: Input, Reservoir, Output
        input_layer = VGroup()
        for i in range(4):
            node = Circle(radius=0.3, color=BLUE, fill_opacity=0.8)
            node.move_to([-5, 2 - i, 0])
            input_layer.add(node)
        
        input_text = Text("Input Layer", font_size=28)
        input_text.next_to(input_layer, DOWN, buff=0.5)
        
        # Reservoir (center) - Pendulum
        pivot = Dot(color=WHITE).move_to([0, 1.5, 0])
        rod = Line(pivot.get_center(), pivot.get_center() + DOWN * 2, color=WHITE)
        bob = Circle(radius=0.4, color=RED, fill_opacity=1).move_to(rod.get_end())
        pendulum = VGroup(pivot, rod, bob)
        
        reservoir_text = Text("Reservoir\n(Fixed Random Weights)", font_size=28, color=RED)
        reservoir_text.next_to(pendulum, DOWN, buff=0.5)
        
        # Output layer (right)
        output_layer = VGroup()
        for i in range(3):
            node = Circle(radius=0.3, color=GREEN, fill_opacity=0.8)
            node.move_to([5, 1.5 - i, 0])
            output_layer.add(node)
        
        output_text = Text("Output Layer\n(Trainable)", font_size=28)
        output_text.next_to(output_layer, DOWN, buff=0.5)
        
        # Connections
        # Input to reservoir
        in_to_res = VGroup()
        for in_node in input_layer:
            connection = Arrow(
                in_node.get_right(),
                pendulum.get_left(),
                buff=0.1,
                stroke_opacity=0.5,
                color=BLUE
            )
            in_to_res.add(connection)
        
        # Reservoir to output
        res_to_out = VGroup()
        for out_node in output_layer:
            connection = Arrow(
                pendulum.get_right(),
                out_node.get_left(),
                buff=0.1,
                stroke_opacity=0.5,
                color=GREEN
            )
            res_to_out.add(connection)
        
        # Display components one by one
        self.play(Create(input_layer), Write(input_text))
        self.wait(0.5)
        
        self.play(Create(pendulum), Write(reservoir_text))
        self.wait(0.5)
        
        self.play(Create(output_layer), Write(output_text))
        self.wait(0.5)
        
        self.play(Create(in_to_res), Create(res_to_out))
        self.wait(2)
        
        # Fade out to prepare for next animation
        self.play(
            FadeOut(input_layer),
            FadeOut(input_text),
            FadeOut(pendulum),
            FadeOut(reservoir_text),
            FadeOut(output_layer),
            FadeOut(output_text),
            FadeOut(in_to_res),
            FadeOut(res_to_out),
            FadeOut(title)
        )


class InputEncoding(Scene):
    def construct(self):
        title = Text("Input Encoding Methods", font_size=42)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        
        # Show two encoding methods
        subtitle1 = Text("Amplitude Encoding", font_size=32)
        subtitle1.to_edge(LEFT, buff=2)
        subtitle1.shift(UP * 1)
        
        subtitle2 = Text("Frequency Encoding", font_size=32)
        subtitle2.to_edge(RIGHT, buff=2)
        subtitle2.shift(UP * 1)
        
        self.play(Write(subtitle1), Write(subtitle2))
        
        # Create visual demonstrations for both methods
        # Amplitude encoding
        amp_group = VGroup()
        pivot1 = Dot(color=WHITE).move_to([-4, 0, 0])
        rod1 = Line(pivot1.get_center(), pivot1.get_center() + DOWN * 2, color=WHITE)
        bob1 = Circle(radius=0.3, color=RED, fill_opacity=1).move_to(rod1.get_end())
        pendulum1 = VGroup(pivot1, rod1, bob1)
        
        input_value1 = Text("Input = 0.8", font_size=24)
        input_value1.next_to(pendulum1, UP, buff=0.5)
        
        force_eq1 = MathTex(r"f = f_{min} + (f_{max} - f_{min}) \cdot input", font_size=24)
        force_eq1.next_to(input_value1, UP, buff=0.3)
        
        # Force arrow showing amplitude
        force_arrow1 = Arrow(
            bob1.get_center() + RIGHT * 0.4,
            bob1.get_center() + RIGHT * 1.2,
            buff=0,
            color=RED
        )
        
        amp_desc = Text("Force amplitude encodes input", font_size=24)
        amp_desc.next_to(pendulum1, DOWN, buff=0.8)
        
        amp_group.add(pendulum1, input_value1, force_eq1, force_arrow1, amp_desc)
        
        # Frequency encoding
        freq_group = VGroup()
        pivot2 = Dot(color=WHITE).move_to([4, 0, 0])
        rod2 = Line(pivot2.get_center(), pivot2.get_center() + DOWN * 2, color=WHITE)
        bob2 = Circle(radius=0.3, color=RED, fill_opacity=1).move_to(rod2.get_end())
        pendulum2 = VGroup(pivot2, rod2, bob2)
        
        input_value2 = Text("Input = 0.8", font_size=24)
        input_value2.next_to(pendulum2, UP, buff=0.5)
        
        freq_eq = MathTex(r"\omega = \omega_{min} + (\omega_{max} - \omega_{min}) \cdot input", font_size=24)
        freq_eq.next_to(input_value2, UP, buff=0.3)
        
        # Sine wave showing frequency
        freq_graph = FunctionGraph(
            lambda x: 0.3 * np.sin(4 * x),
            x_range=[-2, 2],
            color=BLUE
        )
        freq_graph.next_to(pendulum2, DOWN, buff=0.5)
        
        freq_desc = Text("Force frequency encodes input", font_size=24)
        freq_desc.next_to(freq_graph, DOWN, buff=0.3)
        
        freq_group.add(pendulum2, input_value2, freq_eq, freq_graph, freq_desc)
        
        self.play(
            Create(amp_group),
            Create(freq_group)
        )
        self.wait(2)
        
        # Highlighting advantages of frequency encoding
        advantage_text = Text("The paper found frequency encoding to be more robust to noise", font_size=28, color=YELLOW)
        advantage_text.to_edge(DOWN, buff=1)
        
        self.play(Write(advantage_text))
        self.wait(2)
        
        self.play(
            FadeOut(amp_group),
            FadeOut(freq_group),
            FadeOut(advantage_text),
            FadeOut(subtitle1),
            FadeOut(subtitle2),
            FadeOut(title)
        )


class ReservoirComputing(Scene):
    def construct(self):
        title = Text("Reservoir Computing Process", font_size=42)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.5)
        
        # Create step-by-step visual explanation
        steps = VGroup(
            Text("1. Encode input in pendulum parameters", font_size=28),
            Text("2. Record transient dynamics as states", font_size=28),
            Text("3. Form state vectors for each input", font_size=28),
            Text("4. Train output weights using regression", font_size=28),
            Text("5. Use weights to predict new outputs", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        
        steps.next_to(title, DOWN, buff=0.8)
        
        for step in steps:
            self.play(Write(step))
            self.wait(0.7)
        
        self.wait(1)
        
        # Show an example of reservoir states
        self.play(FadeOut(steps))
        
        # Create visualization of reservoir states
        state_title = Text("Reservoir States Visualization", font_size=32)
        state_title.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(state_title))
        
        # Create axes for the time series
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[-2, 2, 1],
            axis_config={"include_tip": False},
            x_axis_config={"numbers_to_include": np.arange(0, 11, 2)},
            y_axis_config={"numbers_to_include": np.arange(-2, 3, 1)}
        )
        
        axes.next_to(state_title, DOWN, buff=0.8)
        
        # Labels
        x_label = Text("Time", font_size=20)
        x_label.next_to(axes.x_axis, DOWN, buff=0.2)
        
        y_label = Text("Pendulum State", font_size=20)
        y_label.next_to(axes.y_axis, LEFT, buff=0.2)
        
        self.play(
            Create(axes),
            Write(x_label),
            Write(y_label)
        )
        
        # Generate pendulum time series data
        # Parameters
        g = 9.8
        l = 1.0
        k = 0.05  # damping
        f = 1.5   # forcing amplitude
        omega = 1.0  # forcing frequency
        
        dt = 0.05
        theta = PI/6
        omega_p = 0  # angular velocity
        
        # Generate data
        times = np.arange(0, 10, dt)
        states = []
        
        t = 0
        for _ in times:
            force = f * np.sign(np.sin(omega * t))
            omega_p += (-g/l * np.sin(theta) - k * omega_p + force) * dt
            theta += omega_p * dt
            states.append(theta)
            t += dt
        
        # Plot the time series
        graph = VMobject()
        graph.set_points_as_corners([
            axes.c2p(t, s) for t, s in zip(times, states)
        ])
        graph.set_color(RED)
        
        self.play(Create(graph), run_time=3)
        self.wait(1)
        
        # Explain that this state information is used for learning
        explanation = Text(
            "These rich dynamic patterns contain the transformed input information",
            font_size=24
        )
        explanation.next_to(axes, DOWN, buff=0.5)
        
        self.play(Write(explanation))
        self.wait(2)
        
        self.play(
            FadeOut(graph),
            FadeOut(axes),
            FadeOut(x_label),
            FadeOut(y_label),
            FadeOut(explanation),
            FadeOut(state_title),
            FadeOut(title)
        )


class TasksAndResults(Scene):
    def construct(self):
        title = Text("Tasks & Results", font_size=42)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        
        # Task 1 - Polynomial approximation
        task1_title = Text("Task 1: Polynomial Approximation", font_size=32)
        task1_title.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(task1_title))
        
        # Create axes for the polynomial
        axes1 = Axes(
            x_range=[-3, 3, 1],
            y_range=[-50, 50, 25],
            axis_config={"include_tip": False},
            x_axis_config={"numbers_to_include": np.arange(-3, 4, 1)},
            y_axis_config={"numbers_to_include": np.arange(-50, 75, 25)}
        )
        
        axes1.next_to(task1_title, DOWN, buff=0.5)
        axes1.scale(0.7)
        
        # Plot polynomial f(x) = (x-3)(x-2)(x-1)x(x+1)(x+2)(x+3)
        def poly(x):
            return (x-3)*(x-2)*(x-1)*x*(x+1)*(x+2)*(x+3)
        
        x_vals = np.linspace(-3, 3, 100)
        poly_graph = VMobject()
        poly_graph.set_points_as_corners([
            axes1.c2p(x, poly(x)) for x in x_vals
        ])
        poly_graph.set_color(BLUE)
        
        poly_label = Text("f(x) = (x-3)(x-2)(x-1)x(x+1)(x+2)(x+3)", font_size=20)
        poly_label.next_to(axes1, UP, buff=0.2)
        
        self.play(
            Create(axes1),
            Write(poly_label)
        )
        
        self.play(Create(poly_graph))
        
        # Add RMSE result
        result1 = Text("RMSE: 10^-10 (Amplitude Encoding)", font_size=24, color=GREEN)
        result1.next_to(axes1, DOWN, buff=0.3)
        
        self.play(Write(result1))
        self.wait(1.5)
        
        # Clear for Task 2
        self.play(
            FadeOut(axes1),
            FadeOut(poly_graph),
            FadeOut(poly_label),
            FadeOut(result1),
            FadeOut(task1_title)
        )
        
        # Task 2 - Lorenz attractor reconstruction
        task2_title = Text("Task 2: Chaotic System State Inference", font_size=32)
        task2_title.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(task2_title))
        
        # Create Lorenz attractor visualization
        lorenz_text = Text("Lorenz System: Predict z-coordinate from x-coordinate", font_size=24)
        lorenz_text.next_to(task2_title, DOWN, buff=0.5)
        
        self.play(Write(lorenz_text))
        
        # Create simplified 2D projection visualization
        axes2 = Axes(
            x_range=[-20, 20, 10],
            y_range=[-20, 20, 10],
            axis_config={"include_tip": False},
            x_axis_config={"numbers_to_include": np.arange(-20, 30, 10)},
            y_axis_config={"numbers_to_include": np.arange(-20, 30, 10)}
        )
        
        axes2.next_to(lorenz_text, DOWN, buff=0.5)
        axes2.scale(0.7)
        
        # Generate a projection of Lorenz attractor (simplified for visualization)
        lorenz_points = VGroup()
        
        for angle in np.linspace(0, 4*np.pi, 200):
            x = 10 * np.sin(angle) + 5 * np.sin(2.1*angle)
            y = 10 * np.cos(angle) + 5 * np.cos(1.9*angle)
            point = Dot(axes2.c2p(x, y), radius=0.03, color=BLUE_A)
            lorenz_points.add(point)
        
        x_label = Text("x", font_size=20)
        x_label.next_to(axes2.x_axis, DOWN, buff=0.2)
        
        z_label = Text("z", font_size=20)
        z_label.next_to(axes2.y_axis, LEFT, buff=0.2)
        
        self.play(
            Create(axes2),
            Write(x_label),
            Write(z_label)
        )
        
        self.play(Create(lorenz_points))
        
        # Add RMSE result
        result2 = Text("RMSE: 10^-5 (Both Encoding Methods)", font_size=24, color=GREEN)
        result2.next_to(axes2, DOWN, buff=0.3)
        
        self.play(Write(result2))
        self.wait(2)
        
        # Clear for conclusion
        self.play(
            FadeOut(lorenz_points),
            FadeOut(axes2),
            FadeOut(x_label),
            FadeOut(z_label),
            FadeOut(result2),
            FadeOut(lorenz_text),
            FadeOut(task2_title),
            FadeOut(title)
        )


class Conclusion(Scene):
    def construct(self):
        title = Text("Key Insights", font_size=42)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        
        # Summarize key points
        key_points = VGroup(
            Text("• Single pendulum can function as a powerful reservoir", font_size=28),
            Text("• Transient dynamics offers rich computational patterns", font_size=28),
            Text("• Frequency encoding is more robust to noise than amplitude encoding", font_size=28),
            Text("• Simple physical systems have remarkable machine learning potential", font_size=28),
            Text("• Reservoir computing bridges physics and machine learning", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        
        key_points.next_to(title, DOWN, buff=0.8)
        
        for point in key_points:
            self.play(Write(point))
            self.wait(0.7)
        
        self.wait(1)
        
        # Final message
        final_message = Text(
            "A single simple nonlinear system can perform\ncomplex computational tasks with high accuracy",
            font_size=36, color=YELLOW
        )
        
        self.play(
            FadeOut(key_points),
            FadeOut(title)
        )
        
        self.play(Write(final_message))
        self.wait(3)
        
        self.play(FadeOut(final_message))


class FullAnimation(Scene):
    def construct(self):
        # Run all scenes in sequence
        ReservoirComputingIntro.construct(self)
        PendulumReservoir.construct(self)
        ReservoirArchitecture.construct(self)
        InputEncoding.construct(self)
        ReservoirComputing.construct(self)
        TasksAndResults.construct(self)
        Conclusion.construct(self)
        

if __name__ == "__main__":
    # Render the main animation
    module_name = "FullAnimation"