# Save this code as reservoir_animation_fixed_v2.py
# Run from terminal: manim -pql reservoir_animation_fixed_v2.py ReservoirComputingAnimation

from manim import *
import networkx as nx
import numpy as np

# Although ArrowTriangleTip is often the default, importing explicitly can sometimes help clarity/avoid issues
from manim import ArrowTriangleTip

class ReservoirComputingAnimation(Scene):
    def construct(self):
        np.random.seed(42) # for reproducible reservoir layout

        # --- Configuration ---
        input_color = BLUE
        reservoir_color = RED_B # Color for reservoir nodes
        reservoir_edge_color = GRAY
        readout_color = GREEN
        signal_color = YELLOW
        connection_color = WHITE
        trained_connection_color = ORANGE

        reservoir_node_count = 15
        reservoir_connection_prob = 0.25

        # --- Texts and Titles ---
        title = Text("Reservoir Computing").scale(0.8)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        arch_title = Text("1. Architecture", font_size=30).next_to(title, DOWN, buff=0.5).to_edge(LEFT)
        self.play(FadeIn(arch_title))

        # --- 1. Architecture ---
        # Input Layer
        input_nodes = VGroup(*[Dot(color=input_color) for _ in range(3)]).arrange(DOWN, buff=0.5)
        input_label = Text("Input", font_size=24).next_to(input_nodes, DOWN)
        input_layer = VGroup(input_nodes, input_label).shift(LEFT * 5.5)

        # Reservoir Layer
        G = nx.gnp_random_graph(reservoir_node_count, reservoir_connection_prob, directed=True, seed=42)
        G.remove_edges_from(nx.selfloop_edges(G))
        pos = nx.circular_layout(G)
        scale_factor = 2.5
        manim_pos = {node: np.array([pos[node][0], pos[node][1], 0]) * scale_factor for node in G.nodes()}

        # *** CORRECTED PART V2 ***
        # Use DiGraph, nested tip_config, BUT OMIT tip_width
        reservoir_edge_config = {
            "color": reservoir_edge_color,
            "stroke_width": 2,
            "tip_config": {
                "tip_shape": ArrowTriangleTip, # Specify shape
                "tip_length": 0.15,          # Specify length
                # "tip_width": 0.1,         # OMITTING tip_width
            }
        }

        reservoir_graph = DiGraph(
            list(G.nodes()),
            list(G.edges()),
            layout=manim_pos,
            vertex_config={"radius": 0.15, "color": reservoir_color, "fill_opacity": 0.8},
            edge_config=reservoir_edge_config # Use the config without tip_width
        ).shift(RIGHT * 0)
        # *** END CORRECTION V2 ***

        reservoir_label = Text("Reservoir", font_size=24).next_to(reservoir_graph, DOWN, buff=0.3)
        fixed_label = Text("(Fixed & Recurrent)", font_size=18).next_to(reservoir_label, DOWN, buff=0.1)
        reservoir_layer = VGroup(reservoir_graph, reservoir_label, fixed_label)

        # Readout Layer
        readout_nodes = VGroup(*[Dot(color=readout_color) for _ in range(2)]).arrange(DOWN, buff=0.5)
        readout_label = Text("Readout", font_size=24).next_to(readout_nodes, DOWN)
        trainable_label = Text("(Trainable)", font_size=18).next_to(readout_label, DOWN, buff=0.1)
        readout_layer = VGroup(readout_nodes, readout_label, trainable_label).shift(RIGHT * 5.5)

        self.play(
            FadeIn(input_layer, shift=RIGHT),
            Create(reservoir_graph),
            FadeIn(VGroup(reservoir_label, fixed_label)),
            FadeIn(readout_layer, shift=LEFT),
            run_time=2
        )
        self.wait(0.5)

        # Connections (Input -> Reservoir)
        input_connections = VGroup()
        input_targets = np.random.choice(list(G.nodes()), len(input_nodes) * 2, replace=False)
        k = 0
        for i, i_node in enumerate(input_nodes):
            num_connections_per_input = 2
            for _ in range(num_connections_per_input):
                if k < len(input_targets):
                    r_node_idx = input_targets[k]
                    # Ensure the target vertex exists before creating arrow
                    if r_node_idx in reservoir_graph.vertices:
                         arrow = Arrow(i_node.get_center(), reservoir_graph.vertices[r_node_idx].get_center(),
                                       buff=0.15, color=connection_color, stroke_width=2, max_tip_length_to_length_ratio=0.1)
                         input_connections.add(arrow)
                    k += 1

        # Connections (Reservoir -> Readout)
        output_connections = VGroup()
        output_sources = np.random.choice(list(G.nodes()), len(readout_nodes) * 3, replace=False)
        k = 0
        for o_node in readout_nodes:
            num_connections_per_output = 3
            for _ in range(num_connections_per_output):
                 if k < len(output_sources):
                    r_node_idx = output_sources[k]
                     # Ensure the source vertex exists before creating arrow
                    if r_node_idx in reservoir_graph.vertices:
                         arrow = Arrow(reservoir_graph.vertices[r_node_idx].get_center(), o_node.get_center(),
                                       buff=0.15, color=connection_color, stroke_width=2, max_tip_length_to_length_ratio=0.1)
                         output_connections.add(arrow)
                    k += 1

        self.play(
            LaggedStart(*[GrowArrow(a) for a in input_connections], lag_ratio=0.1, run_time=1.5),
            LaggedStart(*[GrowArrow(a) for a in output_connections], lag_ratio=0.1, run_time=1.5),
        )
        self.wait(2)

        # --- Cleanup Architecture ---
        self.play(FadeOut(arch_title))
        arch_group = VGroup(input_layer, reservoir_layer, readout_layer, input_connections, output_connections)

        # --- 2. Signal Input & Reservoir Dynamics ---
        dynamics_title = Text("2. Signal Processing & Reservoir Dynamics", font_size=30).next_to(title, DOWN, buff=0.5).to_edge(LEFT)
        self.play(FadeIn(dynamics_title))

        # Simulate input signal
        signal_input_node_index = 1 # Use middle input node
        signal_pulse = Dot(input_nodes[signal_input_node_index].get_center() + LEFT*0.5, color=signal_color, radius=0.15)
        self.play(FadeIn(signal_pulse, scale=0.5))
        self.play(signal_pulse.animate.move_to(input_nodes[signal_input_node_index].get_center()))
        self.play(Indicate(input_nodes[signal_input_node_index], color=signal_color, scale_factor=1.5))
        self.play(FadeOut(signal_pulse))

        # Signal propagation to Reservoir
        flashes = []
        activated_input_center = input_nodes[signal_input_node_index].get_center()
        initial_activated_nodes = []
        for arrow in input_connections:
             if np.linalg.norm(arrow.get_start() - activated_input_center) < 0.1: # Check if arrow starts near the activated node
                 flashes.append(ShowPassingFlash(arrow.copy().set_color(signal_color), time_width=0.3))
                 for idx, vertex in reservoir_graph.vertices.items():
                     if np.allclose(arrow.get_end(), vertex.get_center(), atol=1e-4):
                         initial_activated_nodes.append(idx)
                         break
        if flashes:
            self.play(AnimationGroup(*flashes, lag_ratio=0.1), run_time=1)

        # Reservoir Activation Spread
        activated_nodes = set(initial_activated_nodes)
        node_activation_anims = [reservoir_graph.vertices[i].animate.set_color(signal_color).scale(1.3)
                                 for i in initial_activated_nodes if i in reservoir_graph.vertices] # Check if node exists
        if node_activation_anims:
            self.play(AnimationGroup(*node_activation_anims, lag_ratio=0.05), run_time=0.5)
        else:
             self.wait(0.1)

        # Simulate spread over a few time steps
        all_activated_nodes_over_time = set(activated_nodes) # Keep track of all nodes activated
        for step in range(3):
            newly_activated_this_step = set()
            spread_flashes = []
            activation_anims = []
            deactivation_anims = []

            current_active_list = list(activated_nodes) # Nodes active at the start of this step

            # Nodes that were active might start fading (simple decay visualization)
            nodes_to_fade = activated_nodes - newly_activated_this_step - set(initial_activated_nodes)
            if step > 0:
                 deactivation_anims = [
                     reservoir_graph.vertices[i].animate.set_color(
                         interpolate_color(signal_color, reservoir_color, 0.3 + step*0.2)
                     ).scale(1.0)
                    for i in nodes_to_fade if i in reservoir_graph.vertices] # Check node exists

            # Find nodes activated by currently active nodes
            for source_node_idx in current_active_list:
                if source_node_idx not in G: continue # Skip if node somehow not in original networkx graph
                neighbors = list(G.successors(source_node_idx))
                for target_node_idx in neighbors:
                    edge = (source_node_idx, target_node_idx)
                    if edge in reservoir_graph.edges:
                         edge_mob = reservoir_graph.edges[edge]
                         spread_flashes.append(ShowPassingFlash(edge_mob.copy().set_color(signal_color), time_width=0.4))
                         if target_node_idx not in all_activated_nodes_over_time: # Activate only if never activated before in this sequence
                             if target_node_idx in reservoir_graph.vertices: # Check node exists
                                 newly_activated_this_step.add(target_node_idx)

            # Animate newly activated nodes
            if newly_activated_this_step:
                activation_anims = [reservoir_graph.vertices[i].animate.set_color(signal_color).scale(1.3)
                                    for i in newly_activated_this_step] # No need to check existence, done above

            # Play animations for this step
            anim_group_list = []
            if spread_flashes: anim_group_list.append(AnimationGroup(*spread_flashes, lag_ratio=0.02))
            if activation_anims: anim_group_list.append(AnimationGroup(*activation_anims, lag_ratio=0.05))
            if deactivation_anims: anim_group_list.append(AnimationGroup(*deactivation_anims, lag_ratio=0.02))

            if anim_group_list:
                self.play(AnimationGroup(*anim_group_list, lag_ratio=0.3), run_time=1.0)
            else:
                self.wait(0.5)

            # Update the set of currently 'brightest' nodes for the next iteration
            # Nodes activated this step replace faded ones as the 'active' set
            activated_nodes = newly_activated_this_step
            # Add newly activated to the history
            all_activated_nodes_over_time.update(newly_activated_this_step)

        # Highlight the complex state
        state_text = Text("Complex Internal State", font_size=24).next_to(reservoir_graph, UP, buff=0.5)
        self.play(FadeIn(state_text))
        highlight_box = SurroundingRectangle(reservoir_graph, color=YELLOW, buff=0.2)
        self.play(Create(highlight_box))
        self.play(FadeOut(highlight_box))
        self.wait(0.5)
        self.play(FadeOut(state_text))

        # --- 3. Readout & Training ---
        self.play(FadeOut(dynamics_title))
        train_title = Text("3. Readout & Training", font_size=30).next_to(title, DOWN, buff=0.5).to_edge(LEFT)
        self.play(FadeIn(train_title))

        # Signal propagation to Readout
        readout_flashes = []
        activated_readout_indices = set()
        relevant_output_connections = []

        # Use all nodes that were activated at any point during the dynamics simulation
        final_active_reservoir_nodes = all_activated_nodes_over_time

        for conn in output_connections:
            source_node_idx = -1
            for idx, vertex in reservoir_graph.vertices.items():
                 if np.allclose(conn.get_start(), vertex.get_center(), atol=1e-4):
                     source_node_idx = idx
                     break
            if source_node_idx != -1 and source_node_idx in final_active_reservoir_nodes:
                 readout_flashes.append(ShowPassingFlash(conn.copy().set_color(signal_color), time_width=0.3))
                 relevant_output_connections.append(conn)
                 for idx, r_node in enumerate(readout_nodes):
                     if np.allclose(conn.get_end(), r_node.get_center(), atol=1e-4):
                         activated_readout_indices.add(idx)
                         break
        if readout_flashes:
            self.play(AnimationGroup(*readout_flashes, lag_ratio=0.1), run_time=1)

        # Indicate activated readout nodes
        readout_activation_anims = [Indicate(readout_nodes[i], color=signal_color, scale_factor=1.5)
                                    for i in activated_readout_indices]
        if readout_activation_anims:
            self.play(AnimationGroup(*readout_activation_anims, lag_ratio=0.1), run_time=0.5)

        # Emphasize Training Readout ONLY
        training_label = Text("Training: Adjust Readout Weights Only", font_size=24).next_to(readout_layer, RIGHT, buff=0.8)
        self.play(FadeIn(training_label))

        # Animate readout connections changing color/thickness
        trained_connection_anims = [
            conn.animate.set_color(trained_connection_color).set_stroke(width=4)
            for conn in output_connections # Animate all readout connections
        ]
        # Indicate reservoir connections remain fixed
        reservoir_edges_group = VGroup(*reservoir_graph.edges.values())
        fixed_reservoir_indicator = Indicate(reservoir_edges_group, color=BLUE, scale_factor=1.0)
        fixed_label_indicator = Indicate(fixed_label, color=BLUE)

        self.play(
            AnimationGroup(*trained_connection_anims, lag_ratio=0.05),
            fixed_reservoir_indicator,
            fixed_label_indicator,
            run_time=2
        )
        self.wait(1)

        # Revert colors/thickness and node colors
        # *** CORRECTED FOR EMPTY ANIMATION GROUP ***
        revert_connections_anims = [conn.animate.set_color(connection_color).set_stroke(width=2)
                                    for conn in output_connections]
        revert_nodes_anims = [reservoir_graph.vertices[i].animate.set_color(reservoir_color).scale(1.0)
                              for i in final_active_reservoir_nodes if i in reservoir_graph.vertices] # Check node exists

        animations_to_play = [FadeOut(training_label)]
        if revert_connections_anims:
            animations_to_play.append(AnimationGroup(*revert_connections_anims))
        if revert_nodes_anims:
            animations_to_play.append(AnimationGroup(*revert_nodes_anims))

        if len(animations_to_play) > 1: # Only play if there's more than just FadeOut
            self.play(*animations_to_play)
        else:
            self.play(animations_to_play[0]) # Just play the FadeOut
        # *** END CORRECTION ***
        self.wait(1)


        # --- Final Fade Out ---
        self.play(FadeOut(train_title))
        self.play(FadeOut(arch_group), FadeOut(title))
        self.wait(1)


# --- To run this: ---
# 1. Ensure Manim Community and NetworkX are installed.
# 2. Save code as reservoir_animation_fixed_v2.py
# 3. Run: manim -pql reservoir_animation_fixed_v2.py ReservoirComputingAnimation
