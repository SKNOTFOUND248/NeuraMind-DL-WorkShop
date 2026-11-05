"""
NeuraMind Deep Learning Workshop - Sample Animation (V2 - Clean Layout & Captions)
Topic: The Atom of Deep Learning & Real Learning in Action
Engine: Manim Community Edition
"""

import os
import json
from manim import *
from caption_system import ScreenLayout, CaptionManager

# Load audio manifest for exact synchronisation
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "audio", "manifest.json")
if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH, "r") as f:
        MANIFEST = {item["act"]: item for item in json.load(f)}
else:
    MANIFEST = {}


class DeepLearningAtomScene(Scene):
    def construct(self):
        # Professional 3b1b-style dark theme background
        self.camera.background_color = "#0e1117"

        # Initialize dedicated bottom caption manager
        captions = CaptionManager(self)

        def get_act_info(act_num, default_dur=8.0):
            if act_num in MANIFEST:
                return MANIFEST[act_num]["duration"], MANIFEST[act_num]["path"]
            return default_dur, None

        # -------------------------------------------------------------
        # ACT 1: Introduction to the Artificial Neuron (~9.0s)
        # -------------------------------------------------------------
        dur1, audio1 = get_act_info(1, 9.0)
        if audio1 and os.path.exists(audio1):
            self.add_sound(audio1)

        # Title Screen in Main Stage Area (Zero collision with caption zone)
        club_badge = Text("NEURAMIND WORKSHOP", font_size=22, color="#00cec9", weight=BOLD)
        title_main = Text("Deep Learning: The Atom of Learning", font_size=34, color=WHITE, weight=BOLD)
        subtitle = Text("How a single computational neuron learns patterns from data", font_size=18, color="#a4b0be")
        
        intro_group = VGroup(club_badge, title_main, subtitle).arrange(DOWN, buff=0.3).move_to(ScreenLayout.STAGE_CENTER)

        # Fade in title and first caption
        cap1_anims = captions.set_caption(
            "Welcome to NeuraMind. In deep learning, every neural network begins with an atomic unit: the artificial neuron."
        )
        self.play(
            FadeIn(intro_group, shift=UP * 0.3),
            *cap1_anims,
            run_time=1.2
        )
        self.wait(2.0)

        # Create Persistent Top Header (strictly at ScreenLayout.HEADER_Y = 3.3)
        header = Text("The Artificial Neuron (Forward Pass)", font_size=24, color="#74b9ff", weight=BOLD)
        header.move_to([0, ScreenLayout.HEADER_Y, 0])

        self.play(
            FadeOut(intro_group),
            FadeIn(header),
            run_time=1.0
        )
        self.wait(max(0.5, dur1 - 4.2))

        # -------------------------------------------------------------
        # ACT 2: Forward Pass & Architecture (~9.3s)
        # -------------------------------------------------------------
        dur2, audio2 = get_act_info(2, 9.3)
        if audio2 and os.path.exists(audio2):
            self.add_sound(audio2)

        cap2_anims = captions.set_caption(
            "The neuron receives input features, multiplies them by connection weights, adds a bias, and passes the sum through an activation function."
        )

        # Build Clean Neuron Architecture (Centered in Stage: y in [-2.0, 2.2])
        stage_y = 0.2

        # Input nodes
        x1_circle = Circle(radius=0.45, color="#0984e3", fill_color="#0984e3", fill_opacity=0.3).move_to([-3.8, stage_y + 1.2, 0])
        x1_label = Text("x1", font_size=20, color=WHITE, weight=BOLD).move_to(x1_circle.get_center())
        x1_desc = Text("Feature 1", font_size=13, color="#74b9ff").next_to(x1_circle, LEFT, buff=0.15)
        x1_group = VGroup(x1_circle, x1_label, x1_desc)

        x2_circle = Circle(radius=0.45, color="#0984e3", fill_color="#0984e3", fill_opacity=0.3).move_to([-3.8, stage_y - 1.2, 0])
        x2_label = Text("x2", font_size=20, color=WHITE, weight=BOLD).move_to(x2_circle.get_center())
        x2_desc = Text("Feature 2", font_size=13, color="#74b9ff").next_to(x2_circle, LEFT, buff=0.15)
        x2_group = VGroup(x2_circle, x2_label, x2_desc)

        # Central Neuron Body (Soma)
        soma = Circle(radius=1.05, color="#6c5ce7", fill_color="#6c5ce7", fill_opacity=0.25).move_to([0, stage_y, 0])
        soma_sum = Text("Sum (Sigma)", font_size=18, color="#a29bfe", weight=BOLD).move_to(soma.get_center() + UP * 0.3)
        soma_div = Line(soma.get_left() * 0.85, soma.get_right() * 0.85, color="#6c5ce7", stroke_width=1.5).move_to(soma.get_center())
        soma_act = Text("f(z) [Step]", font_size=16, color="#dfe6e9").move_to(soma.get_center() + DOWN * 0.35)
        soma_group = VGroup(soma, soma_sum, soma_div, soma_act)

        # Connections & Weights
        arrow1 = Arrow(x1_circle.get_right(), soma.get_left() + UP * 0.35, buff=0.1, color="#fdcb6e", stroke_width=3)
        w1_tag = Text("w1", font_size=18, color="#ffeaa7", weight=BOLD).next_to(arrow1, UP, buff=0.08).shift(LEFT * 0.2)

        arrow2 = Arrow(x2_circle.get_right(), soma.get_left() + DOWN * 0.35, buff=0.1, color="#fdcb6e", stroke_width=3)
        w2_tag = Text("w2", font_size=18, color="#ffeaa7", weight=BOLD).next_to(arrow2, DOWN, buff=0.08).shift(LEFT * 0.2)

        # Bias from below
        bias_circle = Circle(radius=0.35, color="#00b894", fill_color="#00b894", fill_opacity=0.3).move_to([0, stage_y - 2.0, 0])
        bias_label = Text("1", font_size=18, color=WHITE, weight=BOLD).move_to(bias_circle.get_center())
        bias_arrow = Arrow(bias_circle.get_top(), soma.get_bottom(), buff=0.1, color="#55efc4", stroke_width=3)
        bias_tag = Text("+ b (bias)", font_size=15, color="#55efc4", weight=BOLD).next_to(bias_arrow, RIGHT, buff=0.12)
        bias_group = VGroup(bias_circle, bias_label, bias_arrow, bias_tag)

        # Output Node
        out_arrow = Arrow(soma.get_right(), soma.get_right() + RIGHT * 2.2, buff=0.1, color="#00cec9", stroke_width=3)
        out_circle = Circle(radius=0.45, color="#00cec9", fill_color="#00cec9", fill_opacity=0.3).next_to(out_arrow, RIGHT, buff=0.1)
        out_label = Text("y_hat", font_size=18, color=WHITE, weight=BOLD).move_to(out_circle.get_center())
        out_desc = Text("Output (0 or 1)", font_size=13, color="#81ecec").next_to(out_circle, UP, buff=0.15)
        out_group = VGroup(out_arrow, out_circle, out_label, out_desc)

        # Compact Formula Card positioned in upper-middle stage
        formula_badge = RoundedRectangle(
            corner_radius=0.15, height=0.6, width=6.5,
            fill_color="#1e272e", fill_opacity=0.9, stroke_color="#485460", stroke_width=1.5
        ).move_to([0, stage_y + 2.1, 0])
        formula_text = Text("z = w1·x1 + w2·x2 + b    |    Output = Step(z)", font_size=16, color="#f5f6fa").move_to(formula_badge)
        formula_group = VGroup(formula_badge, formula_text)

        neuron_model = VGroup(
            x1_group, x2_group, soma_group,
            arrow1, w1_tag, arrow2, w2_tag,
            bias_group, out_group, formula_group
        )

        # Animate Assembly with clean visual flow
        self.play(
            *cap2_anims,
            FadeIn(x1_group, shift=RIGHT * 0.3),
            FadeIn(x2_group, shift=RIGHT * 0.3),
            GrowFromCenter(soma_group),
            FadeIn(bias_group),
            run_time=1.4
        )
        self.play(
            GrowArrow(arrow1), FadeIn(w1_tag),
            GrowArrow(arrow2), FadeIn(w2_tag),
            GrowArrow(out_arrow), FadeIn(out_circle), FadeIn(out_label), FadeIn(out_desc),
            FadeIn(formula_group, shift=DOWN * 0.2),
            run_time=1.4
        )

        # Animated signal pulse into soma
        pulse1 = Dot(color="#ffeaa7", radius=0.09).move_to(x1_circle.get_center())
        pulse2 = Dot(color="#ffeaa7", radius=0.09).move_to(x2_circle.get_center())
        self.play(
            pulse1.animate.move_to(soma.get_center()),
            pulse2.animate.move_to(soma.get_center()),
            run_time=1.1
        )
        self.remove(pulse1, pulse2)
        self.play(Indicate(soma, color="#a29bfe", scale_factor=1.08), run_time=0.8)
        self.wait(max(0.5, dur2 - 4.7))

        # -------------------------------------------------------------
        # ACT 3: Geometric Decision Boundary (~7.8s)
        # -------------------------------------------------------------
        dur3, audio3 = get_act_info(3, 7.8)
        if audio3 and os.path.exists(audio3):
            self.add_sound(audio3)

        cap3_anims = captions.set_caption(
            "In feature space, this linear formula forms a decision boundary line, separating positive and negative classes."
        )

        geo_header = Text("Geometric Interpretation: Feature Space", font_size=24, color="#00cec9", weight=BOLD)
        geo_header.move_to([0, ScreenLayout.HEADER_Y, 0])

        # 2D Coordinate System - Sized safely: Y from -1.8 to +1.8, X from -2.8 to +2.8
        # Will NEVER collide with header (y=3.3) or captions (y=-3.45)!
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=6.0,
            y_length=4.2,
            axis_config={"color": "#57606f", "stroke_width": 2, "include_ticks": True}
        ).move_to([0, -0.1, 0])

        # Clean axis labels placed safely away from arrow tips
        axis_x_label = Text("Feature x1", font_size=15, color="#a4b0be").next_to(axes.x_axis.get_end(), RIGHT, buff=0.15)
        axis_y_label = Text("Feature x2", font_size=15, color="#a4b0be").next_to(axes.y_axis.get_end(), UP, buff=0.15)
        axis_labels = VGroup(axis_x_label, axis_y_label)

        # Clean Transition: Fade out neuron entirely so there is NO unreadable corner clutter!
        self.play(
            *cap3_anims,
            Transform(header, geo_header),
            FadeOut(neuron_model),
            Create(axes),
            FadeIn(axis_labels),
            run_time=1.5
        )

        # Scatter Data Points (Linearly Separable)
        # Class 1 (Blue): Top-Right quadrant
        blue_pts = [[1.2, 1.2], [2.1, 0.7], [0.7, 1.6], [1.8, 1.5]]
        blue_dots = VGroup(*[Dot(axes.c2p(p[0], p[1]), color="#0984e3", radius=0.1) for p in blue_pts])
        blue_legend = Text("Class 1", font_size=14, color="#74b9ff").move_to(axes.c2p(2.2, 2.1))

        # Class 0 (Orange): Bottom-Left quadrant
        orange_pts = [[-1.5, -0.7], [-0.8, -1.6], [-2.0, -1.3], [-1.4, -1.9]]
        orange_dots = VGroup(*[Dot(axes.c2p(p[0], p[1]), color="#e17055", radius=0.1) for p in orange_pts])
        orange_legend = Text("Class 0", font_size=14, color="#fab1a0").move_to(axes.c2p(-2.2, -2.1))

        self.play(
            FadeIn(blue_dots, scale=0.5), FadeIn(blue_legend),
            FadeIn(orange_dots, scale=0.5), FadeIn(orange_legend),
            run_time=1.1
        )

        # Decision Line: Standard ASCII equation badge (NO unicode missing-glyph boxes!)
        # Boundary: x2 = -0.85 * x1 + 0.1
        decision_line = axes.plot(lambda x: -0.85 * x + 0.1, color="#fdcb6e", stroke_width=3.5)
        
        # Dedicated Boundary Label Badge positioned safely in top-left stage area
        boundary_badge = RoundedRectangle(
            corner_radius=0.12, height=0.55, width=4.8,
            fill_color="#1e272e", fill_opacity=0.9, stroke_color="#fdcb6e", stroke_width=1.2
        ).move_to([-3.4, 2.2, 0])
        boundary_label = Text("Boundary: w1·x1 + w2·x2 + b = 0", font_size=14, color="#ffeaa7").move_to(boundary_badge)
        boundary_group = VGroup(boundary_badge, boundary_label)

        self.play(Create(decision_line), FadeIn(boundary_group, shift=DOWN * 0.1), run_time=1.3)
        self.wait(max(0.5, dur3 - 3.9))

        # -------------------------------------------------------------
        # ACT 4: Real Learning in Action (~9.2s)
        # -------------------------------------------------------------
        dur4, audio4 = get_act_info(4, 9.2)
        if audio4 and os.path.exists(audio4):
            self.add_sound(audio4)

        cap4_anims = captions.set_caption(
            "When an untrained model misclassifies a point, the learning rule triggers. It adjusts weights to pull the boundary directly toward the error."
        )

        learn_header = Text("Real Learning: The Perceptron Update Rule", font_size=24, color="#ff7675", weight=BOLD)
        learn_header.move_to([0, ScreenLayout.HEADER_Y, 0])

        # Untrained line cuts through wrongly: y = 0.9 * x + 0.2
        bad_line = axes.plot(lambda x: 0.9 * x + 0.2, color="#d63031", stroke_width=3.5)

        # Add a misclassified Blue point at (1.5, 0.4) that falls on the wrong side
        misclass_pt = Dot(axes.c2p(1.5, 0.4), color="#0984e3", radius=0.12)
        warning_ring = Circle(radius=0.25, color="#ff7675", stroke_width=3).move_to(misclass_pt.get_center())
        
        # Alert tag positioned safely to the right of the point
        alert_tag = Text("Error: Predicted 0, True 1", font_size=13, color="#ff7675", weight=BOLD)
        alert_tag.next_to(misclass_pt, RIGHT, buff=0.18)

        self.play(
            *cap4_anims,
            Transform(header, learn_header),
            Transform(decision_line, bad_line),
            FadeIn(misclass_pt),
            run_time=1.2
        )

        # Alert pulse
        self.play(
            Create(warning_ring),
            FadeIn(alert_tag),
            Indicate(misclass_pt, color="#ff7675", scale_factor=1.3),
            run_time=1.1
        )

        # Reusable Update Rule Card (positioned top-right, perfectly clear of all points and axes)
        rule_badge = RoundedRectangle(
            corner_radius=0.12, height=0.55, width=5.6,
            fill_color="#2d3436", fill_opacity=0.95, stroke_color="#e17055", stroke_width=1.5
        ).move_to([3.1, 2.2, 0])
        rule_label = Text("Update: w_new = w_old + eta · (y - y_hat) · x", font_size=13, color="#ffeaa7").move_to(rule_badge)
        rule_group = VGroup(rule_badge, rule_label)

        self.play(FadeIn(rule_group, shift=LEFT * 0.2), run_time=1.0)

        # The boundary dynamically pivots and shifts across the misclassified point
        corrected_line = axes.plot(lambda x: -0.9 * x + 0.15, color="#00b894", stroke_width=4.0)

        self.play(
            Transform(decision_line, corrected_line),
            FadeOut(warning_ring),
            FadeOut(alert_tag),
            misclass_pt.animate.set_color("#00cec9"),
            run_time=2.0
        )
        self.wait(max(0.5, dur4 - 5.3))

        # -------------------------------------------------------------
        # ACT 5: Convergence & Deep Learning Bridge (~8.0s)
        # -------------------------------------------------------------
        dur5, audio5 = get_act_info(5, 8.0)
        if audio5 and os.path.exists(audio5):
            self.add_sound(audio5)

        cap5_anims = captions.set_caption(
            "Through iterative updates, the boundary converges to zero error. Stacking these simple units creates deep learning."
        )

        conv_header = Text("Zero Error: Full Convergence Achieved!", font_size=24, color="#55efc4", weight=BOLD)
        conv_header.move_to([0, ScreenLayout.HEADER_Y, 0])

        # Clean Success Badge replacing the upper rule badge (ZERO collision with axes)
        success_badge = RoundedRectangle(
            corner_radius=0.15, height=0.55, width=5.2,
            fill_color="#006266", fill_opacity=0.9, stroke_color="#00b894", stroke_width=1.5
        ).move_to([0, 2.4, 0])
        success_text = Text("✓ All Points Correctly Separated", font_size=16, color=WHITE, weight=BOLD).move_to(success_badge)
        success_group = VGroup(success_badge, success_text)

        self.play(
            *cap5_anims,
            Transform(header, conv_header),
            FadeOut(rule_group),
            FadeOut(boundary_group),
            FadeIn(success_group, scale=0.9),
            run_time=1.2
        )
        self.wait(1.0)

        # Transition to Deep Learning Network Architecture in Stage Center:
        # Fade out axes and points to present the Multi-Layer Network cleanly!
        self.play(
            FadeOut(axes),
            FadeOut(axis_labels),
            FadeOut(blue_dots),
            FadeOut(orange_dots),
            FadeOut(blue_legend),
            FadeOut(orange_legend),
            FadeOut(decision_line),
            FadeOut(misclass_pt),
            FadeOut(success_group),
            run_time=1.0
        )

        # Draw a sleek Multi-Layer Perceptron (Input Layer -> Hidden Layer -> Output)
        # Inputs (2)
        in1 = Circle(radius=0.35, color="#0984e3", fill_color="#0984e3", fill_opacity=0.3).move_to([-3.0, 0.8, 0])
        in2 = Circle(radius=0.35, color="#0984e3", fill_color="#0984e3", fill_opacity=0.3).move_to([-3.0, -0.8, 0])
        in_labels = VGroup(
            Text("x1", font_size=16, color=WHITE).move_to(in1),
            Text("x2", font_size=16, color=WHITE).move_to(in2),
            Text("Input Layer", font_size=14, color="#74b9ff").next_to(VGroup(in1, in2), UP, buff=0.3)
        )

        # Hidden Layer (3 neurons)
        h1 = Circle(radius=0.35, color="#6c5ce7", fill_color="#6c5ce7", fill_opacity=0.3).move_to([0.0, 1.2, 0])
        h2 = Circle(radius=0.35, color="#6c5ce7", fill_color="#6c5ce7", fill_opacity=0.3).move_to([0.0, 0.0, 0])
        h3 = Circle(radius=0.35, color="#6c5ce7", fill_color="#6c5ce7", fill_opacity=0.3).move_to([0.0, -1.2, 0])
        h_labels = VGroup(
            Text("h1", font_size=16, color=WHITE).move_to(h1),
            Text("h2", font_size=16, color=WHITE).move_to(h2),
            Text("h3", font_size=16, color=WHITE).move_to(h3),
            Text("Hidden Layer (Non-Linear)", font_size=14, color="#a29bfe").next_to(h1, UP, buff=0.3)
        )

        # Output Layer (1 neuron)
        out1 = Circle(radius=0.35, color="#00cec9", fill_color="#00cec9", fill_opacity=0.3).move_to([3.0, 0.0, 0])
        out_lbl = VGroup(
            Text("y", font_size=16, color=WHITE).move_to(out1),
            Text("Output Layer", font_size=14, color="#81ecec").next_to(out1, UP, buff=0.3)
        )

        # Interconnecting Synapses
        synapses = VGroup()
        for src in [in1, in2]:
            for dst in [h1, h2, h3]:
                synapses.add(Line(src.get_right(), dst.get_left(), color="#57606f", stroke_width=1.5))
        for src in [h1, h2, h3]:
            synapses.add(Line(src.get_right(), out1.get_left(), color="#57606f", stroke_width=1.5))

        mlp_network = VGroup(synapses, in1, in2, in_labels, h1, h2, h3, h_labels, out1, out_lbl)

        # Final Big Concept Card positioned directly under header
        summary_card = RoundedRectangle(
            corner_radius=0.15, height=0.6, width=8.5,
            fill_color="#1e272e", fill_opacity=0.9, stroke_color="#6c5ce7", stroke_width=1.5
        ).move_to([0, 2.3, 0])
        summary_text = Text("Stacking Perceptrons = The Foundation of All Deep Learning", font_size=16, color="#f5f6fa", weight=BOLD).move_to(summary_card)
        summary_group = VGroup(summary_card, summary_text)

        self.play(
            FadeIn(summary_group, shift=DOWN * 0.2),
            FadeIn(mlp_network, scale=0.9),
            run_time=1.5
        )
        self.wait(max(0.5, dur5 - 3.5))

        # Final Outro fade
        self.play(*captions.clear(), run_time=0.8)
        self.wait(0.5)
