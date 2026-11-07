"""
NeuraMind Deep Learning Masterclass: In-Depth Perceptron Video (7 Acts)
Complete Production Scene with:
  - Student Placement Dataset (IQ & CGPA) grounded examples
  - Matrix dot product flow & step-by-step numerical calculation (z = 5.2 -> Placed)
  - Big, prominent, razor-sharp Computer Modern LaTeX equations
  - Slope-intercept geometric derivation with normal vector
  - 3-case unified update rule & Perceptron Loss SGD derivation
  - Live training simulation resolving errors (4 -> 2 -> 1 -> 0)
  - Line-by-line syntax-highlighted Python NumPy code walkthrough
  - Generalization decision mesh, convergence loss curve & XOR failure bridge to MLPs
  - Synchronized with calm Kokoro Michael narration (speed=0.88, 332.72s)
"""

import os
import json
from manim import *

from visualisations.latex_engine import create_latex_mobject
from visualisations.caption_system import MasterLayout, MasterCaption, create_header

MANIFEST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "audio", "manifest.json"))

def load_manifest_durations():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {item["act"]: item["duration"] for item in data}
    return {1: 38.6, 2: 61.6, 3: 48.3, 4: 49.3, 5: 38.3, 6: 41.7, 7: 49.3}

class PerceptronMasterclassScene(Scene):
    def construct(self):
        self.camera.background_color = MasterLayout.BG_DARK
        durations = load_manifest_durations()
        
        # Bottom subtitle/caption bar
        caption_bar = MasterCaption()
        self.add(caption_bar)
        
        # =====================================================================
        # ACT 1: The Concrete Problem: Student Placement & Feature Space
        # =====================================================================
        act1_dur = durations.get(1, 38.64)
        self.play(caption_bar.update_text("Task: Predict Placement (y in {0,1}) from IQ (x1) and CGPA (x2) in 2D Feature Space", 1))
        
        header_1 = create_header("NeuraMind DL Workshop", 1, "The Placement Problem & Feature Space")
        self.play(FadeIn(header_1, shift=DOWN * 0.3), run_time=1.0)
        
        # Left Panel: Clean Data Table Card
        table_box = RoundedRectangle(
            corner_radius=0.18, width=5.8, height=4.4,
            fill_color=MasterLayout.BOX_BG, fill_opacity=0.85,
            stroke_color=MasterLayout.BOX_BORDER, stroke_width=1.5
        ).move_to([-3.4, 0.05, 0])
        
        table_title = Text("STUDENT PLACEMENT DATASET", font_size=14, color=MasterLayout.CYAN_ACCENT, weight=BOLD).move_to([-3.4, 1.95, 0])
        col_headers = Text("Student   |   IQ (x1)   |   CGPA (x2)   |   Placed (y)", font_size=12, color=MasterLayout.ORANGE_ACCENT, weight=BOLD).move_to([-3.4, 1.45, 0])
        
        row1 = Text("Student 1    |     78      |      7.8      |    1  (Placed)", font_size=12, color=MasterLayout.CYAN_ACCENT).move_to([-3.4, 0.95, 0])
        row2 = Text("Student 2    |     69      |      5.1      |    0  (Not Placed)", font_size=12, color=MasterLayout.ORANGE_ACCENT).move_to([-3.4, 0.45, 0])
        row3 = Text("Student 3    |     88      |      8.5      |    1  (Placed)", font_size=12, color=MasterLayout.CYAN_ACCENT).move_to([-3.4, -0.05, 0])
        row4 = Text("Student 4    |     55      |      4.5      |    0  (Not Placed)", font_size=12, color=MasterLayout.ORANGE_ACCENT).move_to([-3.4, -0.55, 0])
        
        data_note = Text("Binary Target: y in {0, 1} | Continuous Inputs: x1, x2", font_size=11, color=MasterLayout.MUTED_GRAY).move_to([-3.4, -1.35, 0])
        
        left_table = Group(table_box, table_title, col_headers, row1, row2, row3, row4, data_note)
        
        # Right Panel: 2D Feature Space Plot
        ax1 = Axes(
            x_range=[40, 100, 20],
            y_range=[3, 10, 2],
            x_length=5.2,
            y_length=4.2,
            axis_config={"color": MasterLayout.BOX_BORDER, "stroke_width": 2},
            tips=True
        ).move_to([3.4, 0.05, 0])
        lbl_x1 = Text("IQ (x1)", font_size=12, color=MasterLayout.MUTED_GRAY).next_to(ax1.x_axis.get_end(), DOWN, buff=0.1)
        lbl_x2 = Text("CGPA (x2)", font_size=12, color=MasterLayout.MUTED_GRAY).next_to(ax1.y_axis.get_end(), LEFT, buff=0.1)
        
        # Placed students (Cyan)
        pt_p1 = Dot(ax1.c2p(78, 7.8), radius=0.13, color=MasterLayout.CYAN_ACCENT)
        pt_p2 = Dot(ax1.c2p(88, 8.5), radius=0.13, color=MasterLayout.CYAN_ACCENT)
        pt_p3 = Dot(ax1.c2p(82, 8.0), radius=0.13, color=MasterLayout.CYAN_ACCENT)
        pt_p4 = Dot(ax1.c2p(92, 7.2), radius=0.13, color=MasterLayout.CYAN_ACCENT)
        
        # Not Placed students (Orange)
        pt_n1 = Dot(ax1.c2p(69, 5.1), radius=0.13, color=MasterLayout.ORANGE_ACCENT)
        pt_n2 = Dot(ax1.c2p(55, 4.5), radius=0.13, color=MasterLayout.ORANGE_ACCENT)
        pt_n3 = Dot(ax1.c2p(60, 5.5), radius=0.13, color=MasterLayout.ORANGE_ACCENT)
        pt_n4 = Dot(ax1.c2p(65, 4.2), radius=0.13, color=MasterLayout.ORANGE_ACCENT)
        
        pts_placed = VGroup(pt_p1, pt_p2, pt_p3, pt_p4)
        pts_not_placed = VGroup(pt_n1, pt_n2, pt_n3, pt_n4)
        
        legend_p = Text("● Placed (y = 1)", font_size=11, color=MasterLayout.CYAN_ACCENT, weight=BOLD).move_to([3.8, 1.8, 0])
        legend_n = Text("● Not Placed (y = 0)", font_size=11, color=MasterLayout.ORANGE_ACCENT, weight=BOLD).move_to([3.8, 1.45, 0])
        
        challenge_txt = Text("Objective: Find Decision Line to Separate Classes!", font_size=12, color="#FBBF24", weight=BOLD).move_to([3.4, -1.8, 0])
        
        right_plot = Group(ax1, lbl_x1, lbl_x2, pts_placed, pts_not_placed, legend_p, legend_n, challenge_txt)
        
        self.play(FadeIn(left_table, shift=RIGHT * 0.4), run_time=2.0)
        self.play(FadeIn(right_plot, shift=LEFT * 0.4), run_time=2.0)
        
        remaining_act1 = max(0.5, act1_dur - 5.5)
        self.wait(remaining_act1)
        
        self.play(FadeOut(header_1), FadeOut(left_table), FadeOut(right_plot), run_time=1.0)
        
        # =====================================================================
        # ACT 2: Artificial Neuron Architecture & Matrix Dot Product Flow
        # =====================================================================
        act2_dur = durations.get(2, 61.64)
        self.play(caption_bar.update_text("Augmented Matrix Flow: z = w~^T x~ = b + w1*x1 + w2*x2 -> Step(z) -> y_hat", 2))
        
        header_2 = create_header("NeuraMind DL Workshop", 2, "Neuron Architecture & Matrix Dot Product Flow")
        self.play(FadeIn(header_2, shift=DOWN * 0.3), run_time=1.0)
        
        # Left Canvas: Neural Computational Graph
        neuron_group = VGroup()
        
        in_x1 = Circle(radius=0.38, color=MasterLayout.CYAN_ACCENT, fill_color=MasterLayout.BOX_BG, fill_opacity=1.0).move_to([-5.2, 1.3, 0])
        lbl_x1 = Text("x1", font_size=17, color=MasterLayout.CYAN_ACCENT).move_to(in_x1.get_center())
        lbl_x1_sub = Text("IQ", font_size=10, color=MasterLayout.MUTED_GRAY).next_to(in_x1, LEFT, buff=0.1)
        
        in_x2 = Circle(radius=0.38, color=MasterLayout.CYAN_ACCENT, fill_color=MasterLayout.BOX_BG, fill_opacity=1.0).move_to([-5.2, 0.05, 0])
        lbl_x2 = Text("x2", font_size=17, color=MasterLayout.CYAN_ACCENT).move_to(in_x2.get_center())
        lbl_x2_sub = Text("CGPA", font_size=10, color=MasterLayout.MUTED_GRAY).next_to(in_x2, LEFT, buff=0.1)
        
        in_b = Circle(radius=0.38, color=MasterLayout.ORANGE_ACCENT, fill_color=MasterLayout.BOX_BG, fill_opacity=1.0).move_to([-5.2, -1.2, 0])
        lbl_b = Text("+1", font_size=17, color=MasterLayout.ORANGE_ACCENT).move_to(in_b.get_center())
        lbl_b_sub = Text("Bias", font_size=10, color=MasterLayout.MUTED_GRAY).next_to(in_b, LEFT, buff=0.1)
        
        soma = Circle(radius=0.85, color=MasterLayout.PURPLE_ACCENT, fill_color="#1E1B4B", fill_opacity=0.95).move_to([-2.6, 0.05, 0])
        soma_line = Line([-2.6, 0.9, 0], [-2.6, -0.8, 0], color=MasterLayout.BOX_BORDER, stroke_width=1.5)
        lbl_sum = Text("Σ", font_size=28, color="#A5B4FC").move_to([-3.0, 0.05, 0])
        lbl_act = Text("Step", font_size=18, color=MasterLayout.CYAN_ACCENT).move_to([-2.2, 0.05, 0])
        
        out_node = Circle(radius=0.38, color=MasterLayout.GREEN_ACCENT, fill_color=MasterLayout.BOX_BG, fill_opacity=1.0).move_to([-0.4, 0.05, 0])
        lbl_yhat = Text("y^", font_size=20, color=MasterLayout.GREEN_ACCENT).move_to(out_node.get_center())
        
        arr_1 = Arrow(in_x1.get_right(), soma.get_left() + UP * 0.4, buff=0.1, color=MasterLayout.MUTED_GRAY, stroke_width=2.5)
        lbl_w1 = Text("w1=1.0", font_size=12, color=MasterLayout.ORANGE_ACCENT).next_to(arr_1.get_center(), UP, buff=0.06)
        
        arr_2 = Arrow(in_x2.get_right(), soma.get_left(), buff=0.1, color=MasterLayout.MUTED_GRAY, stroke_width=2.5)
        lbl_w2 = Text("w2=2.0", font_size=12, color=MasterLayout.ORANGE_ACCENT).next_to(arr_2.get_center(), UP, buff=0.06)
        
        arr_b = Arrow(in_b.get_right(), soma.get_left() + DOWN * 0.4, buff=0.1, color=MasterLayout.MUTED_GRAY, stroke_width=2.5)
        lbl_wb = Text("b=-105", font_size=12, color=MasterLayout.ORANGE_ACCENT).next_to(arr_b.get_center(), DOWN, buff=0.06)
        
        arr_out = Arrow(soma.get_right(), out_node.get_left(), buff=0.1, color=MasterLayout.GREEN_ACCENT, stroke_width=3.0)
        
        neuron_group.add(
            in_x1, lbl_x1, lbl_x1_sub, in_x2, lbl_x2, lbl_x2_sub, in_b, lbl_b, lbl_b_sub,
            arr_1, lbl_w1, arr_2, lbl_w2, arr_b, lbl_wb,
            soma, soma_line, lbl_sum, lbl_act, arr_out, out_node, lbl_yhat
        )
        
        # Right Canvas: Big Prominent LaTeX & Numerical Walkthrough
        calc_box = RoundedRectangle(
            corner_radius=0.18, width=6.2, height=4.5,
            fill_color=MasterLayout.BOX_BG, fill_opacity=0.85,
            stroke_color=MasterLayout.BOX_BORDER, stroke_width=1.5
        ).move_to([3.4, 0.05, 0])
        
        calc_title = Text("MATRIX DOT PRODUCT & REAL CALCULATION", font_size=13, color=MasterLayout.CYAN_ACCENT, weight=BOLD).move_to([3.4, 2.0, 0])
        
        # Big LaTeX Formula: Linear Combination
        eq_dot = create_latex_mobject(
            r"z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^D w_j x_j + b",
            color="#FFFFFF", height=0.72
        ).move_to([3.4, 1.4, 0])
        
        # Augmented Representation Formula
        eq_aug = create_latex_mobject(
            r"\tilde{\mathbf{w}} = [b, w_1, w_2]^T, \ \tilde{\mathbf{x}} = [1, x_1, x_2]^T \Rightarrow z = \tilde{\mathbf{w}}^T \tilde{\mathbf{x}}",
            color="#10B981", height=0.68
        ).move_to([3.4, 0.65, 0])
        
        # Concrete Numerical Walkthrough Box
        num_header = Text("Test Applicant: IQ = 100, CGPA = 5.1", font_size=12, color=MasterLayout.ORANGE_ACCENT, weight=BOLD).move_to([3.4, 0.0, 0])
        
        eq_calc = create_latex_mobject(
            r"z = (-105.0) + (1.0 \times 100) + (2.0 \times 5.1) = 5.2",
            color="#00F0FF", height=0.65
        ).move_to([3.4, -0.5, 0])
        
        eq_step = create_latex_mobject(
            r"z = 5.2 \geq 0 \Rightarrow \hat{y} = \mathrm{Step}(5.2) = 1 \quad (\mathrm{Placed!})",
            color="#A7F3D0", height=0.65
        ).move_to([3.4, -1.2, 0])
        
        importance_badge = Text("Feature Importance: w_CGPA (2.0) > w_IQ (1.0) -> CGPA has 2x higher impact!", font_size=10, color="#FDE047").move_to([3.4, -1.8, 0])
        
        right_calc = Group(calc_box, calc_title, eq_dot, eq_aug, num_header, eq_calc, eq_step, importance_badge)
        
        self.play(FadeIn(neuron_group, shift=RIGHT * 0.4), run_time=2.0)
        self.play(FadeIn(right_calc, shift=LEFT * 0.4), run_time=2.0)
        
        self.play(soma.animate.scale(1.08).set_color(MasterLayout.CYAN_ACCENT), run_time=1.2)
        self.play(soma.animate.scale(1/1.08).set_color(MasterLayout.PURPLE_ACCENT), run_time=1.0)
        
        remaining_act2 = max(0.5, act2_dur - 7.5)
        self.wait(remaining_act2)
        
        self.play(FadeOut(header_2), FadeOut(neuron_group), FadeOut(right_calc), run_time=1.0)
        
        # =====================================================================
        # ACT 3: Geometric Decision Boundary & Slope-Intercept Form
        # =====================================================================
        act3_dur = durations.get(3, 48.33)
        self.play(caption_bar.update_text("Boundary: w1*x1 + w2*x2 + b = 0 => x2 = -(w1/w2)*x1 - (b/w2) | Normal w points to z >= 0", 3))
        
        header_3 = create_header("NeuraMind DL Workshop", 3, "Geometric Decision Boundary & Slope-Intercept Form")
        self.play(FadeIn(header_3, shift=DOWN * 0.3), run_time=1.0)
        
        # Left Canvas: 2D Axes
        ax3 = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=5.2,
            y_length=4.2,
            axis_config={"color": MasterLayout.BOX_BORDER, "stroke_width": 2},
            tips=True
        ).move_to([-3.4, 0.05, 0])
        lbl_ax_x = Text("IQ (x1)", font_size=12, color=MasterLayout.MUTED_GRAY).next_to(ax3.x_axis.get_end(), DOWN, buff=0.1)
        lbl_ax_y = Text("CGPA (x2)", font_size=12, color=MasterLayout.MUTED_GRAY).next_to(ax3.y_axis.get_end(), LEFT, buff=0.1)
        
        line3 = ax3.plot(lambda x: -x, color=MasterLayout.CYAN_ACCENT, stroke_width=3.5)
        
        pos_half = Text("Placed Region (z >= 0)\ny^ = 1", font_size=12, color=MasterLayout.CYAN_ACCENT, weight=BOLD).move_to(ax3.c2p(1.5, 1.4))
        neg_half = Text("Not Placed (z < 0)\ny^ = 0", font_size=12, color=MasterLayout.ORANGE_ACCENT, weight=BOLD).move_to(ax3.c2p(-1.5, -1.4))
        
        w_vec = Arrow(ax3.c2p(0, 0), ax3.c2p(1.0, 1.0), buff=0, color=MasterLayout.GREEN_ACCENT, stroke_width=3.0)
        lbl_w_vec = Text("w (Normal)", font_size=12, color=MasterLayout.GREEN_ACCENT).next_to(w_vec.get_end(), RIGHT, buff=0.08)
        
        left_geom = Group(ax3, lbl_ax_x, lbl_ax_y, line3, pos_half, neg_half, w_vec, lbl_w_vec)
        
        # Right Canvas: Big Prominent Slope-Intercept Derivation
        geom_box = RoundedRectangle(
            corner_radius=0.18, width=6.2, height=4.5,
            fill_color=MasterLayout.BOX_BG, fill_opacity=0.85,
            stroke_color=MasterLayout.BOX_BORDER, stroke_width=1.5
        ).move_to([3.4, 0.05, 0])
        
        geom_title = Text("SLOPE-INTERCEPT DERIVATION", font_size=14, color=MasterLayout.CYAN_ACCENT, weight=BOLD).move_to([3.4, 2.0, 0])
        
        eq_b1 = create_latex_mobject(r"w_1 x_1 + w_2 x_2 + b = 0", color="#FFFFFF", height=0.7).move_to([3.4, 1.35, 0])
        eq_b2 = create_latex_mobject(r"w_2 x_2 = -w_1 x_1 - b", color="#A5B4FC", height=0.68).move_to([3.4, 0.65, 0])
        eq_b3 = create_latex_mobject(r"x_2 = -\left(\frac{w_1}{w_2}\right) x_1 - \left(\frac{b}{w_2}\right)", color="#00F0FF", height=0.9).move_to([3.4, -0.25, 0])
        
        param_desc = VGroup(
            Text("• Slope (m) = -(w1 / w2) : Controls Angle & Orientation", font_size=12, color=MasterLayout.ORANGE_ACCENT),
            Text("• Intercept (c) = -(b / w2) : Shifts Line Parallel Across Space", font_size=12, color=MasterLayout.GREEN_ACCENT),
            Text("• Normal Vector w : Points Orthogonally into Placed Half-Space", font_size=12, color=MasterLayout.TEXT_WHITE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([3.4, -1.45, 0])
        
        right_geom = Group(geom_box, geom_title, eq_b1, eq_b2, eq_b3, param_desc)
        
        self.play(FadeIn(left_geom), FadeIn(right_geom), run_time=2.2)
        
        # Dynamic parameter shift: Translate line (bias changing)
        line_shift = ax3.plot(lambda x: -x + 0.9, color=MasterLayout.GREEN_ACCENT, stroke_width=3.5)
        self.play(Transform(line3, line_shift), run_time=2.5)
        
        # Dynamic parameter shift: Rotate line (weights changing)
        line_rotate = ax3.plot(lambda x: -0.4 * x + 0.4, color=MasterLayout.CYAN_ACCENT, stroke_width=3.5)
        self.play(Transform(line3, line_rotate), run_time=2.5)
        
        remaining_act3 = max(0.5, act3_dur - 8.2)
        self.wait(remaining_act3)
        
        self.play(FadeOut(header_3), FadeOut(left_geom), FadeOut(right_geom), run_time=1.0)
        
        # =====================================================================
        # ACT 4: Learning Mechanics, Error Discrepancy & Vector Updates
        # =====================================================================
        act4_dur = durations.get(4, 49.33)
        self.play(caption_bar.update_text("Unified Rule: w_new = w_old + eta*(y - y_hat)*x | SGD on Loss L = max(0, -y*f(x))", 4))
        
        header_4 = create_header("NeuraMind DL Workshop", 4, "Learning Mechanics & Unified Update Rule")
        self.play(FadeIn(header_4, shift=DOWN * 0.3), run_time=1.0)
        
        # Top Panel: Big Prominent Parameter Equations
        top_box = RoundedRectangle(
            corner_radius=0.18, width=12.4, height=1.5,
            fill_color=MasterLayout.BOX_BG, fill_opacity=0.85,
            stroke_color="#6366F1", stroke_width=1.5
        ).move_to([0, 1.8, 0])
        top_title = Text("THE UNIFIED PARAMETER UPDATE EQUATIONS", font_size=13, color=MasterLayout.CYAN_ACCENT, weight=BOLD).move_to([0, 2.3, 0])
        
        eq_w_up = create_latex_mobject(r"\mathbf{w}_{\mathrm{new}} = \mathbf{w}_{\mathrm{old}} + \eta (y_i - \hat{y}_i) \mathbf{x}_i", color="#FFFFFF", height=0.75).move_to([-3.2, 1.6, 0])
        eq_b_up = create_latex_mobject(r"b_{\mathrm{new}} = b_{\mathrm{old}} + \eta (y_i - \hat{y}_i)", color="#00F0FF", height=0.75).move_to([3.2, 1.6, 0])
        top_panel = Group(top_box, top_title, eq_w_up, eq_b_up)
        
        # Middle Panel: 3-Case Error Discrepancy Cards
        cases_group = Group()
        c_specs = [
            ("CASE 1: CORRECT", "y_i == y^", "(y - y^) = 0", "No Parameter Change", MasterLayout.GREEN_ACCENT, -4.1),
            ("CASE 2: FALSE NEGATIVE", "y_i = 1, y^ = 0", "(1 - 0) = +1", "+ eta * x_i (Pulls Line to Point)", MasterLayout.CYAN_ACCENT, 0.0),
            ("CASE 3: FALSE POSITIVE", "y_i = 0, y^ = 1", "(0 - 1) = -1", "- eta * x_i (Pushes Line Away)", MasterLayout.RED_ACCENT, 4.1)
        ]
        for title, cond, diff, action, col, x_pos in c_specs:
            cb = RoundedRectangle(corner_radius=0.12, width=3.8, height=1.65, fill_color="#111827", fill_opacity=0.9, stroke_color=col, stroke_width=1.5).move_to([x_pos, 0.3, 0])
            ct = Text(title, font_size=11, color=col, weight=BOLD).move_to([x_pos, 0.9, 0])
            cc = Text(cond, font_size=13, color=MasterLayout.TEXT_WHITE).move_to([x_pos, 0.58, 0])
            cd = Text(f"Error: {diff}", font_size=12, color=MasterLayout.ORANGE_ACCENT).move_to([x_pos, 0.28, 0])
            ca = Text(action, font_size=11, color=col, weight=BOLD).move_to([x_pos, -0.05, 0])
            cases_group.add(cb, ct, cc, cd, ca)
            
        # Bottom Panel: Perceptron Loss & Stochastic Gradient Descent
        bot_box = RoundedRectangle(
            corner_radius=0.18, width=12.4, height=1.5,
            fill_color=MasterLayout.BOX_BG, fill_opacity=0.85,
            stroke_color=MasterLayout.BOX_BORDER, stroke_width=1.5
        ).move_to([0, -1.35, 0])
        bot_title = Text("FORMAL PERCEPTRON LOSS & STOCHASTIC GRADIENT DESCENT", font_size=13, color=MasterLayout.ORANGE_ACCENT, weight=BOLD).move_to([0, -0.85, 0])
        
        eq_loss_f = create_latex_mobject(r"L(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N \max(0, -y_i (\mathbf{w}^T \mathbf{x}_i + b))", color="#F8FAFC", height=0.72).move_to([-3.2, -1.5, 0])
        eq_grad_f = create_latex_mobject(r"\nabla_{\mathbf{w}} L_i = -y_i \mathbf{x}_i \Rightarrow \mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i", color="#10B981", height=0.72).move_to([3.2, -1.5, 0])
        bot_panel = Group(bot_box, bot_title, eq_loss_f, eq_grad_f)
        
        self.play(FadeIn(top_panel, shift=DOWN * 0.3), run_time=1.8)
        self.play(FadeIn(cases_group, shift=UP * 0.3), run_time=2.2)
        self.play(FadeIn(bot_panel, shift=UP * 0.3), run_time=1.8)
        
        remaining_act4 = max(0.5, act4_dur - 6.8)
        self.wait(remaining_act4)
        
        self.play(FadeOut(header_4), FadeOut(top_panel), FadeOut(cases_group), FadeOut(bot_panel), run_time=1.0)
        
        # =====================================================================
        # ACT 5: Real-Time Training Simulation & Parameter Evolution
        # =====================================================================
        act5_dur = durations.get(5, 38.25)
        self.play(caption_bar.update_text("Simulation: Live parameter updates resolve errors 4 -> 2 -> 1 -> 0 (Convergence)", 5))
        
        header_5 = create_header("NeuraMind DL Workshop", 5, "Live Training Simulation & Convergence")
        self.play(FadeIn(header_5, shift=DOWN * 0.3), run_time=1.0)
        
        sim_axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=5.6,
            y_length=4.2,
            axis_config={"color": MasterLayout.BOX_BORDER, "stroke_width": 2},
            tips=False
        ).move_to([-3.1, 0.05, 0])
        
        neg_pts_coords = [(-1.8, -1.2), (-1.2, -1.8), (-0.8, -0.6), (-1.6, -0.2)]
        pos_pts_coords = [(1.2, 1.5), (1.8, 0.9), (0.6, 1.7), (1.4, 0.3)]
        
        neg_dots = VGroup(*[Dot(sim_axes.c2p(x, y), radius=0.12, color=MasterLayout.ORANGE_ACCENT) for x, y in neg_pts_coords])
        pos_dots = VGroup(*[Dot(sim_axes.c2p(x, y), radius=0.12, color=MasterLayout.CYAN_ACCENT) for x, y in pos_pts_coords])
        
        # Right Monitor Card
        monitor_box = RoundedRectangle(
            corner_radius=0.18, width=5.0, height=4.2,
            fill_color=MasterLayout.BOX_BG, fill_opacity=0.85,
            stroke_color=MasterLayout.BOX_BORDER, stroke_width=1.5
        ).move_to([3.8, 0.05, 0])
        
        monitor_title = Text("TRAINING TELEMETRY", font_size=14, color=MasterLayout.CYAN_ACCENT, weight=BOLD).move_to([3.8, 1.85, 0])
        iter_txt = Text("Iteration: 0 / 3", font_size=13, color=MasterLayout.TEXT_WHITE).move_to([3.8, 1.25, 0])
        error_badge = RoundedRectangle(corner_radius=0.1, width=3.4, height=0.5, fill_color="#7F1D1D", stroke_color=MasterLayout.RED_ACCENT, stroke_width=1.5).move_to([3.8, 0.65, 0])
        error_txt = Text("Errors: 4 Misclassified", font_size=12, color="#FECACA", weight=BOLD).move_to(error_badge.get_center())
        weights_txt = Text("Weights: [w1=0.2, w2=-0.9, b=0.4]", font_size=11, color=MasterLayout.MUTED_GRAY).move_to([3.8, 0.0, 0])
        status_txt = Text("Status: Initial Untrained Line", font_size=12, color=MasterLayout.ORANGE_ACCENT, weight=BOLD).move_to([3.8, -0.6, 0])
        
        monitor_group = Group(monitor_box, monitor_title, iter_txt, error_badge, error_txt, weights_txt, status_txt)
        
        sim_line = sim_axes.plot(lambda x: 0.6 * x + 0.9, color=MasterLayout.RED_ACCENT, stroke_width=3.5)
        
        self.play(Create(sim_axes), FadeIn(neg_dots), FadeIn(pos_dots), Create(sim_line), FadeIn(monitor_group), run_time=2.0)
        
        # Step 1: Error at (0.6, 1.7)
        self.wait(2.5)
        target_pt1 = pos_dots[2]
        pulse1 = Circle(radius=0.28, color=MasterLayout.RED_ACCENT, stroke_width=3).move_to(target_pt1.get_center())
        self.play(Create(pulse1), run_time=0.8)
        
        sim_line_1 = sim_axes.plot(lambda x: -0.2 * x + 0.4, color=MasterLayout.ORANGE_ACCENT, stroke_width=3.5)
        error_txt_1 = Text("Errors: 2 Misclassified", font_size=12, color="#FECACA", weight=BOLD).move_to(error_badge.get_center())
        iter_txt_1 = Text("Iteration: 1 / 3", font_size=13, color=MasterLayout.TEXT_WHITE).move_to([3.8, 1.25, 0])
        weights_txt_1 = Text("Weights: [w1=0.5, w2=0.3, b=0.2]", font_size=11, color=MasterLayout.MUTED_GRAY).move_to([3.8, 0.0, 0])
        
        self.play(
            Transform(sim_line, sim_line_1),
            Transform(error_txt, error_txt_1),
            Transform(iter_txt, iter_txt_1),
            Transform(weights_txt, weights_txt_1),
            FadeOut(pulse1),
            run_time=2.0
        )
        
        # Step 2: Error at (-0.8, -0.6)
        self.wait(2.5)
        target_pt2 = neg_dots[2]
        pulse2 = Circle(radius=0.28, color=MasterLayout.RED_ACCENT, stroke_width=3).move_to(target_pt2.get_center())
        self.play(Create(pulse2), run_time=0.8)
        
        sim_line_2 = sim_axes.plot(lambda x: -0.7 * x + 0.1, color=MasterLayout.CYAN_ACCENT, stroke_width=3.5)
        error_txt_2 = Text("Errors: 1 Misclassified", font_size=12, color="#FECACA", weight=BOLD).move_to(error_badge.get_center())
        iter_txt_2 = Text("Iteration: 2 / 3", font_size=13, color=MasterLayout.TEXT_WHITE).move_to([3.8, 1.25, 0])
        weights_txt_2 = Text("Weights: [w1=0.9, w2=1.1, b=-0.1]", font_size=11, color=MasterLayout.MUTED_GRAY).move_to([3.8, 0.0, 0])
        
        self.play(
            Transform(sim_line, sim_line_2),
            Transform(error_txt, error_txt_2),
            Transform(iter_txt, iter_txt_2),
            Transform(weights_txt, weights_txt_2),
            FadeOut(pulse2),
            run_time=2.0
        )
        
        # Step 3: Convergence
        self.wait(2.5)
        sim_line_final = sim_axes.plot(lambda x: -1.0 * x + 0.0, color=MasterLayout.GREEN_ACCENT, stroke_width=4.0)
        error_badge_final = RoundedRectangle(corner_radius=0.1, width=3.4, height=0.5, fill_color="#064E3B", stroke_color=MasterLayout.GREEN_ACCENT, stroke_width=1.5).move_to([3.8, 0.65, 0])
        error_txt_final = Text("✓ 0 Errors (Convergence!)", font_size=12, color="#A7F3D0", weight=BOLD).move_to(error_badge_final.get_center())
        iter_txt_final = Text("Iteration: 3 (Done)", font_size=13, color=MasterLayout.TEXT_WHITE).move_to([3.8, 1.25, 0])
        status_txt_final = Text("Status: Perfectly Separated", font_size=12, color=MasterLayout.GREEN_ACCENT, weight=BOLD).move_to([3.8, -0.6, 0])
        
        self.play(
            Transform(sim_line, sim_line_final),
            Transform(error_badge, error_badge_final),
            Transform(error_txt, error_txt_final),
            Transform(iter_txt, iter_txt_final),
            Transform(status_txt, status_txt_final),
            run_time=2.2
        )
        
        remaining_act5 = max(0.5, act5_dur - 15.0)
        self.wait(remaining_act5)
        
        self.play(FadeOut(header_5), FadeOut(sim_axes), FadeOut(neg_dots), FadeOut(pos_dots), FadeOut(sim_line), FadeOut(monitor_group), run_time=1.0)
        
        # =====================================================================
        # ACT 6: Line-by-Line Python NumPy Implementation
        # =====================================================================
        act6_dur = durations.get(6, 41.67)
        self.play(caption_bar.update_text("Code: np.insert(bias) -> dot product z -> Step threshold -> Unified update -> Batch predict", 6))
        
        header_6 = create_header("NeuraMind DL Workshop", 6, "Line-by-Line Python NumPy Implementation")
        self.play(FadeIn(header_6, shift=DOWN * 0.3), run_time=1.0)
        
        py_code_str = (
            "class Perceptron:\n"
            "    def __init__(self, lr=0.1, epochs=100):\n"
            "        self.lr = lr\n"
            "        self.epochs = epochs\n"
            "    def fit(self, X, y):\n"
            "        X_aug = np.insert(X, 0, 1, axis=1)\n"
            "        self.w = np.zeros(X_aug.shape[1])\n"
            "        for epoch in range(self.epochs):\n"
            "            for i in range(len(X_aug)):\n"
            "                z = np.dot(self.w, X_aug[i])\n"
            "                y_hat = 1 if z >= 0 else 0\n"
            "                self.w += self.lr * (y[i] - y_hat) * X_aug[i]\n"
            "    def predict(self, X):\n"
            "        X_aug = np.insert(X, 0, 1, axis=1)\n"
            "        return np.where(np.dot(X_aug, self.w) >= 0, 1, 0)"
        )
        
        code_mob = Code(
            code_string=py_code_str,
            language="python",
            formatter_style="monokai",
            add_line_numbers=True,
            line_numbers_from=1,
            background="rectangle",
            background_config={"fill_color": "#0F172A", "fill_opacity": 0.95, "stroke_color": "#38BDF8", "stroke_width": 1.5}
        ).scale(0.68).move_to([-2.6, 0.05, 0])
        
        explain_box = RoundedRectangle(
            corner_radius=0.18, width=5.2, height=4.4,
            fill_color=MasterLayout.BOX_BG, fill_opacity=0.85,
            stroke_color=MasterLayout.BOX_BORDER, stroke_width=1.5
        ).move_to([4.0, 0.05, 0])
        
        explain_header = Text("LINE-BY-LINE EXECUTION", font_size=13, color=MasterLayout.CYAN_ACCENT, weight=BOLD).move_to([4.0, 1.95, 0])
        step_badge = RoundedRectangle(corner_radius=0.08, width=1.6, height=0.35, fill_color="#312E81", stroke_color="#818CF8", stroke_width=1).move_to([4.0, 1.45, 0])
        step_txt = Text("STEP 1 / 4", font_size=11, color="#C7D2FE", weight=BOLD).move_to(step_badge.get_center())
        
        desc_title = Text("1. Hyperparameters & Bias", font_size=14, color=MasterLayout.ORANGE_ACCENT, weight=BOLD).move_to([4.0, 1.0, 0])
        desc_body = Text(
            "• lr (eta): controls step displacement.\n"
            "• np.insert(..., 1, axis=1):\n"
            "  Prepends 1s column for bias b\n"
            "• w initialized to zeros (shape D+1)",
            font_size=12, color=MasterLayout.TEXT_WHITE, line_spacing=1.3
        ).move_to([4.0, 0.1, 0])
        
        right_explain = Group(explain_box, explain_header, step_badge, step_txt, desc_title, desc_body)
        
        self.play(FadeIn(code_mob, shift=RIGHT * 0.4), FadeIn(right_explain, shift=LEFT * 0.4), run_time=2.0)
        
        # Step 2
        self.wait(6.0)
        desc_title_2 = Text("2. Forward Pass & Activation", font_size=14, color=MasterLayout.CYAN_ACCENT, weight=BOLD).move_to([4.0, 1.0, 0])
        desc_body_2 = Text(
            "• z = np.dot(w, X_aug[i]):\n"
            "  Computes scalar dot product.\n"
            "• y_hat = 1 if z >= 0 else 0:\n"
            "  Heaviside threshold activation.",
            font_size=12, color=MasterLayout.TEXT_WHITE, line_spacing=1.3
        ).move_to([4.0, 0.1, 0])
        step_txt_2 = Text("STEP 2 / 4", font_size=11, color="#C7D2FE", weight=BOLD).move_to(step_badge.get_center())
        self.play(Transform(desc_title, desc_title_2), Transform(desc_body, desc_body_2), Transform(step_txt, step_txt_2), run_time=1.5)
        
        # Step 3
        self.wait(6.0)
        desc_title_3 = Text("3. Unified Weight Update", font_size=14, color=MasterLayout.GREEN_ACCENT, weight=BOLD).move_to([4.0, 1.0, 0])
        desc_body_3 = Text(
            "• error = (y[i] - y_hat)\n"
            "• w += lr * error * X_aug[i]\n"
            "  Zero error -> No update\n"
            "  Misclassified -> Pulls/Pushes line",
            font_size=12, color=MasterLayout.TEXT_WHITE, line_spacing=1.3
        ).move_to([4.0, 0.05, 0])
        step_txt_3 = Text("STEP 3 / 4", font_size=11, color="#C7D2FE", weight=BOLD).move_to(step_badge.get_center())
        self.play(Transform(desc_title, desc_title_3), Transform(desc_body, desc_body_3), Transform(step_txt, step_txt_3), run_time=1.5)
        
        # Step 4
        self.wait(6.0)
        desc_title_4 = Text("4. Vectorized Predict Method", font_size=14, color="#F43F5E", weight=BOLD).move_to([4.0, 1.0, 0])
        desc_body_4 = Text(
            "• np.dot(X_aug, self.w):\n"
            "  Fast batched matrix multiplication.\n"
            "• np.where(z >= 0, 1, 0):\n"
            "  Returns discrete predictions for\n"
            "  all test samples instantly.",
            font_size=12, color=MasterLayout.TEXT_WHITE, line_spacing=1.3
        ).move_to([4.0, 0.05, 0])
        step_txt_4 = Text("STEP 4 / 4", font_size=11, color="#C7D2FE", weight=BOLD).move_to(step_badge.get_center())
        self.play(Transform(desc_title, desc_title_4), Transform(desc_body, desc_body_4), Transform(step_txt, step_txt_4), run_time=1.5)
        
        remaining_act6 = max(0.5, act6_dur - 25.5)
        self.wait(remaining_act6)
        
        self.play(FadeOut(header_6), FadeOut(code_mob), FadeOut(right_explain), run_time=1.0)
        
        # =====================================================================
        # ACT 7: Convergence Graph, Generalization & The XOR Failure
        # =====================================================================
        act7_dur = durations.get(7, 49.26)
        self.play(caption_bar.update_text("Convergence to 0 Loss | Generalization Mesh | Stacking Perceptrons Solves Non-Linear XOR", 7))
        
        header_7 = create_header("NeuraMind DL Workshop", 7, "Convergence, Generalization & The XOR Failure")
        self.play(FadeIn(header_7, shift=DOWN * 0.3), run_time=1.0)
        
        ax_gen = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=5.0,
            y_length=3.8,
            axis_config={"color": MasterLayout.BOX_BORDER, "stroke_width": 2},
            tips=False
        ).move_to([-3.4, 0.15, 0])
        
        gen_line = ax_gen.plot(lambda x: -x, color=MasterLayout.GREEN_ACCENT, stroke_width=3.0)
        
        poly_pos = Polygon(
            ax_gen.c2p(-3, 3), ax_gen.c2p(3, -3), ax_gen.c2p(3, 2.5), ax_gen.c2p(-3, 2.5),
            fill_color=MasterLayout.CYAN_ACCENT, fill_opacity=0.15, stroke_width=0
        )
        poly_neg = Polygon(
            ax_gen.c2p(-3, 3), ax_gen.c2p(3, -3), ax_gen.c2p(3, -2.5), ax_gen.c2p(-3, -2.5),
            fill_color=MasterLayout.ORANGE_ACCENT, fill_opacity=0.15, stroke_width=0
        )
        
        lbl_mesh_pos = Text("y^ = 1 (Placed Region)", font_size=11, color=MasterLayout.CYAN_ACCENT, weight=BOLD).move_to(ax_gen.c2p(1.2, 1.8))
        lbl_mesh_neg = Text("y^ = 0 (Not Placed)", font_size=11, color=MasterLayout.ORANGE_ACCENT, weight=BOLD).move_to(ax_gen.c2p(-1.2, -1.8))
        
        test_dot_1 = Dot(ax_gen.c2p(2.2, 1.0), radius=0.12, color=MasterLayout.CYAN_ACCENT)
        test_dot_2 = Dot(ax_gen.c2p(-2.0, -1.0), radius=0.12, color=MasterLayout.ORANGE_ACCENT)
        test_lbl = Text("New Applicant Evaluated!", font_size=11, color=MasterLayout.TEXT_WHITE).next_to(test_dot_1, UP, buff=0.1)
        
        left_gen_panel = Group(ax_gen, poly_pos, poly_neg, gen_line, lbl_mesh_pos, lbl_mesh_neg, test_dot_1, test_dot_2, test_lbl)
        
        ax_loss = Axes(
            x_range=[0, 15, 3],
            y_range=[0, 5, 1],
            x_length=5.0,
            y_length=3.8,
            axis_config={"color": MasterLayout.BOX_BORDER, "stroke_width": 2},
            tips=True
        ).move_to([3.4, 0.15, 0])
        
        lbl_loss_x = Text("Training Epochs", font_size=12, color=MasterLayout.MUTED_GRAY).next_to(ax_loss.x_axis.get_end(), DOWN, buff=0.1)
        lbl_loss_y = Text("Misclassifications", font_size=12, color=MasterLayout.MUTED_GRAY).next_to(ax_loss.y_axis.get_end(), UP, buff=0.1)
        
        loss_curve = ax_loss.plot_line_graph(
            x_values=[0, 1, 2, 4, 6, 8, 10, 12, 15],
            y_values=[4, 3, 2, 2, 1, 1, 0, 0, 0],
            line_color="#F43F5E",
            vertex_dot_radius=0.06,
            vertex_dot_style={"color": MasterLayout.CYAN_ACCENT}
        )
        
        zero_badge = RoundedRectangle(corner_radius=0.08, width=2.4, height=0.45, fill_color="#064E3B", stroke_color=MasterLayout.GREEN_ACCENT, stroke_width=1.5).move_to(ax_loss.c2p(11, 0.8))
        zero_txt = Text("Zero Loss (Optimal)", font_size=11, color="#A7F3D0", weight=BOLD).move_to(zero_badge.get_center())
        
        right_loss_panel = Group(ax_loss, lbl_loss_x, lbl_loss_y, loss_curve, zero_badge, zero_txt)
        
        self.play(FadeIn(left_gen_panel), FadeIn(right_loss_panel), run_time=2.2)
        
        # Bottom Bridge Banner
        self.wait(6.0)
        bridge_banner = RoundedRectangle(
            corner_radius=0.18, width=12.4, height=1.15,
            fill_color="#1E1B4B", fill_opacity=0.95,
            stroke_color="#818CF8", stroke_width=1.8
        ).move_to([0, -1.85, 0])
        
        bridge_title = Text("THE BRIDGE TO DEEP LEARNING: MULTI-LAYER PERCEPTRONS", font_size=13, color=MasterLayout.CYAN_ACCENT, weight=BOLD).move_to([0, -1.55, 0])
        bridge_desc = Text(
            "Single Perceptron: Linear Boundary (Fails on XOR)  -->  Multi-Layer Perceptrons: Stacks neurons to solve ANY non-linear function!",
            font_size=11, color=MasterLayout.TEXT_WHITE
        ).move_to([0, -1.95, 0])
        
        bridge_group = Group(bridge_banner, bridge_title, bridge_desc)
        self.play(FadeIn(bridge_group, shift=UP * 0.3), run_time=1.8)
        
        remaining_act7 = max(0.5, act7_dur - 11.5)
        self.wait(remaining_act7)
        
        self.play(FadeOut(header_7), FadeOut(left_gen_panel), FadeOut(right_loss_panel), FadeOut(bridge_group), FadeOut(caption_bar), run_time=1.5)
