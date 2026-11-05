from manim import *

class TestScene(Scene):
    def construct(self):
        title = Text("NeuraMind Deep Learning", font_size=40, color=BLUE)
        self.play(FadeIn(title))
        self.wait(1)
        self.play(title.animate.scale(0.8).to_edge(UP))
        dot = Dot(color=YELLOW)
        self.play(Create(dot))
        self.wait(1)

