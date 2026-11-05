"""
NeuraMind Reusable Caption & Layout System for Manim Videos
Provides structured screen zones and synchronized subtitle rendering.
"""

from manim import *


class ScreenLayout:
    """
    Defines safe visual zones on the 16:9 Manim canvas (14.22 x 8.0 units).
    Prevents spatial collisions between headers, visualizations, and subtitles.
    """
    HEADER_Y = 3.3
    STAGE_CENTER = ORIGIN + DOWN * 0.1
    CAPTION_Y = -3.45

    # Safe boundaries for the main stage area
    STAGE_TOP = 2.7
    STAGE_BOTTOM = -2.8
    STAGE_LEFT = -6.5
    STAGE_RIGHT = 6.5


class CaptionManager:
    """
    Manages real-time, non-intrusive subtitle pills at the bottom of the screen.
    """
    def __init__(self, scene: Scene):
        self.scene = scene
        self.current_caption = None

        # Dedicated persistent bottom caption container
        self.bg_box = RoundedRectangle(
            corner_radius=0.18,
            height=0.82,
            width=13.0,
            fill_color="#14171f",
            fill_opacity=0.88,
            stroke_color="#384358",
            stroke_width=1.5
        ).move_to([0, ScreenLayout.CAPTION_Y, 0])

        self.box_active = False

    def set_caption(self, text_str: str, run_time: float = 0.5):
        """
        Updates the caption text smoothly.
        """
        new_text = Text(
            text_str,
            font_size=18,
            color="#f1f2f6",
            line_spacing=1.1
        ).move_to(self.bg_box.get_center())

        # Keep caption within the box width
        if new_text.width > 12.2:
            new_text.scale_to_fit_width(12.2)

        if not self.box_active:
            self.scene.add(self.bg_box)
            self.box_active = True
            self.current_caption = new_text
            return FadeIn(self.bg_box), FadeIn(self.current_caption)

        if self.current_caption is None:
            self.current_caption = new_text
            return (FadeIn(self.current_caption),)

        anim = Transform(self.current_caption, new_text)
        return (anim,)

    def clear(self):
        """Fade out captions when scene ends."""
        anims = []
        if self.current_caption is not None:
            anims.append(FadeOut(self.current_caption))
            self.current_caption = None
        if self.box_active:
            anims.append(FadeOut(self.bg_box))
            self.box_active = False
        return anims

