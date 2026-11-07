"""
Production Screen Layout and Subtitle/Caption System for NeuraMind Masterclass Videos.
Enforces strict 3-tier vertical zoning to eliminate visual overlap:
  1. Header Zone: y = 3.3 (Titles, Act badges)
  2. Stage Canvas: y in [-2.3, 2.7] (Diagrams, Code, Geometry, Graphs)
  3. Bottom Caption Bar: y = -3.45 (Synchronized Speech Subtitles)
"""

from manim import *

class MasterLayout:
    HEADER_Y = 3.3
    STAGE_TOP = 2.6
    STAGE_BOTTOM = -2.3
    CAPTION_Y = -3.42
    
    # Color Palette (Dark High-Tech NeuraMind Theme)
    BG_DARK = "#0B0F19"
    CYAN_ACCENT = "#00F0FF"
    ORANGE_ACCENT = "#F59E0B"
    GREEN_ACCENT = "#10B981"
    PURPLE_ACCENT = "#8B5CF6"
    RED_ACCENT = "#EF4444"
    TEXT_WHITE = "#F8FAFC"
    MUTED_GRAY = "#94A3B8"
    BOX_BG = "#1E293B"
    BOX_BORDER = "#334155"

class MasterCaption(VGroup):
    """
    High-contrast subtitle bar anchored at the bottom zone.
    Features rounded pill background, neon status pip, and crisp readable subtitles.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bg_box = RoundedRectangle(
            corner_radius=0.18,
            width=13.2,
            height=0.9,
            fill_color=MasterLayout.BG_DARK,
            fill_opacity=0.92,
            stroke_color=MasterLayout.CYAN_ACCENT,
            stroke_width=1.5
        )
        self.status_dot = Dot(
            point=[-6.1, 0, 0],
            radius=0.08,
            color=MasterLayout.GREEN_ACCENT
        )
        self.caption_text = Text(
            "Initializing NeuraMind Deep Learning Masterclass...",
            font_size=19,
            color=MasterLayout.TEXT_WHITE,
            weight=MEDIUM
        )
        self.caption_text.move_to([0.2, 0, 0])
        
        self.add(self.bg_box, self.status_dot, self.caption_text)
        self.move_to([0, MasterLayout.CAPTION_Y, 0])
        
    def update_text(self, new_text: str, act_num: int = None):
        if act_num is not None:
            full_str = f"[Act {act_num}]  {new_text}"
        else:
            full_str = new_text
            
        new_caption = Text(
            full_str,
            font_size=18,
            color=MasterLayout.TEXT_WHITE,
            weight=MEDIUM
        )
        new_caption.move_to([0.2, MasterLayout.CAPTION_Y, 0])
        if new_caption.width > 12.0:
            new_caption.scale(12.0 / new_caption.width)
            
        return Transform(self.caption_text, new_caption)

def create_header(title_str: str, act_num: int, act_title: str) -> VGroup:
    """
    Creates a standardized, non-overlapping header banner for any act.
    """
    badge = RoundedRectangle(
        corner_radius=0.1,
        width=1.8,
        height=0.5,
        fill_color="#1E1B4B",
        fill_opacity=0.9,
        stroke_color="#6366F1",
        stroke_width=1.5
    )
    badge_txt = Text(f"ACT 0{act_num}", font_size=14, color="#A5B4FC", weight=BOLD)
    badge_group = VGroup(badge, badge_txt)
    
    title = Text(act_title, font_size=24, color="#F8FAFC", weight=BOLD)
    
    header = VGroup(badge_group, title).arrange(RIGHT, buff=0.35)
    header.move_to([0, MasterLayout.HEADER_Y, 0])
    return header

