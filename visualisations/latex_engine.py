"""
High-Resolution LaTeX Equation Renderer for Manim without external TeX dependencies.
Uses Matplotlib's native Computer Modern MathText engine to render pure LaTeX math into
transparent high-DPI ImageMobjects.
"""

import os
import hashlib
import tempfile

# Set non-conflicting cache dir for matplotlib BEFORE importing it
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "manim_mpl_config")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
import matplotlib.pyplot as plt
from manim import ImageMobject

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'serif'

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "latex_cache"))
os.makedirs(CACHE_DIR, exist_ok=True)

def render_latex_to_png(latex_str: str, color: str = "#FFFFFF", fontsize: int = 28, dpi: int = 300) -> str:
    """
    Renders a LaTeX math expression to a crisp, transparent PNG using Computer Modern font.
    Caches results by MD5 hash of parameters.
    """
    key = f"{latex_str}_{color}_{fontsize}_{dpi}"
    filename = hashlib.md5(key.encode("utf-8")).hexdigest() + ".png"
    filepath = os.path.join(CACHE_DIR, filename)
    
    if os.path.exists(filepath):
        return filepath
        
    # Ensure expression is enclosed in math markers if not already
    expr = latex_str.strip()
    if not expr.startswith("$"):
        expr = f"${expr}$"
        
    fig, ax = plt.subplots(figsize=(8, 2))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.axis("off")
    
    ax.text(0.5, 0.5, expr, fontsize=fontsize, color=color, ha="center", va="center")
    
    plt.savefig(
        filepath,
        format="png",
        transparent=True,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04
    )
    plt.close(fig)
    return filepath

def create_latex_mobject(latex_str: str, color: str = "#FFFFFF", height: float = 0.95, fontsize: int = 38) -> ImageMobject:
    """
    Creates a prominent, razor-sharp scaled Manim ImageMobject representing a LaTeX equation.
    Default height is 0.95 (prominent and easily legible at 720p).
    """
    png_path = render_latex_to_png(latex_str, color=color, fontsize=fontsize, dpi=300)
    mob = ImageMobject(png_path)
    if height is not None:
        mob.set_height(height)
    return mob

if __name__ == "__main__":
    # Self-test equations
    eqs = [
        r"z = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^D w_i x_i + b",
        r"\hat{y} = \mathrm{Step}(z) = \{ 1 \ \mathrm{if}\ z \geq 0, \ 0 \ \mathrm{if}\ z < 0 \}",
        r"\mathbf{w}_{\mathrm{new}} = \mathbf{w}_{\mathrm{old}} + \eta (y_i - \hat{y}_i) \mathbf{x}_i",
        r"b_{\mathrm{new}} = b_{\mathrm{old}} + \eta (y_i - \hat{y}_i)",
        r"L(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N \max(0, -y_i (\mathbf{w}^T \mathbf{x}_i + b))",
        r"w_1 x_1 + w_2 x_2 + b = 0 \Rightarrow x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}",
        r"\nabla_{\mathbf{w}} L_i = -y_i \mathbf{x}_i \Rightarrow \mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i"
    ]
    for eq in eqs:
        p = render_latex_to_png(eq, color="#00F0FF")
        print(f"Rendered: {eq[:35]}... -> {os.path.basename(p)}")
    print("All LaTeX formulas rendered successfully!")
