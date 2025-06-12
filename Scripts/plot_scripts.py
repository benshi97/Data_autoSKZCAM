#!/usr/bin/env python
# coding: utf-8

# Script to set up the plotting environment for Jupyter notebooks
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


def texfalse_import():
    plt.rcParams.update(
        {
            "font.family": "serif",  # use serif/main font for text elements
            "font.size": 9,
            "text.usetex": False,  # use inline math for ticks
        }
    )

def textrue_import():
    import matplotlib as mpl
    mpl.use("pgf")
    plt.rcParams.update({
        "font.family": "serif",  # use serif/main font for text elements
        "font.size": 9,
        "text.usetex": True,     # use inline math for ticks
        "pgf.rcfonts": False,    # don't setup fonts from rc parameters
        "pgf.preamble":"\n".join([
            r"\usepackage{amsmath}",            # load additional packages
            r"\usepackage{amssymb}",   # unicode math setup
            r"\usepackage[mathrm=sym]{unicode-math}",  # serif font via preamble
            r"\setmathfont[BoldFont={FiraSans-SemiBold.otf}]{FiraMath-Regular.otf}",
            r"\setmainfont[BoldFont={FiraSans-SemiBold.otf}]{FiraSans-Regular.otf}"
        ])
    })

# Dictionary of colors for the color scheme in our plots
color_dict = {
    "red": "#e6194b",
    "green": "#3cb44b",
    "yellow": "#ffe119",
    "blue": "#4363d8",
    "orange": "#f58231",
    "purple": "#911eb4",
    "cyan": "#42d4f4",
    "magenta": "#f032e6",
    "lime": "#bfef45",
    "pink": "#fabed4",
    "teal": "#469990",
    "lavendar": "#dcbeff",
    "brown": "#9A6324",
    "beige": "#fffac8",
    "maroon": "#800000",
    "mint": "#aaffc3",
    "olive": "#808000",
    "apricot": "#ffd8b1",
    "navy": "#000075",
    "grey": "#a9a9a9",
    "white": "#ffffff",
    "black": "#000000",
}

# Colors to cycle through for our plots
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=[
        "#4363d8",
        "#e6194B",
        "#3cb44b",
        "#f58231",
        "#ffe119",
        "#911eb4",
        "#42d4f4",
        "#f032e6",
        "#bfef45",
        "#fabed4",
        "#469990",
        "#dcbeff",
        "#9A6324",
        "#fffac8",
        "#800000",
        "#aaffc3",
        "#808000",
        "#ffd8b1",
        "#000075",
        "#a9a9a9",
        "#ffffff",
        "#000000",
    ]
)
