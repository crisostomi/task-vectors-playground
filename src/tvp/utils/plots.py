import json

import matplotlib.pyplot as plt
import numpy as np


class Palette(dict):
    def __init__(self, path):

        with open(path, "r") as f:
            color_palette = json.load(f)

        mapping = {
            "Bittersweet shimmer": "light red",
            "Persian green": "green",
            "Saffron": "yellow",
            "Charcoal": "dark blue",
            "Burgundy": "red",
            "Burnt sienna": "orange",
            "Eggplant": "violet",
            "Sandy brown": "light orange",
        }

        for color, hex_code in color_palette.items():
            self[mapping[color]] = hex_code

    def get_colors(self, n):
        return list(self.values())[:n]
