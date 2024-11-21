import json

import matplotlib.pyplot as plt
import numpy as np


class Palette(dict):
    def __init__(self, palette_path, map_path):
        with open(palette_path, "r") as f:
            color_palette = json.load(f)

        with open(map_path, "r") as f:
            mapping = json.load(f)

        for color, hex_code in color_palette.items():
            self[mapping[color]] = hex_code

    def get_colors(self, n):
        return list(self.values())[:n]
