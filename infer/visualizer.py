#!/usr/bin/env python3

from PyQt5.QtWidgets import QLabel, QWidget, QApplication
from infer.sliding_window_inference_engine import SlidingWindowInferenceEngine


# Program which demonstrates the effectiveness or ineffectiveness of a lane detection model
# by displaying an image and highlighting the areas in which it predicts there are road lines.
# Created by brendon-ai, September 2017


# Main PyQt5 QWidget class
class Visualizer(QWidget):

    # Call various initialization functions
    def __init__(self):

        super(Visualizer, self).__init__()

        self.init_ui()

    def init_ui(self):
        pass
