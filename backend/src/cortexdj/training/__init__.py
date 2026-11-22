"""Training code for the quadrant classifier: folds, the metric, the loop and the ``train-classifier`` command.

Imports only the ML kernel and the platform kernel, and nothing in the app imports it (an
import-linter contract), so the inference kernel stays free of training code.
"""
