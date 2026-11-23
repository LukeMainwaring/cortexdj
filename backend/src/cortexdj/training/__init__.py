"""Training code for the quadrant classifier: folds, the metric, the loop, the ``train-classifier`` command and its Modal twin.

It also keeps the cross-subject baseline script, evidence for the per-participant design that
is not part of the training path. Imports only the ML kernel and the platform kernel, and
nothing in the app imports it (an import-linter contract), so the inference kernel stays free
of training code.
"""
