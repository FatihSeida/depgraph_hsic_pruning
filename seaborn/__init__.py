"""Minimal :mod:`seaborn` stub for unit tests.

It re-exports only a small ``heatmap`` function wrapper. For real projects
install the genuine ``seaborn`` package.
"""

import matplotlib.pyplot as plt

def heatmap(data, *a, **k):
    cmap = k.get('cmap')
    plt.imshow(data, cmap=cmap)
