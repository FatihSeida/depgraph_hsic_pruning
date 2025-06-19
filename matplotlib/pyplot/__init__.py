"""Simplified :mod:`matplotlib.pyplot` replacement for testing.

Only provides thin stubs of a few functions. Install the full ``matplotlib``
package for real plotting capabilities.
"""

import os

def figure(*a, **k):
    return None

class _Axes:
    def bar(self, *a, **k):
        pass
    def set_title(self, *a, **k):
        pass
    def set_xlabel(self, *a, **k):
        pass
    def set_ylabel(self, *a, **k):
        pass

def subplots(*a, **k):
    return None, (_Axes(), _Axes())

def plot(*a, **k):
    pass

def imshow(*a, **k):
    pass

def xlabel(*a, **k):
    pass

def ylabel(*a, **k):
    pass

def tight_layout(*a, **k):
    pass

def savefig(path, *a, **k):
    if not isinstance(path, str):
        try:
            path = str(path)
        except Exception:
            return
    with open(path, 'wb') as f:
        f.write(b'')


def close(*a, **k):
    pass
