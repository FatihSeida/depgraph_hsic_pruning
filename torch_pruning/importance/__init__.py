"""Simplified importance metrics used for testing.

These classes only serve as placeholders. Install ``torch-pruning`` to access
the full set of importance measures in real applications.
"""


class RandomImportance:
    pass

class MagnitudeImportance:
    def __init__(self, p=1):
        self.p = p
