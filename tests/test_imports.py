import prune_methods

class DummyYOLO:
    def __init__(self):
        self.model = []

def test_all_prune_methods_instantiable(tmp_path):
    yolo = DummyYOLO()
    for name in prune_methods.__all__:
        if name == "BasePruningMethod":
            continue
        cls = getattr(prune_methods, name)
        instance = cls(yolo, tmp_path)
        assert isinstance(instance, cls)
