import types
from pipeline.model_reconfig import AdaptiveLayerReconfiguration

class DummyTensor:
    def __init__(self, device="cpu"):
        self.device = device
    def to(self, device):
        self.device = device
        return self

class DummyConv2d:
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, padding_mode="zeros",
                 groups=1, device="cpu"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.groups = groups
        self.weight = DummyTensor(device)
        self.bias = DummyTensor(device) if bias else None
        self._device = device
    def to(self, device):
        self._device = device
        self.weight.to(device)
        if self.bias is not None:
            self.bias.to(device)
        return self
    @property
    def device(self):
        return self._device

class DummyModule:
    def __init__(self, conv):
        self.conv = conv
    def named_modules(self):
        yield "", self
        yield "conv", self.conv
    def get_submodule(self, name):
        return getattr(self, name)
    def to(self, device):
        self.conv.to(device)
        return self

class DummyInnerModel:
    def __init__(self, modules, device="cpu"):
        self.model = modules
        self._device = device
    def parameters(self):
        return iter([types.SimpleNamespace(device=self._device)])
    def to(self, device):
        self._device = device
        for m in self.model:
            m.to(device)
        return self

class DummyYOLO:
    def __init__(self, modules, device="cpu"):
        self.model = DummyInnerModel(modules, device)
        self.to_calls = []
    def to(self, device):
        self.to_calls.append(device)
        self.model.to(device)
        return self
    def save(self, path):
        pass

def build_yolo(device="cuda"):
    modules = [
        DummyModule(DummyConv2d(3, 4, device=device)),
        DummyModule(DummyConv2d(4, 5, device=device)),
        DummyModule(DummyConv2d(7, 6, device=device)),  # mismatch
    ]
    for _ in range(3, 10):
        modules.append(DummyModule(DummyConv2d(6, 6, device=device)))
    modules.append(DummyModule(DummyConv2d(7, 6, device=device)))  # head mismatch
    return DummyYOLO(modules, device)

def test_reconfig_preserves_device(monkeypatch):
    import pipeline.model_reconfig as mr

    dummy_nn = types.SimpleNamespace(
        Conv2d=DummyConv2d,
        init=types.SimpleNamespace(
            kaiming_normal_=lambda *a, **k: None,
            zeros_=lambda *a, **k: None,
        ),
    )
    monkeypatch.setattr(mr, "nn", dummy_nn)

    yolo = build_yolo("cuda")
    cfg = AdaptiveLayerReconfiguration()
    cfg.reconfigure_model(yolo)

    assert yolo.to_calls[-1] == "cuda"
    assert yolo.model.model[2].conv.device == "cuda"
    assert yolo.model.model[10].conv.device == "cuda"
