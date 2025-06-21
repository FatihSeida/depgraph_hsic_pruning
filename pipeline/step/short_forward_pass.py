from __future__ import annotations

from pathlib import Path

from ..context import PipelineContext
from . import PipelineStep


class ShortForwardPassStep(PipelineStep):
    """Run short forward passes on a couple of validation images."""

    def run(self, context: PipelineContext) -> None:  # pragma: no cover - heavy deps
        step = self.__class__.__name__
        context.logger.info("Starting %s", step)
        if context.model is None:
            raise ValueError("Model is not loaded")
        if context.pruning_method is None:
            raise NotImplementedError
        try:
            import torch
            import yaml
            from PIL import Image  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover - optional deps missing
            context.logger.warning("short forward pass skipped: %s", exc)
            context.logger.info("Finished %s", step)
            return

        with open(context.data) as f:
            ds_cfg = yaml.safe_load(f)
        base = Path(ds_cfg.get("path", Path(context.data).parent))
        val = ds_cfg.get("val") or ds_cfg.get("train")
        val_path = base / val if val is not None else base

        imgs: list[Path] = []
        if val_path.is_dir():
            imgs = sorted(p for p in val_path.glob("*.*") if p.is_file())
        elif val_path.exists():
            imgs = [val_path]

        if not imgs:
            context.logger.warning("no validation image found for short forward pass")
            context.logger.info("Finished %s", step)
            return

        imgs = imgs[:2]
        try:
            context.pruning_method.register_hooks()
        except Exception:
            pass

        device = next(context.model.model.parameters()).device

        for img in imgs:
            context.logger.info("short forward pass image: %s", img)
            label_file = Path(str(img)).with_suffix(".txt").as_posix()
            label_file = label_file.replace("/images/", "/labels/")
            context.logger.info("short forward pass label file: %s", label_file)
            y = torch.tensor([])
            lf = Path(label_file)
            if lf.exists():
                with lf.open() as f:
                    labels = [float(line.split()[0]) for line in f if line.strip()]
                if labels:
                    y = torch.tensor([labels[0]])
                    context.logger.info("label file %s has %d entries", label_file, len(labels))
                else:
                    context.logger.warning("label file %s is empty", label_file)
            else:
                context.logger.warning("label file %s does not exist", label_file)

            if y.numel() > 0:
                try:
                    img_pil = Image.open(img).convert("RGB")
                    orig_size = img_pil.size
                    if orig_size != (640, 640):
                        context.logger.debug(
                            "resizing short forward pass image from %s to (640, 640)",
                            orig_size,
                        )
                        img_pil = img_pil.resize((640, 640))
                    arr = np.array(img_pil, dtype=np.float32)
                    arr = np.transpose(arr, (2, 0, 1))
                    inp = torch.tensor(arr).unsqueeze(0)
                    context.logger.debug("short forward pass tensor shape: %s", tuple(inp.shape))
                except Exception:  # pragma: no cover - fallback
                    inp = getattr(
                        context.pruning_method,
                        "example_inputs",
                        torch.randn(1, 3, 640, 640),
                    )
                with torch.no_grad():
                    context.model.model(inp.to(device))
                context.pruning_method.add_labels(y)
        context.logger.info("Finished %s", step)


__all__ = ["ShortForwardPassStep"]
