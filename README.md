# depgraph_hsic_pruning

This repository contains utilities based on the Ultralytics YOLO stack. A pruning
pipeline is provided for orchestrating model preparation, pruning and
fine-tuning.

## Using `PruningPipeline`

Below is a minimal example that prunes a segmentation model pretrained on COCO.

```python
from pipeline import PruningPipeline

pipeline = PruningPipeline("yolov8n-seg.pt", data="coco8.yaml")
pipeline.load_model()
pipeline.calc_initial_stats()
pipeline.pretrain(epochs=1)
pipeline.analyze_structure()
pipeline.generate_pruning_mask(ratio=0.2)
pipeline.apply_pruning()
pipeline.calc_pruned_stats()
pipeline.finetune(epochs=3)
print(pipeline.record_metrics())
```

The example relies on the `ultralytics_pruning.YOLO` class for model loading and
training. Adjust the dataset and hyperparameters as needed for your
experiments.
