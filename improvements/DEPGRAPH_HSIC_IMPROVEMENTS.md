# Perbaikan DepGraph-HSIC Implementation

## Masalah Sebelumnya

Implementasi DepGraph-HSIC sebelumnya memiliki beberapa masalah:

1. **Fallback ke synthetic data**: Ketika tidak ada dataloader, menggunakan data sintetis yang menghasilkan HSIC scores = 0
2. **Fallback ke L1-norm**: Ketika HSIC gagal, otomatis fallback ke L1-norm pruning
3. **Dependency graph tidak optimal**: Hanya menganalisis 10 layer pertama dari model
4. **Error handling tidak jelas**: Tidak memberikan error yang jelas ketika pruning gagal
5. **Tidak ada kontrol scope pruning**: Selalu menggunakan semua layer atau hanya backbone

## Perbaikan yang Dibuat

### 1. **Menghilangkan Fallback ke Synthetic Data**

**Sebelum:**
```python
# Menggunakan synthetic labels jika tidak ada real labels
if len(self.labels) < 2:
    synthetic_labels = torch.randn(activations.size(0), 1)
```

**Sesudah:**
```python
if dataloader is None:
    raise ValueError("Dataloader is required for HSIC computation. No fallback to synthetic data allowed.")
```

### 2. **Menghilangkan Fallback ke L1-norm**

**Sebelum:**
```python
try:
    self._hsic_lasso_plan(ratio)
except Exception as e:
    self.logger.error("HSIC-Lasso pruning failed: %s", str(e))
    # Fallback ke L1-norm
```

**Sesudah:**
```python
# Apply HSIC-Lasso pruning
self._apply_hsic_lasso_pruning(pruning_groups, group_scores, ratio)

if pruned_count == 0:
    raise RuntimeError("No pruning was applied. Check dependency graph structure and pruning ratio.")
```

### 3. **Kontrol Scope Pruning yang Fleksibel**

**Fitur Baru:**
```python
# Menggunakan argumen --pruning-scope
python main.py --model yolov8n.pt --data data.yaml --pruning-scope backbone  # Hanya 10 layer pertama
python main.py --model yolov8n.pt --data data.yaml --pruning-scope full      # Semua layer
```

**Implementasi:**
```python
def __init__(self, model: Any, workdir: str = "runs/pruning", gamma: float = 1.0, 
             num_modules: int = 10, pruning_scope: str = "backbone") -> None:
    self.pruning_scope = pruning_scope
    # ...

def register_hooks(self) -> None:
    if self.pruning_scope == "backbone":
        # Gunakan utilitas yang ada untuk backbone-only
        from .utils import collect_backbone_convs
        backbone_convs = collect_backbone_convs(self.model, self.num_modules)
        # ...
    else:
        # Register hooks untuk semua layer Conv2d
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # ...
```

### 4. **Pengumpulan Aktivasi yang Lebih Robust**

**Fitur Baru:**
- Batasan jumlah maksimum samples (`max_samples`)
- Progress logging setiap 10 samples
- Error handling yang lebih baik
- Validasi input yang lebih ketat

### 5. **Validasi dan Monitoring**

**Method Baru:**
- `validate_pruning()`: Memvalidasi bahwa pruning berhasil dan model masih berfungsi
- `get_pruning_summary()`: Memberikan ringkasan lengkap operasi pruning

## Cara Penggunaan

### 1. **Basic Usage dengan Scope Control**

```python
from prune_methods.depgraph_hsic import DepgraphHSICMethod

# Backbone-only pruning (10 layer pertama)
pruner = DepgraphHSICMethod(model, workdir="runs/pruning", pruning_scope="backbone")

# Full model pruning (semua layer)
pruner = DepgraphHSICMethod(model, workdir="runs/pruning", pruning_scope="full")

# Generate pruning mask (akan raise error jika gagal)
pruner.generate_pruning_mask(ratio=0.3, dataloader=valid_dataloader)

# Apply pruning
pruner.apply_pruning()

# Validate hasil
validation = pruner.validate_pruning()
print(f"Model functional: {validation['model_functional']}")
print(f"Pruning ratio achieved: {validation['pruning_ratio_achieved']:.3f}")
```

### 2. **Command Line Usage**

```bash
# Menggunakan semua method yang tersedia (default)
python main.py --model yolov8n-seg.pt --data biotech_model_train.yaml

# Backbone-only pruning dengan DepGraph-HSIC
python main.py --model yolov8n-seg.pt --data biotech_model_train.yaml --methods depgraph_hsic --pruning-scope backbone

# Full model pruning dengan DepGraph-HSIC
python main.py --model yolov8n-seg.pt --data biotech_model_train.yaml --methods depgraph_hsic --pruning-scope full

# Multiple methods dengan scope yang sama
python main.py --model yolov8n-seg.pt --data biotech_model_train.yaml --methods depgraph_hsic l1 random --pruning-scope backbone

# Menggunakan method tertentu saja
python main.py --model yolov8n-seg.pt --data biotech_model_train.yaml --methods l1 random depgraph_hsic
```

### 3. **Error Handling**

```python
try:
    pruner.generate_pruning_mask(ratio=0.3, dataloader=dataloader)
except ValueError as e:
    print(f"Input error: {e}")
except RuntimeError as e:
    print(f"Pruning failed: {e}")
```

### 4. **Monitoring Progress**

```python
# Log akan menampilkan progress dan scope
# INFO - Registered hooks for 27 Conv2d layers (scope: full)
# INFO - Collecting activations using real data for HSIC
# DEBUG - Collected 10 samples
# INFO - Found 15 pruning groups
# INFO - HSIC-Lasso pruning mask generated successfully
```

## Keuntungan Perbaikan

1. **Konsistensi**: Tidak ada fallback yang tidak diinginkan
2. **Transparansi**: Error yang jelas ketika pruning gagal
3. **Robustness**: Pengumpulan data yang lebih reliable
4. **Monitoring**: Validasi dan ringkasan hasil pruning
5. **Flexibility**: Kontrol scope pruning (backbone vs full)
6. **Integration**: Menggunakan utilitas yang sudah ada (`collect_backbone_convs`)
7. **Comprehensive**: Default menggunakan semua method yang tersedia

## Troubleshooting

### Error: "Dataloader is required for HSIC computation"
**Solusi:** Pastikan memberikan dataloader yang valid dengan data real

### Error: "No pruning groups found in dependency graph"
**Solusi:** Periksa struktur model dan pastikan dependency graph dibangun dengan benar

### Error: "No pruning was applied"
**Solusi:** Periksa pruning ratio dan struktur dependency groups

### Warning: "No channels selected for pruning"
**Solusi:** Coba dengan pruning ratio yang lebih tinggi

### Perbedaan Scope Pruning
**Backbone (default):** Hanya 10 layer pertama, lebih cepat, fokus pada feature extraction
**Full:** Semua layer, lebih lambat, pruning komprehensif

### Method yang Tersedia
Saat ini tersedia 8 method pruning:
- `l1`: L1NormMethod
- `random`: RandomMethod  
- `depgraph`: DepgraphMethod
- `tp_random`: TorchRandomMethod
- `isomorphic`: IsomorphicMethod
- `hsic_lasso`: HSICLassoMethod
- `whc`: WeightedHybridMethod
- `depgraph_hsic`: DepgraphHSICMethod