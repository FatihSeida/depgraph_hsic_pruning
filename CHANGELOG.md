# Changelog

## [Unreleased] - 2024-12-19

### Added
- **Synthetic Data Collection**: Implementasi synthetic data generation untuk activation collection yang lebih efisien
- **Kondisional Pipeline Selection**: Fungsi `create_pipeline()` untuk memilih pipeline berdasarkan tipe metode pruning
- **Common Functionality**: Konsolidasi fungsi umum ke `BasePruningPipeline`

### Changed
- **Pipeline Architecture**: 
  - `PruningPipeline` untuk metode non-DepGraph (menggunakan AdaptiveReconfigurator)
  - `PruningPipeline2` untuk metode DepGraph (tidak memerlukan rekonfigurasi) dan kini mendukung semua metode berbasis DepGraph
- **Activation Collection**: Mengganti `ShortForwardPassStep` dengan synthetic data collection
- **Memory Efficiency**: Menghilangkan penggunaan dataloader untuk activation collection

### Removed
- **ShortForwardPassStep**: Dihapus karena diganti dengan synthetic data collection
- **Redundant Code**: Menghapus kode yang tumpang tindih antara kedua pipeline

### Fixed
- **Memory Issues**: Mengatasi masalah memori GPU dengan synthetic data
- **Pipeline Selection**: Memperbaiki logika pemilihan pipeline berdasarkan tipe metode

## Metode Pruning yang Menggunakan DepGraph:
- `DepgraphHSICMethod`
- `DepgraphMethod`
- `IsomorphicMethod`
- `TorchRandomMethod`

## Metode Pruning Non-DepGraph:
- `L1NormMethod`
- `RandomMethod`
- `HSICLassoMethod`
- `WeightedHybridMethod`
