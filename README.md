# STRIDE

**Stride-level Tracking to Reveal Impairment before Dementia Emerges**

STRIDE is an automated pipeline for dual cognitive and motor phenotyping of mice in T-maze behavioral assays using [SLEAP](https://sleap.ai) pose estimation. It extracts 27 behavioral metrics from standard bottom-up T-maze video — up from 4 with manual scoring — without specialized hardware.

The key finding driving this work: **motor gait deficits are detectable before cognitive decline becomes apparent**, making stride-level gait analysis a potential early biomarker for Alzheimer's disease progression in mouse models.

---

## What STRIDE Does

```
Raw video (.mp4) + SLEAP models
    |
    v
[1] Camera calibration & distortion check
[2] SLEAP inference (15-keypoint pose + 7-region ROI segmentation)
[3] T-maze behavioral analysis (arm choice, snout probes, junction exploration)
[4] Keypoint preprocessing (interpolation, smoothing, velocity)
[5] Stride detection (swing phases, foot-strikes, stride segmentation)
[6] Stride filtering (confidence, edge removal, angular velocity)
[7] Trial phase tagging (task vs return locomotion)
[8] Derived metrics (step frequency, regional assignment)
[9] Linear mixed model analysis (genotype effects, FDR correction)
[10] WT vs Tau classification (logistic regression, ROC-AUC)
```

## Pipeline Overview

### Cognitive Phenotyping (T-maze)
- Automated arm entry detection with depth-based validation
- Snout probe counting (geometric snout-in-polygon detection)
- Junction exploration timing
- Learning curve analysis with reward reversal detection
- Choice latency and stem traversal metrics

### Motor Phenotyping (Gait)
- 14 stride-level gait metrics following [Sheppard et al. (2022)](https://doi.org/10.1016/j.celrep.2022.110231) and [Bellardita & Kiehn (2015)](https://doi.org/10.1016/j.cub.2015.04.005)
- Spatial: stride length, step length, step width
- Temporal: stride duration, step frequency, limb duty factor, temporal symmetry
- Body dynamics: lateral displacement of snout, tailbase, tailtip (normalized)
- Phase: timing of maximum lateral displacement within stride cycle
- Angular velocity for straight-line walking selection

### Combined Analysis
- Linear mixed models (M1: all effects, M2: speed-corrected direct effects)
- FDR-corrected significance testing across 14 metrics
- WT vs Tau logistic regression classifier combining cognitive + motor features
- Regional gait analysis (stem, junction, arm)

## Installation

```bash
# Clone
git clone https://github.com/LeoMeow123/stride.git
cd stride

# Install (requires Python >= 3.12)
pip install -e .

# Or with uv
uv pip install -e .
```

## Quick Start

### Full pipeline (config-driven)

```bash
# Edit config.yaml with your batch paths, then:
python run_all.py --config config.yaml
```

### Individual stages

```bash
# T-maze behavioral analysis
stride analyze-decisions --videos /path/to/videos --yml /path/to/roi_yml --meta meta.csv

# Gait extraction (stride detection)
python -m stride.stages.run_gait_extraction --config config.yaml

# LMM statistical analysis
python -m stride.stages.gait_lmm --config config.yaml

# Classification
python -m stride.stages.classification --config config.yaml
```

### Adding a new batch

1. Add a batch entry to `config.yaml` with video paths, model paths, and metadata
2. Run `python run_all.py --batch your_batch_name`

No code changes required — all batch-specific parameters are config-driven.

## Configuration

All parameters are centralized in `config.yaml`:

```yaml
batches:
  my_batch:
    name: "My Batch"
    video_dir: "/path/to/videos/"
    yml_dir: "/path/to/roi_yml/"
    meta_trials_csv: "/path/to/meta_trials.csv"
    metadata_csv: "/path/to/genotype_metadata.csv"
    fps: 120
    column_map:
      mouse_id: ["mouse"]
      sex: "Gender"

stride_detection:
  min_walk_velocity: 5.0      # cm/s
  min_paw_velocity: 8.0       # cm/s
  min_peak_paw_velocity: 15.0 # cm/s

stride_filtering:
  confidence_threshold: 0.3
  angular_velocity_min: -20.0 # deg/s
  angular_velocity_max: 20.0  # deg/s

models:
  M1:
    formula: "{outcome} ~ C(Genotype, Treatment('WT')) + scale(body_length_cm)"
  M2:
    formula: "{outcome} ~ C(Genotype, Treatment('WT')) + scale(stride_speed_cm_s_mean) + scale(body_length_cm)"
```

## Project Structure

```
stride/
  config.yaml              # Pipeline configuration
  run_all.py               # Orchestrator (runs all stages)
  src/stride/
    cli.py                 # Click CLI for individual stages
    config.py              # PipelineConfig dataclass
    stages/
      distortion_check.py  # [1] Camera distortion assessment
      pose_inference.py    # [2] SLEAP pose estimation
      roi_inference.py     # [2] ROI region segmentation
      slp_to_yaml.py       # [2] SLP to polygon YAML conversion
      decision_analysis.py # [3] T-maze arm entry decisions
      gait_extraction.py   # [4-5] Keypoint preprocessing + stride detection
      gait_filtering.py    # [6] Stride filtering
      gait_lmm.py          # [7-9] Trial tagging + LMM analysis
      classification.py    # [10] WT vs Tau logistic regression
      run_tmaze.py         # Config-driven T-maze runner
      run_gait_extraction.py # Config-driven gait extraction runner
    utils/
      checkpoint.py        # Resume support for batch processing
      parallel.py          # Multi-GPU parallel processing
      video.py             # Video I/O utilities
```

## Key Outputs

| Stage | Output | Description |
|-------|--------|-------------|
| 3 | `decisions.csv` | Per-trial arm choice, correctness, entry timing |
| 3 | `events.csv` | Detailed timing: segment entries, probes, junction exploration |
| 3 | `learning_cycle_1.csv` | Trials before reward reversal |
| 5 | `gait_per_stride.csv` | Per-stride metrics (14 gait variables) |
| 6 | `gait_per_stride_filtered.csv` | Quality-filtered strides |
| 9 | `lmm_results_all.csv` | M1/M2 LMM coefficients, p-values, FDR |
| 9 | `heatmap_*.png` | Dual-panel heatmaps (-log10(p) + effect sizes) |
| 10 | `regression_roc_curves.png` | ROC curves for T-maze, Gait, Combined |

## Citation

If you use STRIDE in your research, please cite:

> Li, Y. et al. (2026). *Dual cognitive and motor phenotyping for early detection of AD progression in APP and Tau models.* (in preparation)

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Built on [SLEAP](https://sleap.ai) for pose estimation. Gait analysis methodology adapted from [Sheppard et al. (2022)](https://doi.org/10.1016/j.celrep.2022.110231) and [Bellardita & Kiehn (2015)](https://doi.org/10.1016/j.cub.2015.04.005).
