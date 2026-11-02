# AI-ZETA

Code and results for **"AI-ZETA: neuro-symbolic one-shot tracking of phase evolution in synchrotron Laue diffraction"**.

AI-ZETA follows every reflection in a sequence of in situ Laue frames, labels each one as δ, γ or a non-specimen signal with domain-knowledge rules, and derives the δ → γ transformation start t\*, the δ phase fraction and the Avrami parameters t₀.₅ and n.

## Repository contents

| Folder | Contents |
|---|---|
| `aizeta/` | Analysis pipeline (detection, tracking, labeling, kinetics) |
| `scripts/make_videos.py` | Renders the detection videos |
| `results/` | Results reported in the paper |
| `videos/` | Detection videos of the five runs |
| `data/` | Detector masks of the five runs and the raw frames shown in the paper figures |

## Runs

| Run | Steel | Cooling rate (°C min⁻¹) | Frame rate (fps) |
|---|---|---|---|
| `2_TRIP1` | TRIP T1 | 110 | 10 |
| `8_TRIP2` | TRIP T2 | 30 | 10 |
| `31_A003` | ULC | 132 | 10 |
| `26_V19_5A` | Steel C | 110 | 20 |
| `21_V19_1A` | Steel C | 30 | 20 |

## Pipeline

All modules are run from one working folder that holds `mask_<run>.npy`, `runs.json` and the outputs of the previous steps. Set the data location in `aizeta/config.py` or with the environment variable `AIZETA_DATA` (`<AIZETA_DATA>/laue/<run>/*.tif`, `<AIZETA_DATA>/temperature/<run>.xlsx`).

| Step | Module | Output |
|---|---|---|
| 1. Detection (SAM ViT-H, every frame) | `track01_detect.py <run> <first> <last+1> 1` | `track_det_<run>.csv` |
| 2. Tracking and labeling | `rk_experiment.py <run>` | `rk_out/<run>_site_full_units.csv`, `_obs.csv`, `_site_merges.json`, `rk_results.json` |
| 3. Transformation events | `rk_events.py <run>` | `rk_events.json` |
| 4. Numbered labels | `export_labels.py <run>` | `rk_out/<run>_labels.csv` |
| 5. γ on sapphire (difference images) | `diffsam_sapphire.py <run>` | `rk_out/<run>_sap_gamma.csv` |
| 6. t\*, δ phase fraction, Avrami fit | `refine_timing.py <run>` | `refine_timing.json` |
| 7. Numbered labels, final | `export_labels.py <run>` | `rk_out/<run>_labels.csv` |
| 8. δ of Table 1 and δ disappearance | `table1_paper_delta.py`, `delta_end.py` | `table1_paper_delta.json`, `delta_end.json` |

`clean_delta.py` selects the δ reflections of Steel C 30 with a clean intensity signal (`rk_out/21_V19_1A_clean_delta.json`). The rule parameters are in `aizeta/rk_params.py`. The detection constants are in `prep.py`, `sam_amg.py` and `track01_detect.py`, and the kinetics constants in `refine_timing.py` and `track03_kinetics.py` (all listed in Supplementary Table 5). SAM is loaded through Hugging Face `transformers` (`facebook/sam-vit-huge`).

## Results

`results/runs/` holds, for each run:

- `<run>_labels.csv`: one row per reflection with its label (δ1, γ1, S1, B1, …), phase, position, first and last frame, and the rules that set the label
- `<run>_site_full_units.csv`: labeled lit episodes of each tracked site
- `<run>_site_full_obs.csv`: SAM detections linked to sites (centroid, ellipse, intensity)
- `<run>_site_merges.json`: sites merged into one reflection
- `<run>_sap_gamma.csv`: γ reflections found on sapphire signals

`results/` also holds `refine_timing.json` (t\*, δ phase fraction and Avrami fit), `table1_paper_delta.json` (Table 1), `delta_end.json` (δ disappearance), `rk_results.json`, `rk_events.json` and `runs.json` (analysis window of each run).

## Videos

Each run has two videos. `<run>_overview.mp4` shows the whole run once per second, and `<run>_transformation.mp4` shows every frame from 3 s before the transformation to 2 s after the last δ disappears. Reflections are drawn with their SAM ellipse: red δ, blue γ, gray non-specimen. A δ that is not detected in a frame but has not yet disappeared is drawn as a dashed circle.

The videos are rendered from the pipeline working folder, which needs the raw frames:

```
python <repo>/scripts/make_videos.py <output folder> [run ...]
```

## Raw data

`data/frames/26_V19_5A/` holds the five raw frames of Steel C 110 shown in the paper figures (frames 2752, 2797, 2818, 2842 and 3031, counted from 0; file `IMG2753_…` is frame 2752). The full frame sequences are not included in this repository.

## Requirements

Python 3.11 and the packages in `requirements.txt`. Detection was run on one NVIDIA GeForce RTX 4070 GPU.

## Citation

If you use this code, please cite the paper (reference to be added on publication).

## License

MIT License. See `LICENSE`.
