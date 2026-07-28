# Fork notes

This is a fork of [**6DGS** (mbortolon97/6dgs)](https://github.com/mbortolon97/6dgs),
base commit `5236d68`, adapted for our paper:

> **Visual Relocalization from Sparse Views in Aliased and Low-Texture Environments via Novel
> View Synthesis** — Maria Peribañez, Javier Civera, Rudolph Triebel, Riccardo Giubilato (IROS 2026).

The 6DGS pose-estimation method itself is **unchanged**. Our edits only adapt it to our data and
add the retrieval-driven entry point. See the main project repository for the full pipeline.

## Changes over upstream

- **Robustness** (`pose_estimation/sampling.py`, `quadricell.py`): guard quadricell sampling against
  NaN/inf and degenerate ellipsoid scales (our sparse, low-texture splats produce them).
- **DINOv2 loading** (`pose_estimation/backbone.py`): fall back to the cached hub repo when
  `torch.hub` resolves an inconsistent `hubconf.py`.
- **Evaluation** (`pose_estimation/test.py`): optional `gt_c2w_list` to inject ground-truth poses and
  report per-frame translation/rotation error; recall measured against the GT top-k rays.
- **Data conventions** (`scene/tanksandtemples.py`, `scene/cameras.py`): X/Y axis flip for our pose
  files and a larger `zfar` for our scene scale.
- **Bug fix / performance** (`pretrain_eval_attention.py`): `requires_grad` typo fix and one-time ray
  caching.
- **Configuration** (`pretrain_eval_attention.py`, `estimate_pose_from_retrieval_csv.py`):
  `sample_quadricell_targets` lowered from 50 to 10; this is the value used for all our experiments.
- **Build** (`submodules/simple-knn/simple_knn.cu`): add a missing `<cfloat>` include.

## Added files

- `estimate_pose_from_retrieval_csv.py` — run pose estimation for each query against its retrieved
  submaps (reads the retrieval CSV, writes predicted poses and error tables).
- `export_to_6dgs.py` — export a trained nerfstudio 3DGS model to the 6DGS dataset format.

## License

Distributed under the original **Gaussian-Splatting License** (Inria / MPII), non-commercial
research use only. See [`LICENSE.md`](LICENSE.md). All upstream copyright and attribution notices
are retained.
