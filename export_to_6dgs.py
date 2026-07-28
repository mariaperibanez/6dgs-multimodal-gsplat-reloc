import json
import os
import numpy as np
from pathlib import Path
from shutil import copyfile
import argparse

import torch
from nerfstudio.utils.eval_utils import eval_setup

# ============================================================
# CONSTANTS
# ============================================================

# Flip OpenGL → GLM (same as for the poses)
FLIP = np.diag([-1, 1, -1])


# ============================================================
# TRANSFORMS.JSON
# ============================================================

def load_original_transforms(path):
    with open(path, "r") as f:
        return json.load(f)


def export_intrinsics(data, out_dir):
    fx, fy = data["fl_x"], data["fl_y"]
    cx, cy = data["cx"], data["cy"]

    intr_path = Path(out_dir) / "intrinsics.txt"
    with open(intr_path, "w") as f:
        f.write(f"{fx} 0 {cx}\n")
        f.write(f"0 {fy} {cy}\n")
        f.write("0 0 1\n")

    print(f"[ok] intrinsics.txt → {intr_path}")


# ============================================================
# CAMERA MATRICES
# ============================================================

def c2w_from_3x4(block34):
    M = np.eye(4)
    M[:3, :4] = block34
    return M


def convert_c2w_to_6dgs(c2w):
    """Convert c2w (Nerfstudio / OpenGL) to GLM (6DGS)."""

    R = c2w[:3, :3]          # Rotation (OpenGL)
    T = c2w[:3, 3]           # Translation (OpenGL)

    # Rotation OpenGL → GLM
    R_glm = FLIP @ R @ FLIP

    # Translation OpenGL → GLM
    T_glm = FLIP @ T

    # Build the GLM 4x4 c2w
    M = np.eye(4)
    M[:3, :3] = R_glm
    M[:3, 3] = T_glm
    return M

# ============================================================
# EXPORT GAUSSIANS IN 6DGS FORMAT
# ============================================================

def export_gaussians_for_6dgs(model, out_ply):
    gauss = model.gauss_params
    sh_degree = model.config.sh_degree

    means = gauss.means.detach().cpu().numpy()
    opacities = gauss.opacities.detach().cpu().numpy()
    scales = gauss.scales.detach().cpu().numpy()
    quats = gauss.quats.detach().cpu().numpy()
    feat_dc = gauss.features_dc.detach().cpu().numpy()
    feat_rest = gauss.features_rest.detach().cpu().numpy()

    N = means.shape[0]

    # Flatten rest features if they are 2D per gaussian
    if feat_rest.ndim == 3:
        # (N, M, D) --> (N, M*D)
        feat_rest = feat_rest.reshape(N, -1)

    expected_rest = 3 * (sh_degree + 1) ** 2 - 3
    assert feat_dc.shape[1] == 3
    assert feat_rest.shape[1] == expected_rest, (feat_rest.shape, expected_rest)

    # OpenGL → GLM flip on positions
    means = means @ FLIP

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
    ]

    for i in range(feat_dc.shape[1]):
        header.append(f"property float f_dc_{i}")

    for i in range(feat_rest.shape[1]):
        header.append(f"property float f_rest_{i}")

    header.append("end_header\n")

    out_ply = Path(out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)

    with open(out_ply, "w") as f:
        f.write("\n".join(header))

        for i in range(N):
            row = np.concatenate([
                means[i],
                opacities[i],
                scales[i],
                quats[i],
                feat_dc[i],
                feat_rest[i],   # now it is a 1D vector
            ])
            f.write(" ".join(map(str, row)) + "\n")

    print(f"[ok] Gaussian Model exported for 6DGS → {out_ply}")

# ============================================================
# EXTRACT OPTIMIZED POSES + MODEL
# ============================================================

def extract_optimized_poses_and_model(config_path):
    config_path = Path(config_path)

    def fix_paths(cfg):
        true_dir = (config_path.parent / "nerfstudio_models").resolve()
        print(f"[INFO] load_dir = {true_dir}")
        cfg.get_checkpoint_dir = lambda: true_dir
        cfg.load_dir = true_dir
        print(f"[INFO] data dir = {getattr(cfg, 'data', None)}")
        return cfg

    config, pipeline, _, _ = eval_setup(
        config_path,
        update_config_callback=fix_paths
    )
    pipeline.eval()

    cameras = pipeline.datamanager.train_dataparser_outputs.cameras
    model = pipeline.model

    optimized_c2w_list = []
    for i in range(len(cameras)):
        cam = cameras[i:i+1]
        opt34 = model.camera_optimizer.apply_to_camera(cam)[0].cpu().numpy()
        optimized_c2w_list.append(c2w_from_3x4(opt34))

    return optimized_c2w_list, model


# ============================================================
# EXPORT DATASET 6DGS
# ============================================================

def export_dataset(config_path, data_path, out_dir, ply_out):
    out_dir = Path(out_dir)
    rgb_out = out_dir / "rgb"
    pose_out = out_dir / "pose"
    rgb_out.mkdir(parents=True, exist_ok=True)
    pose_out.mkdir(parents=True, exist_ok=True)

    transforms = load_original_transforms(Path(data_path) / "transforms.json")
    export_intrinsics(transforms, out_dir)

    frames = transforms["frames"]
    N = len(frames)
    N_train = int(0.8 * N)

    print(f"Total frames: {N} | Train: {N_train} | Test: {N - N_train}")

    # Use the ORIGINAL poses from the dataset (not optimized)
    original_c2ws = []
    for f in frames:
        M = np.array(f["transform_matrix"])
        original_c2ws.append(M)

    # Extract optimized poses + GS model
    optimized_c2ws, model = extract_optimized_poses_and_model(config_path)

    # Export images + poses
    print("Exporting 6DGS dataset…")
    for idx, (frame, c2w) in enumerate(zip(frames, original_c2ws)):
        prefix = "0_" if idx < N_train else "2_"
        name = f"{prefix}{idx:06d}"

        # Image
        src_img = frame["file_path"]
        dst_img = rgb_out / f"{name}.png"
        copyfile(src_img, dst_img)

        # GLM pose
        M = convert_c2w_to_6dgs(c2w)
        np.savetxt(pose_out / f"{name}.txt", M)

    print("[ok] 6DGS dataset exported correctly.")

    # Export Gaussian Model in 6DGS format
    export_gaussians_for_6dgs(model, ply_out)

    print("\n[done] Export COMPLETE: dataset + gaussian model ready for 6DGS")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export dataset + Gaussian Model for 6DGS")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yml")
    parser.add_argument("--data", type=str, required=True, help="Directory with transforms.json")
    parser.add_argument("--out", type=str, required=True, help="Destination directory for the 6DGS dataset")
    parser.add_argument("--ply_out", type=str, required=True,
                        help="Destination path for the 6DGS PLY (e.g.: /.../model_6dgs.ply)")

    args = parser.parse_args()

    export_dataset(args.config, args.data, args.out, args.ply_out)
