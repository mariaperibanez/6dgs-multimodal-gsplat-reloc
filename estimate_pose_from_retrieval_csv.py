#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
estimate_pose_from_retrieval_csv.py

Generates a CSV with pose predictions (pred_c2w) for each (query, candidate submap).
It now also saves:
- pred_sub: submap used for pose estimation (previously cand_submap)
- dt_ms: estimation time (milliseconds)
"""

from pose_estimation.file_utils import get_checkpoint_arguments
from pose_estimation.identification_module import IdentificationModule
from pose_estimation.sampling import generate_all_possible_rays
from scene import GaussianModel, load_data
from pose_estimation.test import test_pose_estimation
from scene.scene_structure import CameraInfo

import os
import csv
import glob
import re
import argparse
import traceback
import numpy as np
import torch
from PIL import Image
import time

import json

def query_timestamp_ns(query_img: str):
    m = re.search(r"img(\d+(?:\.\d+)?)", str(query_img))
    if not m:
        return None
    return int(round(float(m.group(1)) * 1e9))

def load_kf_index(path):
    idx = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split(",")
            idx[int(p[0])] = (int(p[2]), int(p[3]))  # (submap_id, kf_idx)
    return idx

def load_submap_info(submap_id, base_dir):
    path = os.path.join(base_dir, f"submap_{submap_id}", "transforms.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    frames = {os.path.basename(fr["file_path"]): np.array(fr["transform_matrix"], dtype=np.float32)
              for fr in data.get("frames", [])}
    return frames

# -----------------------------
# Model / Rays helpers
# -----------------------------
def load_model(ply_path: str, device: str, sh_degrees: int = 3):
    model = GaussianModel(sh_degrees)
    model.load_ply(ply_path)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def explore_model(model: GaussianModel):
    rays_ori, rays_dirs, rays_rgb = generate_all_possible_rays(
        model,
        sample_quadricell_targets=10,
    )
    return rays_ori, rays_dirs, rays_rgb


def make_query_camerainfo(qpath: str, scene_info, uid: int = 10**9):
    img = Image.open(qpath).convert("RGB")
    ref = scene_info.train_cameras[0]

    if hasattr(ref, "width") and hasattr(ref, "height"):
        img = img.resize((ref.width, ref.height), resample=Image.BILINEAR)
        width, height = ref.width, ref.height
    else:
        width, height = img.size[0], img.size[1]

    R = np.eye(3, dtype=np.float32)    # dummy
    T = np.zeros(3, dtype=np.float32)  # dummy

    return CameraInfo(
        uid=uid,
        R=R,
        T=T,
        FovY=ref.FovY,
        FovX=ref.FovX,
        image=img,
        image_path=qpath,
        image_name=os.path.basename(qpath),
        width=width,
        height=height,
    )


@torch.no_grad()
def estimate_pose_for_query(
    qpath: str,
    scene_info,
    id_module,
    rays_ori,
    rays_dirs,
    rays_rgb,
    model_up,
    gt_c2w=None,
):
    q_cam = make_query_camerainfo(qpath, scene_info)

    t0 = time.perf_counter()
    results, *_ = test_pose_estimation(
        cameras_info=[q_cam],
        id_module=id_module,
        rays_ori=rays_ori,
        rays_dirs=rays_dirs,
        rays_rgb=rays_rgb,
        model_up=model_up,
        loss_fn=None,
        save=False,
        save_all=False,
        gt_c2w_list=[gt_c2w] if gt_c2w is not None else None,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    pred = np.array(results[0]["pred_c2w"])
    info = results[0]
    t_err = info.get("t_err_m", "")
    r_err = info.get("r_err_deg", "")

    info["dt_ms"] = dt_ms
    return pred, info


# -----------------------------
# Submap indexing + caching
# -----------------------------
class SubmapCache:
    def __init__(self, device="cuda"):
        self.device = device
        self.cache = {}

    def get(self, experiment_entry: dict):
        key = str(experiment_entry["sequence_id"])
        if key in self.cache:
            return self.cache[key]

        exp_dir = experiment_entry["exp_dir_filepath"]
        print(f"[INFO] Loading submap {key} from {exp_dir}...")
        ply_path = experiment_entry["checkpoint_filepath"]
        id_module_path = experiment_entry["id_module_path"]

        ckpt_args = get_checkpoint_arguments(exp_dir)
        print(f"[INFO] Checkpoint arguments: {ckpt_args}")
        gs_model = load_model(ply_path, self.device, sh_degrees=ckpt_args.sh_degree)
        scene_info = load_data(ckpt_args)

        id_module = IdentificationModule(backbone_type="dino").to(self.device, non_blocking=True).eval()
        ckpt_dict = torch.load(id_module_path, map_location=self.device)
        id_module.load_state_dict(ckpt_dict["model_state_dict"])

        rays_ori, rays_dirs, rays_rgb = explore_model(gs_model)

        model_up_np = np.mean(
            np.asarray([cam.R[:3, 1] for cam in scene_info.train_cameras], dtype=np.float32),
            axis=0,
        )
        model_up = torch.from_numpy(model_up_np).to(device=self.device, non_blocking=True)

        self.cache[key] = {
            "ply_path": ply_path,
            "scene_info": scene_info,
            "id_module": id_module,
            "rays_ori": rays_ori,
            "rays_dirs": rays_dirs,
            "rays_rgb": rays_rgb,
            "model_up": model_up,
        }
        return self.cache[key]


def _latest_ply_in_dir(d: str):
    cands = glob.glob(os.path.join(d, "point_cloud", "iteration_*", "point_cloud.ply"))
    if not cands:
        return None

    def itnum(p):
        m = re.search(r"iteration_(\d+)", p)
        return int(m.group(1)) if m else -1

    cands.sort(key=itnum)
    return cands[-1]


def build_experiments_by_submap_doneprefix(output_root: str):
    experiments_by_submap = {}
    dirs = sorted(glob.glob(os.path.join(output_root, "_nomvs_chamfer_lidarinit_*")))
    print(f"[INFO] Searching for submaps in {output_root}...")
    for d in dirs:
        base = os.path.basename(d)
        m = re.match(r"_nomvs_chamfer_lidarinit_submap_(\d+)", base)
        if not m:
            continue
        sid = m.group(1)

        id_module_path = os.path.join(d, "id_module.th")
        ply_path = _latest_ply_in_dir(d)

        if not os.path.exists(id_module_path):
            print(f"[WARN] Missing id_module.th in {d}, skipping.")
            continue
        if ply_path is None or (not os.path.exists(ply_path)):
            print(f"[WARN] Cannot find point_cloud.ply in {d}, skipping.")
            continue

        experiments_by_submap[sid] = {
            "sequence_id": sid,
            "exp_dir_filepath": d,
            "id_module_path": id_module_path,
            "checkpoint_filepath": ply_path,
        }

    print(f"[INFO] Indexed {len(experiments_by_submap)} submaps from {output_root}")
    return experiments_by_submap


# -----------------------------
# Retrieval CSV parsing + output
# -----------------------------
def _to_bool(x):
    s = str(x).strip().lower()
    return s in ("true", "1", "yes", "y", "t")


def read_retrieval_csv(csv_path: str):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for d in reader:
            if not d:
                continue

            q = d.get("query_image") or d.get("query_img") or d.get("query_path")
            if q is None:
                continue

            cs = d.get("matched_kf_submap") or d.get("cand_submap") or d.get("submap")
            if cs is None:
                continue

            cand_img = d.get("matched_kf_image") or d.get("cand_img") or d.get("matched_kf_path") or ""

            rank = d.get("rank") or d.get("topk_rank") or d.get("k") or ""
            try:
                rank = float(rank) if rank != "" else ""
            except Exception:
                rank = ""

            sim = d.get("sim") or d.get("score") or d.get("similarity") or ""
            try:
                sim = float(sim) if sim != "" else ""
            except Exception:
                sim = ""

            cand_ok = d.get("cand_ok") or d.get("matched_kf_ok") or ""
            cand_ok = _to_bool(cand_ok) if cand_ok != "" else True

            try:
                pred_sub = int(float(cs))
            except Exception:
                m = re.search(r"(\d+)", str(cs))
                if not m:
                    continue
                pred_sub = int(m.group(1))

            rows.append({
                "query_img": q,
                "pred_sub": pred_sub,   # clear name
                "cand_img": cand_img,
                "rank": rank,
                "sim": sim,
                "cand_ok": cand_ok,
            })
    return rows


def flatten_4x4(mat4):
    m = np.asarray(mat4, dtype=np.float32).reshape(4, 4)
    return m.reshape(-1).tolist()


def init_retr_pose_csv(out_csv: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    header = ["query_img","pred_sub","gt_sub","cand_img","rank","sim","ply_path","dt_ms","t_err_m","r_err_deg"]
    header += [f"pred_c2w_{i:02d}" for i in range(16)]
    header += ["score"]
    if not os.path.exists(out_csv):
        with open(out_csv, "w", newline="") as f:
            csv.writer(f).writerow(header)


def append_retr_pose_csv(out_csv: str, d: dict, ply_path: str, pred_c2w, info: dict):
    score = ""
    if isinstance(info, dict):
        for k in ("score", "pred_score", "best_score"):
            if k in info:
                score = info[k]
                break
    dt_ms = info.get("dt_ms", "")
    t_err = info.get("t_err_m", "")
    r_err = info.get("r_err_deg", "")

    row = [
        d["query_img"],
        int(d["pred_sub"]),
        int(d["gt_sub"]),
        d["cand_img"],
        d["rank"],
        d["sim"],
        ply_path,
        dt_ms,
        t_err,
        r_err,
    ] + flatten_4x4(pred_c2w) + [score]

    with open(out_csv, "a", newline="") as f:
        csv.writer(f).writerow(row)


def run_pose_for_retrieval_csv(
        retrieval_csv: str,
        out_csv: str,
        experiments_by_submap: dict,
        kf_path: str,
        submap_dir: str,
        device: str = "cuda",
        only_if_cand_ok: bool = False,
):

    init_retr_pose_csv(out_csv)
    rows = read_retrieval_csv(retrieval_csv)

    cache = SubmapCache(device=device)
    seen = set()
    kf_index = load_kf_index(kf_path)
    all_stamps = np.array(sorted(kf_index.keys()), dtype=np.int64)
    frames_cache = {}  # gt_submap_id -> frames dict

    for d in rows:
        if only_if_cand_ok and (not d["cand_ok"]):
            continue
        key = (d["query_img"], int(d["pred_sub"]))
        if key in seen:
            continue
        seen.add(key)

        submap_id = str(d["pred_sub"])
        if submap_id not in experiments_by_submap:
            print(f"[WARN] No submap {submap_id} in output. Skipping.")
            continue

        try:
            pack = cache.get(experiments_by_submap[submap_id])
            tq = query_timestamp_ns(d["query_img"])
            if tq is None:
                print(f"[WARN] No timestamp in query_img={d['query_img']}, skipping.")
                continue

            idx_near = int(np.argmin(np.abs(all_stamps - tq)))
            gt_sub_id, gt_kf_idx = kf_index[int(all_stamps[idx_near])]
            kf_used = f"kf_{gt_kf_idx}.png"

            if gt_sub_id not in frames_cache:
                frames_cache[gt_sub_id] = load_submap_info(gt_sub_id, submap_dir)

            frames_gt = frames_cache[gt_sub_id]
            if frames_gt is None or (kf_used not in frames_gt):
                print(f"[WARN] Cannot find GT frame {kf_used} in submap {gt_sub_id}, skipping.")
                continue

            gt_c2w = frames_gt[kf_used]   # 4x4 numpy
            d["gt_sub"] = int(gt_sub_id)


            pred_c2w, info = estimate_pose_for_query(
                d["query_img"],
                pack["scene_info"],
                pack["id_module"],
                pack["rays_ori"],
                pack["rays_dirs"],
                pack["rays_rgb"],
                pack["model_up"],
                gt_c2w=gt_c2w,
            )

            append_retr_pose_csv(out_csv, d, pack["ply_path"], pred_c2w, info)
            sim_str = f"{d['sim']:.3f}" if isinstance(d["sim"], float) else str(d["sim"])
            print(f"[OK] query={os.path.basename(d['query_img'])} submap={submap_id} rank={d['rank']} sim={sim_str} dt_ms={info.get('dt_ms',0):.1f}")

        except Exception as e:
            print(f"[FAIL] query={d['query_img']} submap={submap_id}: {e}")
            traceback.print_exc()

import pandas as pd
import numpy as np
import os

def make_tables_from_pose_csv(pred_csv: str, out_tables_dir: str):
    os.makedirs(out_tables_dir, exist_ok=True)
    df = pd.read_csv(pred_csv)

    # in case there are empty strings
    for c in ["rank", "dt_ms", "t_err_m", "r_err_deg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if df.empty:
        print(f"[WARN] Empty CSV: {pred_csv}. Not generating tables.")
        return

    def agg_block(g):
        return pd.Series({
            "N": len(g),
            "t_med": g["t_err_m"].median(),
            "t_mean": g["t_err_m"].mean(),
            "r_med": g["r_err_deg"].median(),
            "r_mean": g["r_err_deg"].mean(),
            "dt_med": g["dt_ms"].median() if "dt_ms" in g else np.nan,
        })

    # Global
    agg_block(df).to_frame().T.to_csv(os.path.join(out_tables_dir, "summary_global.csv"), index=False)

    # By predicted submap
    if "pred_sub" in df.columns:
        df.groupby("pred_sub", as_index=False).apply(agg_block).reset_index(drop=True) \
            .to_csv(os.path.join(out_tables_dir, "summary_by_pred_sub.csv"), index=False)

    # By rank
    if "rank" in df.columns:
        df.dropna(subset=["rank"]).groupby("rank", as_index=False).apply(agg_block).reset_index(drop=True) \
            .to_csv(os.path.join(out_tables_dir, "summary_by_rank.csv"), index=False)

    # Recall@K (with two typical thresholds: strict and loose)
    thresholds = [(2.0, 10.0), (10.0, 15.0)]
    Ks = [1, 3, 5, 10]

    if "rank" in df.columns and "query_img" in df.columns:
        df_ranked = df.dropna(subset=["rank"]).copy()
        df_ranked["query"] = df_ranked["query_img"].apply(lambda x: os.path.basename(str(x)))

        rec_rows = []
        for K in Ks:
            dK = df_ranked[df_ranked["rank"] <= K]
            row = {"K": K}
            num_queries = None
            for thr_t, thr_r in thresholds:
                ok = dK.groupby("query").apply(
                    lambda g: ((g["t_err_m"] <= thr_t) & (g["r_err_deg"] <= thr_r)).any()
                )
                row[f"recall_t<{thr_t}_r<{thr_r}"] = float(ok.mean()) if len(ok) else np.nan
                num_queries = int(len(ok))
            row["num_queries"] = num_queries
            rec_rows.append(row)
        pd.DataFrame(rec_rows).to_csv(os.path.join(out_tables_dir, "recall_at_k.csv"), index=False)

    print(f"[OK] Tables saved in: {out_tables_dir}")

def parse_args():
    ap = argparse.ArgumentParser(
        description="Estimate 6-DoF pose for each query against its retrieved submaps."
    )
    ap.add_argument("--output-root", required=True,
                    help="6dgs output dir with the trained per-submap models "
                         "(e.g. <SIXDGS_ROOT>/output/<GS_TAG>).")
    ap.add_argument("--retrieval-csv", required=True,
                    help="MPRF retrieval CSV (query -> candidate submaps).")
    ap.add_argument("--kf-path", required=True,
                    help="keyframe_timestamps.csv.")
    ap.add_argument("--submap-dir", required=True,
                    help="parent folder of submap_<id>/.")
    ap.add_argument("--out-csv", required=True,
                    help="output CSV with predicted poses.")
    ap.add_argument("--tables-dir", required=True,
                    help="output dir for the metric tables.")
    ap.add_argument("--only-if-cand-ok", action="store_true",
                    help="only run for candidates flagged ok.")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    experiments_by_submap = build_experiments_by_submap_doneprefix(args.output_root)

    if os.path.exists(args.out_csv):
        os.remove(args.out_csv)

    run_pose_for_retrieval_csv(
        retrieval_csv=args.retrieval_csv,
        out_csv=args.out_csv,
        experiments_by_submap=experiments_by_submap,
        kf_path=args.kf_path,
        submap_dir=args.submap_dir,
        device=device,
        only_if_cand_ok=args.only_if_cand_ok,
    )
    make_tables_from_pose_csv(args.out_csv, args.tables_dir)


if __name__ == "__main__":
    main()
