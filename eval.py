from __future__ import print_function

import csv
import json
import os

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

import options
from models import model as model_cls
from video_dataset_anomaly_balance_uni_sample_ucf import (
    Dataset_Con_all_feedback_XD,
    infer_feature_info,
    normalize_dataset_name,
)


FINAL_FUSION_LAMBDA = 64.0
FINAL_GATE_Q = 97.0
FINAL_AUX_MODE = "rank_inv_mean"
FINAL_POST_SMOOTH = 7
FINAL_POST_MIX = 0.12
FINAL_POST_Q = 45.0
FINAL_POST_STRENGTH = 0.70


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _safe_label(path):
    if not path:
        return ""
    name = os.path.basename(path.rstrip(os.sep))
    parent = os.path.basename(os.path.dirname(path.rstrip(os.sep)))
    if parent == "bestckpt":
        return os.path.join(parent, name)
    if parent == "results":
        return os.path.join(parent, name)
    return name


def _resolve_cuda_index(device_arg):
    if isinstance(device_arg, int):
        return int(device_arg)
    value = str(device_arg).strip().lower()
    if value.startswith("cuda:"):
        value = value.split(":", 1)[1]
    if value.isdigit():
        return int(value)
    raise ValueError("Unsupported --device value '{}'. Use 0 or cuda:0.".format(device_arg))


def _load_model_state(ckpt_path):
    state = load_checkpoint(ckpt_path)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            return state, state["model_state_dict"]
        if "model_state" in state:
            return state, state["model_state"]
    return state, state


def _adapt_args_from_state(args, state):
    if not isinstance(state, dict):
        return
    model_state = state.get("model_state_dict", state.get("model_state", state))
    if not isinstance(model_state, dict):
        return
    bank = model_state.get("inc_memory_bank", None)
    if torch.is_tensor(bank) and bank.ndim == 2 and bank.shape[0] > 0:
        args.inc_memory_max = int(bank.shape[0])
        print("Auto set inc_memory_max={} from checkpoint.".format(args.inc_memory_max))


def _freeze_model(net):
    for param in net.parameters():
        param.requires_grad = False


def _safe_load_state(model, model_state):
    try:
        model.load_state_dict(model_state, strict=False)
        return
    except RuntimeError as exc:
        own_state = model.state_dict()
        filtered = {}
        skipped = []
        for key, value in model_state.items():
            if key in own_state and tuple(own_state[key].shape) == tuple(value.shape):
                filtered[key] = value
            else:
                skipped.append(key)
        print("[WARN] filtered checkpoint keys after shape mismatch: {}".format(len(skipped)))
        if skipped:
            print("[WARN] first skipped keys:", ", ".join(skipped[:12]))
        model.load_state_dict(filtered, strict=False)
        print("[WARN] original load error was:", str(exc).split("\n")[0])


def build_model_from_args(args):
    train_dim, _, _ = infer_feature_info(args, train=True)
    test_dim, _, test_shape = infer_feature_info(args, train=False)
    inferred_input_dim = int(args.feature_size)
    if train_dim is not None:
        inferred_input_dim = int(train_dim)
    elif test_dim is not None:
        inferred_input_dim = int(test_dim)

    proj_dim = int(getattr(args, "proj_feature_size", 0))
    model_feature_size = proj_dim if proj_dim > 0 else int(args.feature_size)
    input_feature_size = int(inferred_input_dim)
    args.feature_size = int(model_feature_size)

    if test_shape is not None:
        print("Eval test npy shape:", test_shape)
    print("Eval feature config: input_dim={} -> model_dim={}".format(input_feature_size, args.feature_size))

    net = model_cls(
        args.max_seqlen,
        feature_size=args.feature_size,
        input_feature_size=input_feature_size,
        Vitblock_num=args.Vitblock_num,
        cross_clip=args.cross_clip,
        split=0,
        beta=args.beta,
        delta=args.delta,
        score_rec_consistency_weight=args.score_rec_consistency_weight,
        use_qavclab=args.use_qavclab,
        short_clip=args.short_clip,
        long_clip=args.long_clip,
        retrieval_topk=args.retrieval_topk,
        retrieval_samples=args.retrieval_samples,
        bank_decay=args.bank_decay,
        bank_age_weight=args.bank_age_weight,
        difficulty_weight=args.difficulty_weight,
        uncertainty_weight=args.uncertainty_weight,
        consistency_weight=args.consistency_weight,
        teacher_momentum=args.teacher_momentum,
        teacher_threshold=args.teacher_threshold,
        min_pseudo_pos=args.min_pseudo_pos,
        max_pseudo_pos=args.max_pseudo_pos,
        max_pseudo_pos_start=args.max_pseudo_pos_start,
        max_pseudo_pos_end=args.max_pseudo_pos_end,
        max_pseudo_pos_anneal_iters=args.max_pseudo_pos_anneal_iters,
        pseudo_loss_mode=args.pseudo_loss_mode,
        pseudo_pos_weight_max=args.pseudo_pos_weight_max,
        pseudo_rank_weight=args.pseudo_rank_weight,
        pseudo_rank_margin=args.pseudo_rank_margin,
        pseudo_rank_max_samples=args.pseudo_rank_max_samples,
        pseudo_rank_neg_ratio=args.pseudo_rank_neg_ratio,
        rec_rank_weight=args.rec_rank_weight,
        rec_rank_margin=args.rec_rank_margin,
        rec_rank_top_ratio=args.rec_rank_top_ratio,
        rec_rank_bottom_ratio=args.rec_rank_bottom_ratio,
        rec_rank_max_samples=args.rec_rank_max_samples,
        easy_neg_weight=args.easy_neg_weight,
        easy_neg_ratio=args.easy_neg_ratio,
        easy_neg_margin=args.easy_neg_margin,
        easy_neg_max_samples=args.easy_neg_max_samples,
        easy_neg_focus_ratio=args.easy_neg_focus_ratio,
        score_sparse_weight=args.score_sparse_weight,
        score_sparse_target=args.score_sparse_target,
        score_sparse_flat_weight=args.score_sparse_flat_weight,
        score_sparse_flat_margin=args.score_sparse_flat_margin,
        score_sparse_top_ratio=args.score_sparse_top_ratio,
        score_sparse_rec_aware=args.score_sparse_rec_aware,
        score_sparse_min_weight=args.score_sparse_min_weight,
        enforce_min_pseudo_pos=args.enforce_min_pseudo_pos,
        hard_route_infer=args.hard_route_infer,
        infer_with_teacher=args.infer_with_teacher,
        infer_teacher_alpha=args.infer_teacher_alpha,
        infer_with_anchor=args.infer_with_anchor,
        infer_anchor_alpha=args.infer_anchor_alpha,
        infer_with_recon=args.infer_with_recon,
        infer_recon_alpha=args.infer_recon_alpha,
        infer_with_memory_novelty=args.infer_with_memory_novelty,
        infer_memory_alpha=args.infer_memory_alpha,
        infer_memory_frozen_only=args.infer_memory_frozen_only,
        full_evidence_enable=args.full_evidence_enable,
        full_evidence_loss_weight=args.full_evidence_loss_weight,
        full_evidence_infer=args.full_evidence_infer,
        full_evidence_alpha=args.full_evidence_alpha,
        full_evidence_rec_weight=args.full_evidence_rec_weight,
        full_evidence_temp_weight=args.full_evidence_temp_weight,
        full_evidence_center_weight=args.full_evidence_center_weight,
        full_evidence_power=args.full_evidence_power,
        lab_start_iter=args.lab_start_iter,
        route_start_iter=args.route_start_iter,
        enable_teacher_gate=args.enable_teacher_gate,
        teacher_gate_start_iter=args.teacher_gate_start_iter,
        route_batch_cap=args.route_batch_cap,
        freeze_c2fpl=args.freeze_c2fpl,
        scorer_dropout=args.scorer_dropout,
        input_norm=args.input_norm,
        scorer_hidden1=args.scorer_hidden1,
        scorer_hidden2=args.scorer_hidden2,
        inc_enable=args.inc_enable,
        inc_scene_id=args.inc_scene_id,
        inc_memory_max=args.inc_memory_max,
        inc_memory_init=args.inc_memory_init,
        inc_match_temp=args.inc_match_temp,
        inc_match_threshold=args.inc_match_threshold,
        inc_match_sim_threshold=args.inc_match_sim_threshold,
        inc_expand_warmup_iters=args.inc_expand_warmup_iters,
        inc_proto_momentum=args.inc_proto_momentum,
        inc_mem_easy_ratio=args.inc_mem_easy_ratio,
        inc_mem_score_quantile=args.inc_mem_score_quantile,
        inc_mem_score_weight=args.inc_mem_score_weight,
        inc_replay_ratio=args.inc_replay_ratio,
        inc_replay_max_samples=args.inc_replay_max_samples,
        inc_replay_scene_quota=args.inc_replay_scene_quota,
        inc_replay_topk=args.inc_replay_topk,
        inc_replay_alpha=args.inc_replay_alpha,
        inc_replay_noise=args.inc_replay_noise,
        inc_replay_min_sim=args.inc_replay_min_sim,
        inc_replay_hard_ratio=args.inc_replay_hard_ratio,
        inc_replay_weight=args.inc_replay_weight,
        inc_replay_warmup_iters=args.inc_replay_warmup_iters,
        inc_replay_score_weight=args.inc_replay_score_weight,
        inc_replay_loss_mode=args.inc_replay_loss_mode,
        inc_replay_putback_memory=args.inc_replay_putback_memory,
        inc_expected_scenes=args.inc_expected_scenes,
        inc_scene_proto_cap=args.inc_scene_proto_cap,
        inc_scene_min_slots=args.inc_scene_min_slots,
        inc_overflow_policy=args.inc_overflow_policy,
        inc_overflow_expand=args.inc_overflow_expand,
        inc_anchor_replay_weight=args.inc_anchor_replay_weight,
    )
    return net


def _list_domains(dil_root, prefix):
    if not os.path.isdir(dil_root):
        return []
    return [
        name
        for name in sorted(os.listdir(dil_root))
        if name.startswith(prefix) and os.path.isdir(os.path.join(dil_root, name))
    ]


def _prepare_dil_args(args):
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError("checkpoint file is missing")
    if not os.path.isdir(args.dil_root):
        raise FileNotFoundError("DIL root is missing or not a directory")

    args.ucf_test_npy = os.path.join(args.dil_root, "global_test.npy")
    args.eval_gt_npy = os.path.join(args.dil_root, "global_test_gt.npy")
    args.concat_meta_json = os.path.join(args.dil_root, "global_build_meta.json")
    for name, path in [
        ("global_test.npy", args.ucf_test_npy),
        ("global_test_gt.npy", args.eval_gt_npy),
        ("global_build_meta.json", args.concat_meta_json),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError("{} is missing under DIL root".format(name))

    if args.dil_dataset == "ucf":
        domains = _list_domains(args.dil_root, "domain")
    else:
        domains = _list_domains(args.dil_root, "scene") or _list_domains(args.dil_root, "domain")

    if int(getattr(args, "inc_scene_id", 0)) <= 0 and domains and int(getattr(args, "inc_expected_scenes", 1)) > 1:
        args.inc_scene_id = len(domains) - 1
    if not getattr(args, "ucf_train_npy", "") and domains:
        first_train = os.path.join(args.dil_root, domains[0], "train_unlabel.npy")
        if os.path.exists(first_train):
            args.ucf_train_npy = first_train


def _load_eval_model(args, device):
    raw_state, model_state = _load_model_state(args.ckpt_path)
    _adapt_args_from_state(args, raw_state)
    model = build_model_from_args(args)
    _freeze_model(model)
    _safe_load_state(model, model_state)
    if hasattr(model, "set_incremental_scene"):
        model.set_incremental_scene(int(getattr(args, "inc_scene_id", 0)))
    model.to(device)
    model.eval()
    return model


def _reduce_logits(logits_np, mode="first"):
    arr = np.asarray(logits_np)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr[:, 0].astype(np.float32)
        if mode == "max":
            return np.max(arr, axis=1).astype(np.float32)
        if mode == "last":
            return arr[:, -1].astype(np.float32)
        if mode == "mean":
            return np.mean(arr, axis=1).astype(np.float32)
        return arr[:, 0].astype(np.float32)
    if arr.ndim >= 3:
        if arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        else:
            arr = np.mean(arr, axis=-1)
        return _reduce_logits(arr, mode=mode)
    return arr.reshape(-1).astype(np.float32)


def _load_video_token_lengths(args, token_count):
    candidates = []
    if getattr(args, "concat_meta_json", ""):
        candidates.append(args.concat_meta_json)
    if getattr(args, "eval_gt_npy", ""):
        candidates.append(os.path.join(os.path.dirname(args.eval_gt_npy), "build_meta.json"))
    if getattr(args, "ucf_test_npy", ""):
        candidates.append(os.path.join(os.path.dirname(args.ucf_test_npy), "build_meta.json"))

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        lens = None
        if isinstance(meta.get("test_video_token_lengths"), list):
            lens = [int(item.get("tokens", 0)) for item in meta["test_video_token_lengths"]]
        elif isinstance(meta.get("test_token_lengths"), dict):
            lens = [int(v) for v in meta["test_token_lengths"].values()]
        elif isinstance(meta.get("test_token_lengths"), list):
            lens = [int(v) for v in meta["test_token_lengths"]]
        if lens:
            lens = [v for v in lens if v > 0]
            if sum(lens) == int(token_count):
                return lens
    return None


def _moving_average_1d(x, kernel):
    x = np.asarray(x, dtype=np.float32)
    kernel = int(kernel)
    if kernel <= 1 or x.size <= 1:
        return x
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    filt = np.ones((kernel,), dtype=np.float32) / float(kernel)
    return np.convolve(padded, filt, mode="valid")[:x.size].astype(np.float32)


def _video_baseline_suppress(x, q=55.0, strength=0.45, rescale=True):
    x = np.asarray(x, dtype=np.float32)
    if x.size <= 1:
        return x
    q = float(np.clip(q, 0.0, 99.5))
    strength = float(np.clip(strength, 0.0, 1.0))
    baseline = float(np.percentile(x, q))
    if not np.isfinite(baseline) or baseline <= 1e-6 or strength <= 0:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    shift = float(np.clip(strength * baseline, 0.0, 0.98))
    y = x - shift
    if bool(rescale):
        y = y / max(1.0 - shift, 1e-6)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _postprocess_tokens(token_scores, args, video_lengths=None):
    x = np.asarray(token_scores, dtype=np.float32)
    if x.size == 0:
        return x
    if not bool(getattr(args, "use_eval_postproc", False)):
        return np.clip(x, 0.0, 1.0)

    smooth = int(getattr(args, "test_smooth_kernel", 1))
    baseline_mix = min(max(float(getattr(args, "test_video_baseline_mix", 0.0)), 0.0), 1.0)
    baseline_q = float(getattr(args, "test_video_baseline_q", 55.0))
    baseline_strength = float(getattr(args, "test_video_baseline_strength", 0.45))
    baseline_rescale = bool(getattr(args, "test_video_baseline_rescale", True))

    if not video_lengths:
        video_lengths = [len(x)]
    out = np.zeros_like(x)
    start = 0
    for length in video_lengths:
        end = start + int(length)
        score = x[start:end].astype(np.float32)
        if score.size == 0:
            start = end
            continue
        score = _moving_average_1d(score, smooth)
        if baseline_mix > 0:
            base_score = _video_baseline_suppress(
                score,
                q=baseline_q,
                strength=baseline_strength,
                rescale=baseline_rescale,
            )
            score = (1.0 - baseline_mix) * score + baseline_mix * base_score
        out[start:end] = np.clip(score, 0.0, 1.0)
        start = end
    return out


def _safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.float32)
    y_score = np.asarray(y_score).astype(np.float32)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(auc(fpr, tpr))


def _safe_far(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float32)
    if y_true.size == 0:
        return float("nan")
    pred = y_score.copy()
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1
    neg_mask = y_true == 0
    denom = int(np.sum(neg_mask))
    if denom <= 0:
        return float("nan")
    fp = int(np.sum((pred.astype(np.int32) == 1) & neg_mask))
    return float(fp) / float(denom)


def _safe_eer(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.float32)
    y_score = np.asarray(y_score).astype(np.float32)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    return float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))


def _profile_labels(values):
    y = np.asarray(values).astype(np.float32).reshape(-1)
    total = int(y.size)
    if total <= 0:
        return {"total": 0, "pos": 0, "neg": 0, "single_class": True}
    pos = int(np.sum(y >= 0.5))
    neg = int(total - pos)
    return {"total": total, "pos": pos, "neg": neg, "single_class": bool(pos == 0 or neg == 0)}


def _metrics_from_tokens(token_scores, gt, video_lengths):
    pred = np.repeat(token_scores.astype(np.float32), 16)
    valid_len = min(len(gt), len(pred))
    gt = gt[:valid_len]
    pred = pred[:valid_len]

    abnormal_pred = np.zeros(0, dtype=np.float32)
    abnormal_gt = np.zeros(0, dtype=np.float32)
    if video_lengths and sum(video_lengths) * 16 <= valid_len:
        frame_start = 0
        for length in video_lengths:
            frame_end = frame_start + int(length) * 16
            gt_single = gt[frame_start:frame_end]
            pred_single = pred[frame_start:frame_end]
            if np.sum(gt_single) > 0:
                abnormal_pred = np.concatenate((abnormal_pred, pred_single))
                abnormal_gt = np.concatenate((abnormal_gt, gt_single))
            frame_start = frame_end
    elif np.sum(gt) > 0:
        abnormal_pred = pred.copy()
        abnormal_gt = gt.copy()

    all_profile = _profile_labels(gt)
    abn_profile = _profile_labels(abnormal_gt)
    return {
        "auc_all": float(_safe_auc(gt, pred)),
        "auc_abn": float(_safe_auc(abnormal_gt, abnormal_pred)),
        "far_all": float(_safe_far(gt, pred, threshold=0.5)),
        "far_abn": float(_safe_far(abnormal_gt, abnormal_pred, threshold=0.5)),
        "eer": float(_safe_eer(gt, pred)),
        "stats_all_total": all_profile["total"],
        "stats_all_pos": all_profile["pos"],
        "stats_all_neg": all_profile["neg"],
        "stats_all_single_class": all_profile["single_class"],
        "stats_abn_total": abn_profile["total"],
        "stats_abn_pos": abn_profile["pos"],
        "stats_abn_neg": abn_profile["neg"],
        "stats_abn_single_class": abn_profile["single_class"],
        "score_mean": float(np.mean(pred)) if pred.size else 0.0,
        "score_p95": float(np.percentile(pred, 95.0)) if pred.size else 0.0,
        "score_max": float(np.max(pred)) if pred.size else 0.0,
    }


def _rank01(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size <= 1:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    order = np.argsort(np.argsort(x))
    return (order.astype(np.float32) / float(max(1, x.size - 1))).astype(np.float32)


def _per_video_transform(x, video_lengths, fn):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if not video_lengths:
        return fn(x)
    out = np.zeros_like(x)
    start = 0
    for length in video_lengths:
        end = min(len(x), start + int(length))
        if end > start:
            out[start:end] = fn(x[start:end])
        start = end
    if start < len(x):
        out[start:] = fn(x[start:])
    return out.astype(np.float32)


def _build_fusion_aux(recon, mem, video_lengths):
    recon_rank = _per_video_transform(recon, video_lengths, _rank01)
    mem_rank = _per_video_transform(mem, video_lengths, _rank01)
    return np.clip(0.5 * (1.0 - recon_rank) + 0.5 * (1.0 - mem_rank), 0.0, 1.0).astype(np.float32)


def _gate_quantile_per_video(aux, video_lengths, q):
    def gate_one(v):
        if len(v) == 0:
            return v.astype(np.float32)
        threshold = np.percentile(v, float(q))
        y = np.maximum(0.0, v - threshold)
        hi = float(np.max(y)) if len(y) else 0.0
        if hi > 1e-8:
            y = y / hi
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    return _per_video_transform(aux, video_lengths, gate_one)


def _collect_standard_scores(model, loader, args, device):
    score_mode = str(getattr(args, "concat_test_score_mode", "first")).lower()
    rows = []
    for _, input_data in enumerate(loader):
        input_data = input_data.to(device)
        with torch.no_grad():
            logits = model(input_data, None, None, is_training=False)
        rows.append(_reduce_logits(logits.detach().cpu().numpy(), mode=score_mode))
    return np.concatenate(rows, axis=0).astype(np.float32) if rows else np.zeros(0, dtype=np.float32)


def _collect_fusion_scores(model, loader, args, device):
    score_mode = str(getattr(args, "concat_test_score_mode", "first")).lower()
    student_rows, recon_rows, mem_rows = [], [], []
    for _, input_data in enumerate(loader):
        input_data = input_data.to(device)
        with torch.no_grad():
            student = model.scorer(input_data, is_training=False)
            if student.dim() >= 3 and student.shape[1] == model.cross_clip:
                student = model.slide_score(student)
            recon = model._infer_recon_scores(input_data)
            mem = model._infer_memory_novelty_scores(input_data)
        student_rows.append(_reduce_logits(student.detach().cpu().numpy(), mode=score_mode))
        recon_rows.append(_reduce_logits(recon.detach().cpu().numpy(), mode=score_mode))
        mem_rows.append(_reduce_logits(mem.detach().cpu().numpy(), mode=score_mode))
    return (
        np.concatenate(student_rows, axis=0).astype(np.float32),
        np.concatenate(recon_rows, axis=0).astype(np.float32),
        np.concatenate(mem_rows, axis=0).astype(np.float32),
    )


def _print_metrics(task, mode, ckpt_path, out_dir, metrics):
    print("task:", task)
    print("eval_mode:", mode)
    print("ckpt:", _safe_label(ckpt_path))
    for key in [
        "auc_all",
        "auc_abn",
        "far_all",
        "far_abn",
        "eer",
        "stats_all_total",
        "stats_all_pos",
        "stats_all_neg",
        "stats_all_single_class",
        "stats_abn_total",
        "stats_abn_pos",
        "stats_abn_neg",
        "stats_abn_single_class",
        "score_mean",
        "score_p95",
        "score_max",
    ]:
        print("{}: {}".format(key, metrics[key]))
    print("out_dir:", _safe_label(out_dir))


def _save_outputs(args, metrics, extra):
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    payload = {
        "task": args.task,
        "eval_mode": args.eval_mode,
        "metrics": metrics,
        "checkpoint": _safe_label(args.ckpt_path),
        "dataset": args.dataset_name,
    }
    payload.update(extra)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fields = [
        "task",
        "eval_mode",
        "auc_all",
        "auc_abn",
        "far_all",
        "far_abn",
        "eer",
        "stats_all_total",
        "stats_all_pos",
        "stats_all_neg",
        "stats_all_single_class",
        "stats_abn_total",
        "stats_abn_pos",
        "stats_abn_neg",
        "stats_abn_single_class",
        "score_mean",
        "score_p95",
        "score_max",
        "ckpt",
    ]
    row = {"task": args.task, "eval_mode": args.eval_mode, "ckpt": _safe_label(args.ckpt_path)}
    row.update(metrics)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def main():
    parser = options.parser
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--eval_mode", type=str, default="standard", choices=["standard", "ucf_inc_fusion"])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--dil_root", type=str, required=True)
    parser.add_argument("--dil_dataset", type=str, required=True, choices=["ucf", "sh"])
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--batch_eval", type=int, default=128)
    args = parser.parse_args()

    args.dataset_name = normalize_dataset_name(args.dataset_name)
    args.data_mode = "concat"
    args.concat_test_use_cross = False
    args.concat_test_score_mode = "first"
    if int(getattr(args, "clip_len", 10)) != 10:
        args.clip_len = 10

    _prepare_dil_args(args)

    if torch.cuda.is_available():
        cuda_idx = _resolve_cuda_index(args.device)
        device = torch.device("cuda:{}".format(cuda_idx))
        torch.cuda.set_device(cuda_idx)
    else:
        device = torch.device("cpu")

    model = _load_eval_model(args, device)
    dataset = Dataset_Con_all_feedback_XD(args, test_mode=True)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_eval),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    gt = np.load(args.eval_gt_npy)

    if args.eval_mode == "ucf_inc_fusion":
        student, recon, mem = _collect_fusion_scores(model, loader, args, device)
        video_lengths = _load_video_token_lengths(args, token_count=len(student))
        fusion_args = args
        fusion_args.use_eval_postproc = True
        fusion_args.test_smooth_kernel = FINAL_POST_SMOOTH
        fusion_args.test_video_baseline_mix = FINAL_POST_MIX
        fusion_args.test_video_baseline_q = FINAL_POST_Q
        fusion_args.test_video_baseline_strength = FINAL_POST_STRENGTH
        fusion_args.test_video_baseline_rescale = True
        aux = _build_fusion_aux(recon, mem, video_lengths)
        gate = _gate_quantile_per_video(aux, video_lengths, FINAL_GATE_Q)
        tokens = np.clip(student * (1.0 + FINAL_FUSION_LAMBDA * gate), 0.0, 1.0).astype(np.float32)
        tokens = _postprocess_tokens(tokens, fusion_args, video_lengths=video_lengths)
        extra = {
            "fusion_lambda": FINAL_FUSION_LAMBDA,
            "fusion_gate_q": FINAL_GATE_Q,
            "fusion_aux_mode": FINAL_AUX_MODE,
            "test_smooth_kernel": FINAL_POST_SMOOTH,
            "test_video_baseline_mix": FINAL_POST_MIX,
            "test_video_baseline_q": FINAL_POST_Q,
            "test_video_baseline_strength": FINAL_POST_STRENGTH,
        }
    else:
        tokens = _collect_standard_scores(model, loader, args, device)
        video_lengths = _load_video_token_lengths(args, token_count=len(tokens))
        tokens = _postprocess_tokens(tokens, args, video_lengths=video_lengths)
        extra = {
            "test_smooth_kernel": int(getattr(args, "test_smooth_kernel", 1)),
            "test_video_baseline_mix": float(getattr(args, "test_video_baseline_mix", 0.0)),
            "test_video_baseline_q": float(getattr(args, "test_video_baseline_q", 55.0)),
            "test_video_baseline_strength": float(getattr(args, "test_video_baseline_strength", 0.0)),
        }

    metrics = _metrics_from_tokens(tokens, gt, video_lengths)
    _save_outputs(args, metrics, extra)
    _print_metrics(args.task, args.eval_mode, args.ckpt_path, os.path.dirname(args.output_json), metrics)


if __name__ == "__main__":
    main()
