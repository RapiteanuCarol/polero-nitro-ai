import glob
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss


def load_classes():
    with open("data/processed/classes_order.txt", "r", encoding="utf-8") as f:
        classes = [line.strip().split(": ", 1)[1] for line in f if line.strip()]
    return classes, {c: i for i, c in enumerate(classes)}


def safe_log_loss(y_true, probs):
    p = np.clip(probs, 1e-7, 1.0 - 1e-7)
    p = p / p.sum(axis=1, keepdims=True)
    return log_loss(y_true, p)


def fast_macro_f1(y_true, y_pred, n_cls):
    total = 0.0
    for c in range(n_cls):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        if denom > 0:
            total += (2.0 * tp) / denom
    return total / n_cls


def fit_class_multipliers(probs, y_true, n_cls, passes=5):
    counts = np.bincount(y_true, minlength=n_cls).astype(np.float64)
    weights = (counts.mean() / np.maximum(counts, 1.0)) ** 0.35
    scaled = probs * weights
    candidates = np.geomspace(0.05, 20.0, 61)

    for _ in range(passes):
        for c in range(n_cls):
            other_cols = [j for j in range(n_cls) if j != c]
            other_scaled = scaled[:, other_cols]
            best_other_idx = np.argmax(other_scaled, axis=1)
            best_other_cls = np.array(other_cols)[best_other_idx]
            best_other_val = other_scaled[np.arange(len(y_true)), best_other_idx]

            best_w = weights[c]
            best_score = fast_macro_f1(y_true, scaled.argmax(axis=1), n_cls)

            for w in candidates:
                preds = np.where(probs[:, c] * w > best_other_val, c, best_other_cls)
                score = fast_macro_f1(y_true, preds, n_cls)
                if score > best_score + 1e-6:
                    best_score = score
                    best_w = w

            weights[c] = best_w
            scaled[:, c] = probs[:, c] * best_w

    return weights


def optimize_blend_weights(oof_list, y_true):
    n_models = len(oof_list)
    if n_models == 1:
        return np.array([1.0])

    clipped = [np.clip(p, 1e-7, 1.0 - 1e-7) for p in oof_list]

    def loss_fn(w):
        w = np.array(w)
        w = w / np.sum(w)
        blend = sum(wi * pi for wi, pi in zip(w, clipped))
        return safe_log_loss(y_true, blend)

    init_w = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(loss_fn, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x / np.sum(res.x)


def get_alignment_indices(target_ids, id_csv_pattern):
    matches = glob.glob(f"**/{id_csv_pattern}", recursive=True)
    if not matches:
        return None
    df_ids = pd.read_csv(matches[0])
    id_col = "datapointID" if "datapointID" in df_ids.columns else df_ids.columns[0]
    src_ids = df_ids[id_col].values
    if len(src_ids) != len(target_ids) or np.array_equal(src_ids, target_ids):
        return None
    pos_map = {dp_id: idx for idx, dp_id in enumerate(src_ids)}
    return np.array([pos_map[dp_id] for dp_id in target_ids], dtype=int)


def discover_models(train_df, test_df, classes, y_all, exclude=None):
    exclude = set(exclude or [])
    n_train, n_test, n_classes = len(train_df), len(test_df), len(classes)

    train_align = get_alignment_indices(train_df["datapointID"].values, "train_prediction_ids.csv")
    test_align = get_alignment_indices(test_df["datapointID"].values, "test_prediction_ids.csv")
    bianca_models = {"lgb", "baseline"}

    alpha_classes = sorted(classes)
    alpha_to_canonical = [alpha_classes.index(c) for c in classes]

    models = {}
    for oof_path in sorted(glob.glob("predictions/oof_*.npy")):
        name = os.path.basename(oof_path)[len("oof_") : -len(".npy")]
        if name in exclude or name.startswith("s1_"):
            continue
        test_path = f"predictions/test_{name}.npy"
        if not os.path.exists(test_path):
            print(f"[SKIP] Lipseste {test_path}")
            continue

        oof = np.load(oof_path).astype(np.float64)
        test = np.load(test_path).astype(np.float64)

        if oof.shape != (n_train, n_classes) or test.shape != (n_test, n_classes):
            print(f"[SKIP] {name}: dimensiuni invalide OOF={oof.shape}, Test={test.shape}")
            continue
        if np.isnan(oof).any() or np.isnan(test).any():
            print(f"[SKIP] {name}: contine NaN")
            continue

        if name in bianca_models:
            if train_align is not None:
                oof = oof[train_align]
                print(f"[INFO] Realiniat randurile OOF pentru {name}")
            if test_align is not None:
                test = test[test_align]
                print(f"[INFO] Realiniat randurile Test pentru {name}")

        oof = oof / np.clip(oof.sum(axis=1, keepdims=True), 1e-12, None)
        test = test / np.clip(test.sum(axis=1, keepdims=True), 1e-12, None)

        ll_canonical = safe_log_loss(y_all, oof)
        oof_reordered = oof[:, alpha_to_canonical]
        ll_alpha = safe_log_loss(y_all, oof_reordered)

        if ll_alpha < ll_canonical - 0.2:
            oof = oof_reordered
            test = test[:, alpha_to_canonical]
            print(f"[INFO] Reordonat coloanele (alfabetic -> classes_order.txt) pentru {name}")

        models[name] = (oof, test)

    return models


def main():
    classes, label_to_id = load_classes()
    num_classes = len(classes)

    train_df = pd.read_csv("data/processed/train_context.csv")
    test_df = pd.read_csv("data/processed/test_context.csv")
    y_evasion = train_df["evasion_label"].map(label_to_id).values
    folds = train_df["fold"].values

    clarity_classes = sorted(train_df["clarity_label"].unique())
    clarity_to_id = {c: i for i, c in enumerate(clarity_classes)}
    y_clarity = train_df["clarity_label"].map(clarity_to_id).values
    num_clarity = len(clarity_classes)

    models = discover_models(train_df, test_df, classes, y_evasion, exclude=[])
    assert len(models) >= 1, "Nu s-a gasit niciun model valid in predictions/!"

    print("\n=== Modele incluse in Ensemble (Subtask 2 - Evasion) ===")
    names = list(models.keys())
    oof_list = [models[n][0] for n in names]
    test_list = [models[n][1] for n in names]

    for n, oof_m in zip(names, oof_list):
        raw_f1 = f1_score(y_evasion, oof_m.argmax(axis=1), average="macro")
        ll = safe_log_loss(y_evasion, oof_m)
        print(f"  - {n:<18} | OOF Macro-F1 brut: {raw_f1:.4f} | LogLoss: {ll:.4f}")

    oof_blend = np.zeros_like(oof_list[0])
    oof_ev_tuned_preds = np.zeros(len(y_evasion), dtype=int)
    oof_cl_raw_probs = np.zeros((len(y_clarity), num_clarity), dtype=np.float64)
    oof_cl_tuned_preds = np.zeros(len(y_clarity), dtype=int)

    fold_blend_weights = []
    fold_ev_multipliers = []
    fold_cl_multipliers = []

    for f in range(5):
        trn_mask = folds != f
        val_mask = folds == f

    
        w_models = optimize_blend_weights([p[trn_mask] for p in oof_list], y_evasion[trn_mask])
        fold_blend_weights.append(w_models)

        trn_blend = sum(wi * pi[trn_mask] for wi, pi in zip(w_models, oof_list))
        val_blend = sum(wi * pi[val_mask] for wi, pi in zip(w_models, oof_list))
        oof_blend[val_mask] = val_blend

       
        w_ev = fit_class_multipliers(trn_blend, y_evasion[trn_mask], num_classes)
        fold_ev_multipliers.append(w_ev)
        oof_ev_tuned_preds[val_mask] = (val_blend * w_ev).argmax(axis=1)

      
        proj_mat = np.zeros((num_classes, num_clarity), dtype=np.float64)
        for ev_idx in range(num_classes):
            sub_cl = y_clarity[trn_mask & (y_evasion == ev_idx)]
            if len(sub_cl) > 0:
                proj_mat[ev_idx] = np.bincount(sub_cl, minlength=num_clarity) / len(sub_cl)

        trn_cl_probs = trn_blend @ proj_mat
        val_cl_probs = val_blend @ proj_mat
        oof_cl_raw_probs[val_mask] = val_cl_probs

        w_cl = fit_class_multipliers(trn_cl_probs, y_clarity[trn_mask], num_clarity)
        fold_cl_multipliers.append(w_cl)
        oof_cl_tuned_preds[val_mask] = (val_cl_probs * w_cl).argmax(axis=1)

    mean_model_weights = np.mean(np.vstack(fold_blend_weights), axis=0)
    mean_model_weights /= np.sum(mean_model_weights)

    print("\n=== Ponderi optime pe modele ===")
    for n, w in zip(names, mean_model_weights):
        print(f"  - {n:<18}: {w:.4f}")

    s2_raw = f1_score(y_evasion, oof_blend.argmax(axis=1), average="macro")
    s2_tuned = f1_score(y_evasion, oof_ev_tuned_preds, average="macro")
    s1_raw = f1_score(y_clarity, oof_cl_raw_probs.argmax(axis=1), average="macro")
    s1_tuned = f1_score(y_clarity, oof_cl_tuned_preds, average="macro")

    print(f"\n[Subtask 2 - Evasion] OOF Macro-F1 brut: {s2_raw:.4f} -> calibrat: {s2_tuned:.4f}")
    print(f"[Subtask 1 - Clarity] OOF Macro-F1 brut: {s1_raw:.4f} -> calibrat: {s1_tuned:.4f}")
    print(f"Media OOF Macro-F1 (Subtask 1 + 2)      : {(s1_tuned + s2_tuned) / 2.0:.4f}")

    test_blend = sum(wi * pi for wi, pi in zip(mean_model_weights, test_list))
    mean_ev_mult = np.exp(np.mean(np.log(np.vstack(fold_ev_multipliers)), axis=0))
    test_ev_idx = (test_blend * mean_ev_mult).argmax(axis=1)
    evasion_preds = [classes[i] for i in test_ev_idx]

    full_proj_mat = np.zeros((num_classes, num_clarity), dtype=np.float64)
    for ev_idx in range(num_classes):
        sub_cl = y_clarity[y_evasion == ev_idx]
        if len(sub_cl) > 0:
            full_proj_mat[ev_idx] = np.bincount(sub_cl, minlength=num_clarity) / len(sub_cl)

    test_cl_probs = test_blend @ full_proj_mat
    mean_cl_mult = np.exp(np.mean(np.log(np.vstack(fold_cl_multipliers)), axis=0))
    test_cl_idx = (test_cl_probs * mean_cl_mult).argmax(axis=1)
    clarity_preds = [clarity_classes[i] for i in test_cl_idx]

    id_to_evasion = dict(zip(test_df["datapointID"], evasion_preds))
    id_to_clarity = dict(zip(test_df["datapointID"], clarity_preds))

    sub = pd.read_csv("data/raw/sample_output.csv")
    unique_subtasks = list(sub["subtaskID"].unique())

    if len(unique_subtasks) == 1:
        sub["answer"] = sub["datapointID"].map(id_to_evasion)
    else:
        s1_id = [s for s in unique_subtasks if "1" in str(s)][0]
        s2_id = [s for s in unique_subtasks if "2" in str(s)][0]
        sub.loc[sub["subtaskID"] == s1_id, "answer"] = sub.loc[sub["subtaskID"] == s1_id, "datapointID"].map(id_to_clarity)
        sub.loc[sub["subtaskID"] == s2_id, "answer"] = sub.loc[sub["subtaskID"] == s2_id, "datapointID"].map(id_to_evasion)

    assert not sub["answer"].isna().any(), "Exista datapointID-uri nemapate!"
    os.makedirs("submissions", exist_ok=True)
    out_path = "submissions/submission_ensemble.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSalvat {out_path} -> shape: {sub.shape}")


if __name__ == "__main__":
    main()
