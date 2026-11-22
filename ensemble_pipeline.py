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


def linear_blend(probs_list, weights):
    blend = sum(w * p for w, p in zip(weights, probs_list))
    return blend / np.clip(blend.sum(axis=1, keepdims=True), 1e-12, None)


def geometric_blend(probs_list, weights):
    log_p = sum(w * np.log(np.clip(p, 1e-7, 1.0)) for w, p in zip(weights, probs_list))
    log_p -= np.max(log_p, axis=1, keepdims=True)
    exp_p = np.exp(log_p)
    return exp_p / np.clip(exp_p.sum(axis=1, keepdims=True), 1e-12, None)


def optimize_blend_weights(oof_list, y_true, mode="linear"):
    n_models = len(oof_list)
    if n_models == 1:
        return np.array([1.0])

    blend_fn = linear_blend if mode == "linear" else geometric_blend

    def loss_fn(w):
        w = np.array(w)
        w = w / np.sum(w)
        return safe_log_loss(y_true, blend_fn(oof_list, w))

    init_w = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(loss_fn, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x / np.sum(res.x)


def fit_bayes_prior_alpha(probs, y_true, n_cls):
    """Un singur parametru scalar alpha (0..0.6) pentru a evita overfitting-ul pe clase rare."""
    counts = np.bincount(y_true, minlength=n_cls).astype(np.float64)
    priors = np.maximum(counts / counts.sum(), 1e-6)

    best_alpha = 0.0
    best_score = f1_score(y_true, probs.argmax(axis=1), average="macro")

    for alpha in np.linspace(0.0, 0.65, 66):
        adj = probs / (priors ** alpha)
        score = f1_score(y_true, adj.argmax(axis=1), average="macro")
        if score > best_score + 1e-6:
            best_score = score
            best_alpha = float(alpha)

    return best_alpha, priors


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
            continue

        oof = np.load(oof_path).astype(np.float64)
        test = np.load(test_path).astype(np.float64)

        if oof.shape != (n_train, n_classes) or test.shape != (n_test, n_classes):
            continue
        if np.isnan(oof).any() or np.isnan(test).any():
            continue

        if name in bianca_models:
            if train_align is not None:
                oof = oof[train_align]
            if test_align is not None:
                test = test[test_align]

        oof = oof / np.clip(oof.sum(axis=1, keepdims=True), 1e-12, None)
        test = test / np.clip(test.sum(axis=1, keepdims=True), 1e-12, None)

        if safe_log_loss(y_all, oof[:, alpha_to_canonical]) < safe_log_loss(y_all, oof) - 0.2:
            oof = oof[:, alpha_to_canonical]
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
    names = list(models.keys())
    oof_list = [models[n][0] for n in names]
    test_list = [models[n][1] for n in names]

    print("\n=== Modele incluse in Ensemble (Subtask 2 - Evasion) ===")
    for n, oof_m in zip(names, oof_list):
        raw_f1 = f1_score(y_evasion, oof_m.argmax(axis=1), average="macro")
        ll = safe_log_loss(y_evasion, oof_m)
        print(f"  - {n:<18} | OOF Macro-F1 brut: {raw_f1:.4f} | LogLoss: {ll:.4f}")

  
    results = {}
    for mode in ["linear", "geometric"]:
        blend_fn = linear_blend if mode == "linear" else geometric_blend
        oof_b = np.zeros_like(oof_list[0])
        oof_adj_b = np.zeros_like(oof_list[0])
        fold_w, fold_alphas = [], []

        for f in range(5):
            trn = folds != f
            val = folds == f
            w_f = optimize_blend_weights([p[trn] for p in oof_list], y_evasion[trn], mode=mode)
            fold_w.append(w_f)

            trn_p = blend_fn([p[trn] for p in oof_list], w_f)
            val_p = blend_fn([p[val] for p in oof_list], w_f)
            oof_b[val] = val_p

            alpha_f, priors_f = fit_bayes_prior_alpha(trn_p, y_evasion[trn], num_classes)
            fold_alphas.append(alpha_f)
            oof_adj_b[val] = val_p / (priors_f ** alpha_f)

        mean_w = np.mean(np.vstack(fold_w), axis=0)
        mean_w /= mean_w.sum()
        mean_alpha = float(np.mean(fold_alphas))

        f1_raw = f1_score(y_evasion, oof_b.argmax(axis=1), average="macro")
        f1_adj = f1_score(y_evasion, oof_adj_b.argmax(axis=1), average="macro")
        results[mode] = {
            "weights": mean_w,
            "alpha": mean_alpha,
            "oof_raw": oof_b,
            "oof_adj": oof_adj_b,
            "f1_raw": f1_raw,
            "f1_adj": f1_adj,
        }
        print(f"\n[{mode.upper()} BLEND] LogLoss={safe_log_loss(y_evasion, oof_b):.4f} | F1 brut={f1_raw:.4f} | F1 Bayes(alpha={mean_alpha:.2f})={f1_adj:.4f}")

   
    best_mode = max(results.keys(), key=lambda m: max(results[m]["f1_raw"], results[m]["f1_adj"]))
    best_res = results[best_mode]
    use_bayes_s2 = best_res["f1_adj"] > best_res["f1_raw"]
    s2_final_f1 = best_res["f1_adj"] if use_bayes_s2 else best_res["f1_raw"]
    oof_best_probs = best_res["oof_raw"]

    print(f"\n>>> Selectat pentru Subtask 2: {best_mode.upper()} + {'Bayes(alpha=' + str(round(best_res['alpha'], 2)) + ')' if use_bayes_s2 else 'argmax'}")
    print("=== Ponderi optime pe modele ===")
    for n, w in zip(names, best_res["weights"]):
        print(f"  - {n:<18}: {w:.4f}")

   
    oof_cl_raw = np.zeros((len(y_clarity), num_clarity), dtype=np.float64)
    oof_cl_adj = np.zeros((len(y_clarity), num_clarity), dtype=np.float64)
    cl_alphas = []

    for f in range(5):
        trn = folds != f
        val = folds == f
        proj = np.zeros((num_classes, num_clarity), dtype=np.float64)
        for ev_idx in range(num_classes):
            sub_cl = y_clarity[trn & (y_evasion == ev_idx)]
            if len(sub_cl) > 0:
                proj[ev_idx] = np.bincount(sub_cl, minlength=num_clarity) / len(sub_cl)

        trn_cl = oof_best_probs[trn] @ proj
        val_cl = oof_best_probs[val] @ proj
        oof_cl_raw[val] = val_cl

        alpha_cl, priors_cl = fit_bayes_prior_alpha(trn_cl, y_clarity[trn], num_clarity)
        cl_alphas.append(alpha_cl)
        oof_cl_adj[val] = val_cl / (priors_cl ** alpha_cl)

    s1_f1_raw = f1_score(y_clarity, oof_cl_raw.argmax(axis=1), average="macro")
    s1_f1_adj = f1_score(y_clarity, oof_cl_adj.argmax(axis=1), average="macro")
    use_bayes_s1 = s1_f1_adj > s1_f1_raw
    s1_final_f1 = s1_f1_adj if use_bayes_s1 else s1_f1_raw
    mean_cl_alpha = float(np.mean(cl_alphas))

    print(f"\n[Subtask 2 - Evasion] OOF Macro-F1 final: {s2_final_f1:.4f}")
    print(f"[Subtask 1 - Clarity] OOF Macro-F1 final: {s1_final_f1:.4f} (alpha={mean_cl_alpha:.2f}, Bayes={use_bayes_s1})")
    print(f"Media OOF Macro-F1 (Subtask 1 + 2)      : {(s1_final_f1 + s2_final_f1) / 2.0:.4f}")

  
    blend_fn = linear_blend if best_mode == "linear" else geometric_blend
    test_blend = blend_fn(test_list, best_res["weights"])

    if use_bayes_s2:
        counts_ev = np.bincount(y_evasion, minlength=num_classes).astype(np.float64)
        priors_ev = np.maximum(counts_ev / counts_ev.sum(), 1e-6)
        test_ev_scores = test_blend / (priors_ev ** best_res["alpha"])
    else:
        test_ev_scores = test_blend
    evasion_preds = [classes[i] for i in test_ev_scores.argmax(axis=1)]

    full_proj = np.zeros((num_classes, num_clarity), dtype=np.float64)
    for ev_idx in range(num_classes):
        sub_cl = y_clarity[y_evasion == ev_idx]
        if len(sub_cl) > 0:
            full_proj[ev_idx] = np.bincount(sub_cl, minlength=num_clarity) / len(sub_cl)

    test_cl_probs = test_blend @ full_proj
    if use_bayes_s1:
        counts_cl = np.bincount(y_clarity, minlength=num_clarity).astype(np.float64)
        priors_cl = np.maximum(counts_cl / counts_cl.sum(), 1e-6)
        test_cl_scores = test_cl_probs / (priors_cl ** mean_cl_alpha)
    else:
        test_cl_scores = test_cl_probs
    clarity_preds = [clarity_classes[i] for i in test_cl_scores.argmax(axis=1)]

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
    for st, grp in sub.groupby("subtaskID"):
        print(f"\nDistributie test subtaskID={st}:")
        print(grp["answer"].value_counts())


if __name__ == "__main__":
    main()
