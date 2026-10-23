"""
heldout_evaluation.py

Scores several models on data they were NOT trained on:
  gnn      Hetero-GNN (trained from scratch with the fine-tuning recipe, so no saved checkpoint is reused)
  xgb      XGBoost on each node's own 8 features (same as benchmark_baselines.py)
  xgb_nbr  XGBoost on own features plus the mean features of each node's neighbours
  ridge    Ridge regression on own features (L2-penalised logistic regression for failure and churn)
  ols      Multiple linear regression on own features (logistic regression with almost no penalty for failure and churn)
  ridge_poly  (optional, not run by default) Ridge on own features plus all pairwise products of them

Linear models can only add up effects. Several labels here contain products (for example power = rated power x
season stress), so ridge_poly checks whether the missing products are what holds the plain linear models back.

Two kinds of split:
  months : train on 8 months, test on the other 4 (default test months 3, 6, 9, 12)
  nodes  : train on 80% of the locations / devices / residents / parts, test on the other 20%
           (test nodes are still part of the graph, so this is the usual "transductive" node split)

Seeds: --seeds 42,43,44 repeats everything with each seed. The seed changes the label noise, the node split and the
model initialisation (not the 100 customer profiles sampled from Online Retail II). With several seeds the script
also prints the mean and standard deviation of the test scores.

Caveats: the labels are generated from rules in data_pipeline.py plus noise, and there are only 10 locations,
40 devices, 100 residents and 20 parts, so the node split has very few test rows (see the n_test columns).

Run on the lab server:
    cd hvac_gml_project
    python heldout_evaluation.py                                         # both splits, seed 42, 5 models
    python heldout_evaluation.py --seeds 42,43,44 --out heldout_5models_main.csv
    python heldout_evaluation.py --split months --test-months 10,11,12 --seeds 42,43,44 --out heldout_5models_forecast.csv
    python heldout_evaluation.py --models ridge,ols,ridge_poly           # fast: only the linear models
Output: artifacts/<out> (default heldout_evaluation.csv), artifacts/<out>_summary.csv when several seeds are used,
and a table on screen.
"""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from config import Config
from data_pipeline import HVACGraphDatasetBuilder
from fine_tune_v100 import UncertaintyMultiTaskLoss, load_empirical_fine_tuning_tables
from hetero_gnn_model import MultiTaskHVACHeteroGNN
from telemetry_premodel import compute_monthly_telemetry_scores

NODE_TYPES = ["location", "device", "resident", "part"]
# task name -> (node type, target attribute)
TASKS = {
    "demand": ("location", "y_demand"),
    "power": ("device", "y_power"),
    "failure": ("device", "y_failure"),
    "churn": ("resident", "y_sleeper"),
    "inventory": ("part", "y_inventory"),
}
FAIL_THRESHOLD, CHURN_THRESHOLD = 0.40, 0.45   # same thresholds as evaluate_model_metrics()


# ----------------------------------------------------------------------------- data
def build_monthly_graphs(builder, tables, power_profiles):
    """The same 12 monthly graphs that fine_tune_v100.py builds (kept on the CPU)."""
    df_res, df_loc, df_dev, df_part = tables[:4]
    graphs = {}
    for m in Config.MONTHS:
        t_scores = compute_monthly_telemetry_scores(df_dev, month=m, device="cpu")
        if m in power_profiles:
            t_scores["power_kw"] = np.clip(t_scores["power_kw"] * 0.5 + power_profiles[m]["mean_power_kw"] * 0.7, 0.4, 4.2)
        _, regime_id, _, _ = Config.MONTH_REGIME[m]
        if regime_id == 2:
            t_scores["sensor_fail_prob"] = np.clip(t_scores["sensor_fail_prob"] * 1.30, 0.04, 0.95)
        elif regime_id == 1:
            t_scores["sensor_fail_prob"] = np.clip(t_scores["sensor_fail_prob"] * 1.25, 0.03, 0.95)
        g = builder.build_monthly_hetero_data(month=m, tables=tables, telemetry_scores=t_scores)

        rec_days = 365.0 * (1.0 - df_res["base_app_engagement"].values)
        off_season = 1.0 if regime_id in [0, 2] else 0.5
        churn = ((rec_days > 90) & (np.random.rand(len(df_res)) < 0.65 * off_season)).astype(np.float32)
        g["resident"].y_sleeper = torch.tensor(churn, dtype=torch.float).unsqueeze(-1)
        graphs[m] = g
    return graphs


def make_split(kind, graphs, test_months, seed):
    """Boolean masks train[(month, node_type)] and test[(month, node_type)] over the nodes of that type."""
    sizes = {t: graphs[1][t].x.shape[0] for t in NODE_TYPES}
    rng = np.random.default_rng(seed)
    node_is_test = {t: np.zeros(n, dtype=bool) for t, n in sizes.items()}
    if kind == "nodes":
        for t, n in sizes.items():
            node_is_test[t][rng.permutation(n)[: max(1, int(round(0.2 * n)))]] = True
    train, test = {}, {}
    for m in Config.MONTHS:
        for t, n in sizes.items():
            if kind == "months":
                train[(m, t)] = np.full(n, m not in test_months)
                test[(m, t)] = np.full(n, m in test_months)
            else:
                train[(m, t)] = ~node_is_test[t]
                test[(m, t)] = node_is_test[t]
    return train, test


# ----------------------------------------------------------------------------- scoring
def collect(graphs, masks, predict):
    """Stack predictions and targets over all (month, node type) rows selected by `masks`."""
    ys = {k: [] for k in TASKS}
    ps = {k: [] for k in TASKS}
    for m in Config.MONTHS:
        if not any(masks[(m, t)].any() for t in NODE_TYPES):
            continue
        pred = predict(m)   # dict task -> numpy array over ALL nodes of that task's node type
        for task, (ntype, attr) in TASKS.items():
            sel = masks[(m, ntype)]
            if sel.any():
                ys[task].append(graphs[m][ntype][attr].numpy().flatten()[sel])
                ps[task].append(pred[task][sel])
    return {k: np.concatenate(v) for k, v in ys.items()}, {k: np.concatenate(v) for k, v in ps.items()}


def score(y, p):
    return {
        "demand_mse": mean_squared_error(y["demand"], p["demand"]),
        "power_mse": mean_squared_error(y["power"], p["power"]),
        "inventory_mse": mean_squared_error(y["inventory"], p["inventory"]),
        "failure_f1": f1_score(y["failure"].astype(int), (p["failure"] > FAIL_THRESHOLD).astype(int), zero_division=0),
        "churn_f1": f1_score(y["churn"].astype(int), (p["churn"] > CHURN_THRESHOLD).astype(int), zero_division=0),
        "n_device_rows": len(y["failure"]),
        "n_resident_rows": len(y["churn"]),
        "failure_positive_rate": float(np.mean(y["failure"])),
        "churn_positive_rate": float(np.mean(y["churn"])),
    }


def reference_scores(y_train, y_test):
    """Dumb baselines on the test rows: predict the training mean; flag everything as positive."""
    out = {}
    for task in ("demand", "power", "inventory"):
        out[f"{task}_mse"] = float(np.mean((y_test[task] - y_train[task].mean()) ** 2))
    out["failure_f1"] = f1_score(y_test["failure"].astype(int), np.ones(len(y_test["failure"]), dtype=int), zero_division=0)
    out["churn_f1"] = f1_score(y_test["churn"].astype(int), np.ones(len(y_test["churn"]), dtype=int), zero_division=0)
    return out


# ----------------------------------------------------------------------------- GNN
def train_and_predict_gnn(graphs, train, epochs, lr, seed):
    device = torch.device(Config.DEVICE)
    torch.manual_seed(seed)
    gd = {m: copy.deepcopy(g).to(device) for m, g in graphs.items()}   # .to() works in place, so copy first
    model = MultiTaskHVACHeteroGNN(hidden_channels=Config.HIDDEN_CHANNELS, out_channels=Config.OUT_CHANNELS,
                                   num_layers=Config.NUM_GNN_LAYERS, dropout=0.15).to(device)
    with torch.no_grad():   # the SAGEConv layers create their weights on the first forward pass
        model(gd[1].x_dict, gd[1].edge_index_dict)
    unc = UncertaintyMultiTaskLoss(num_tasks=4).to(device)
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": lr, "weight_decay": 1e-4},
                             {"params": unc.parameters(), "lr": 5e-3, "weight_decay": 0.0}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    huber_big, huber_small = nn.SmoothL1Loss(beta=4.0), nn.SmoothL1Loss(beta=0.5)
    bce_fail = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5], device=device))
    bce_churn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.8], device=device))
    train_t = {k: torch.from_numpy(v).to(device) for k, v in train.items()}
    months = [m for m in Config.MONTHS if any(train[(m, t)].any() for t in NODE_TYPES)]

    for _ in range(epochs):
        model.train()
        for m in months:
            g = gd[m]
            mk = {t: train_t[(m, t)] for t in NODE_TYPES}
            opt.zero_grad()
            out = model(g.x_dict, g.edge_index_dict)
            l_dem = huber_big(out["demand_pred"][mk["location"]], g["location"].y_demand[mk["location"]])
            l_pow = huber_small(out["power_pred"][mk["device"]], g["device"].y_power[mk["device"]])
            l_fail = bce_fail(out["failure_logits"][mk["device"]], g["device"].y_failure[mk["device"]])
            l_churn = bce_churn(out["sleeper_logits"][mk["resident"]], g["resident"].y_sleeper[mk["resident"]])
            l_inv = huber_big(out["inventory_pred"][mk["part"]], g["part"].y_inventory[mk["part"]])
            loss = unc(l_dem, l_pow + l_fail, l_churn, l_inv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()

    model.eval()

    def predict(m):
        with torch.no_grad():
            out = model(gd[m].x_dict, gd[m].edge_index_dict)
        return {
            "demand": out["demand_pred"].cpu().numpy().flatten(),
            "power": out["power_pred"].cpu().numpy().flatten(),
            "failure": torch.sigmoid(out["failure_logits"]).cpu().numpy().flatten(),
            "churn": torch.sigmoid(out["sleeper_logits"]).cpu().numpy().flatten(),
            "inventory": out["inventory_pred"].cpu().numpy().flatten(),
        }
    return predict


# ----------------------------------------------------------------------------- XGBoost
def node_features(graphs, with_neighbours):
    """feats[m][node_type] -> numpy matrix. With neighbours: add the mean feature vector of the neighbours
    for every relation that points at this node type."""
    feats = {}
    for m, g in graphs.items():
        parts = {t: [g[t].x.numpy()] for t in NODE_TYPES}
        if with_neighbours:
            for (src, rel, dst) in sorted(g.edge_index_dict.keys()):
                ei = g.edge_index_dict[(src, rel, dst)]
                xs = g[src].x
                n_dst = g[dst].x.shape[0]
                sums = torch.zeros(n_dst, xs.shape[1]).index_add_(0, ei[1], xs[ei[0]])
                deg = torch.zeros(n_dst).index_add_(0, ei[1], torch.ones(ei.shape[1]))
                parts[dst].append((sums / deg.clamp(min=1).unsqueeze(1)).numpy())
        feats[m] = {t: np.hstack(v) for t, v in parts.items()}
    return feats


def make_gbm(kind, seed, scale_pos_weight=1.0):
    if HAS_XGB:
        common = dict(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=seed)
        if kind == "reg":
            return xgb.XGBRegressor(**common)
        return xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **common)
    if kind == "reg":
        return HistGradientBoostingRegressor(max_iter=100, random_state=seed)
    return HistGradientBoostingClassifier(max_iter=100, random_state=seed)


def train_and_predict_xgb(graphs, train, with_neighbours, seed):
    feats = node_features(graphs, with_neighbours)
    models = {}
    for task, (ntype, attr) in TASKS.items():
        X, y = [], []
        for m in Config.MONTHS:
            sel = train[(m, ntype)]
            if sel.any():
                X.append(feats[m][ntype][sel])
                y.append(graphs[m][ntype][attr].numpy().flatten()[sel])
        X, y = np.vstack(X), np.concatenate(y)
        if task in ("failure", "churn"):
            model = make_gbm("clf", seed, 2.5 if task == "failure" else 1.8)
        else:
            model = make_gbm("reg", seed)
        models[task] = model.fit(X, y)

    def predict(m):
        out = {}
        for task, (ntype, _) in TASKS.items():
            X = feats[m][ntype]
            if task in ("failure", "churn"):
                out[task] = models[task].predict_proba(X)[:, 1]
            elif task == "inventory":
                out[task] = np.clip(models[task].predict(X), 0, None)
            else:
                out[task] = models[task].predict(X)
        return out
    return predict


# ----------------------------------------------------------------------------- linear models
def train_and_predict_linear(graphs, train, regularised, polynomial, ridge_alpha):
    """Ridge (regularised=True) or multiple linear regression (False) on each node's own features.
    Regression tasks use Ridge / LinearRegression. Failure and churn use logistic regression (C=1 for ridge,
    C=1e6 = almost no penalty for the plain version) with the same positive-class weights as XGBoost.
    polynomial=True adds all pairwise products of the features first."""
    feats = node_features(graphs, False)
    models = {}
    for task, (ntype, attr) in TASKS.items():
        X, y = [], []
        for m in Config.MONTHS:
            sel = train[(m, ntype)]
            if sel.any():
                X.append(feats[m][ntype][sel])
                y.append(graphs[m][ntype][attr].numpy().flatten()[sel])
        X, y = np.vstack(X), np.concatenate(y)
        steps = [PolynomialFeatures(degree=2, include_bias=False)] if polynomial else []
        steps.append(StandardScaler())
        if task in ("failure", "churn"):
            weight = 2.5 if task == "failure" else 1.8
            steps.append(LogisticRegression(C=1.0 if regularised else 1e6, class_weight={0: 1.0, 1: weight}, max_iter=5000))
            y = y.astype(int)
        else:
            steps.append(Ridge(alpha=ridge_alpha) if regularised else LinearRegression())
        models[task] = make_pipeline(*steps).fit(X, y)

    def predict(m):
        out = {}
        for task, (ntype, _) in TASKS.items():
            X = feats[m][ntype]
            if task in ("failure", "churn"):
                out[task] = models[task].predict_proba(X)[:, 1]
            elif task == "inventory":
                out[task] = np.clip(models[task].predict(X), 0, None)
            else:
                out[task] = models[task].predict(X)
        return out
    return predict


# ----------------------------------------------------------------------------- main
METRICS = ["demand_mse", "power_mse", "inventory_mse", "failure_f1", "churn_f1"]
MODEL_NAMES = {
    "gnn": "Hetero-GNN",
    "xgb": "XGBoost (own features)",
    "xgb_nbr": "XGBoost (+ neighbour features)",
    "ridge": "Ridge regression (own features)",
    "ols": "Multiple linear regression (own features)",
    "ridge_poly": "Ridge + pairwise products (own features)",
}


def main():
    ap = argparse.ArgumentParser(description="Held-out evaluation: GNN vs XGBoost vs linear models")
    ap.add_argument("--split", choices=["months", "nodes", "both"], default="both")
    ap.add_argument("--test-months", default="3,6,9,12", help="comma-separated months held out in the months split")
    ap.add_argument("--models", default="gnn,xgb,xgb_nbr,ridge,ols",
                    help="comma-separated subset of: " + ",".join(MODEL_NAMES))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=Config.SEED)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds, e.g. 42,43,44 (overrides --seed)")
    ap.add_argument("--out", default="heldout_evaluation.csv", help="file name inside artifacts/")
    args = ap.parse_args()
    test_months = [int(x) for x in args.test_months.split(",")]
    wanted = [x.strip() for x in args.models.split(",")]
    unknown = [x for x in wanted if x not in MODEL_NAMES]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; choose from {list(MODEL_NAMES)}")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    print(f"Device: {Config.DEVICE} | XGBoost available: {HAS_XGB} | models: {wanted} | seeds: {seeds} | "
          f"test months (months split): {test_months}")

    rows = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        builder = HVACGraphDatasetBuilder(seed=seed)
        tables, power_profiles = load_empirical_fine_tuning_tables(builder)
        graphs = build_monthly_graphs(builder, tables, power_profiles)

        for kind in (["months", "nodes"] if args.split == "both" else [args.split]):
            print("\n" + "=" * 70 + f"\n SEED {seed} | SPLIT: {kind}\n" + "=" * 70)
            train, test = make_split(kind, graphs, test_months, seed)
            t0 = time.time()
            predictors = {}
            for key in wanted:
                if key == "gnn":
                    predictors[MODEL_NAMES[key]] = train_and_predict_gnn(graphs, train, args.epochs, args.lr, seed)
                elif key == "xgb":
                    predictors[MODEL_NAMES[key]] = train_and_predict_xgb(graphs, train, False, seed)
                elif key == "xgb_nbr":
                    predictors[MODEL_NAMES[key]] = train_and_predict_xgb(graphs, train, True, seed)
                else:
                    predictors[MODEL_NAMES[key]] = train_and_predict_linear(
                        graphs, train, regularised=key.startswith("ridge"), polynomial=(key == "ridge_poly"),
                        ridge_alpha=args.ridge_alpha)
            print(f"trained {len(predictors)} models in {time.time() - t0:.1f}s")
            zeros = lambda m: {k: np.zeros(graphs[m][TASKS[k][0]].x.shape[0]) for k in TASKS}
            y_tr_ref, _ = collect(graphs, train, zeros)
            y_te_ref, _ = collect(graphs, test, zeros)
            ref = reference_scores(y_tr_ref, y_te_ref)
            rows.append({"seed": seed, "split": kind, "model": "Reference: predict train mean / flag everything",
                         "data": "test", **ref})
            for name, predict in predictors.items():
                for data_name, masks in (("train", train), ("test", test)):
                    y, p = collect(graphs, masks, predict)
                    rows.append({"seed": seed, "split": kind, "model": name, "data": data_name, **score(y, p)})

    df = pd.DataFrame(rows)
    os.makedirs(Config.ARTIFACT_DIR, exist_ok=True)
    out_path = os.path.join(Config.ARTIFACT_DIR, args.out)
    df.to_csv(out_path, index=False)
    show = ["seed", "split", "model", "data"] + METRICS + ["n_device_rows", "n_resident_rows"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print("\n" + df[show].to_string(index=False))
    print(f"\nSaved {out_path}")

    if len(seeds) > 1:
        test_rows = df[df["data"] == "test"]
        summary = test_rows.groupby(["split", "model"])[METRICS].agg(["mean", "std"])
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        summary = summary.reset_index()
        summary_path = out_path.replace(".csv", "_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\nTEST scores, mean and standard deviation over seeds {seeds}:")
        for kind, grp in summary.groupby("split"):
            print(f"\n  split: {kind}")
            table = pd.DataFrame({"model": grp["model"]})
            for c in METRICS:
                table[c] = [f"{m:.4g} +/- {s:.2g}" for m, s in zip(grp[f"{c}_mean"], grp[f"{c}_std"])]
            print(table.to_string(index=False))
        print(f"\nSaved {summary_path}")
    print("Compare 'train' with 'test' rows: a big gap means the model memorised the training data.")


if __name__ == "__main__":
    main()
