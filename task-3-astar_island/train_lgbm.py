"""
Train LightGBM to predict terrain class distributions from spatial features.
Uses ground truth from completed rounds. Leave-one-round-out CV.

Usage:
  uv run python train_lgbm.py           # CV comparison
  uv run python train_lgbm.py --final   # Train final model
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
from client import AstarClient
from features import SeedAnalysis
from observation_store import ObservationStore

NUM_CLASSES = 6
TERRAIN_OCEAN = 10
TERRAIN_MOUNTAIN = 5

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def compute_pressure(settlements, h, w):
    pressure = np.zeros((h, w), dtype=np.float64)
    yy, xx = np.ogrid[0:h, 0:w]
    for s in settlements:
        d = np.maximum(np.abs(xx - s["x"]), np.abs(yy - s["y"])).astype(np.float64)
        pressure += np.exp(-d / 3.0)
    return pressure

FEATURE_NAMES = [
    "terrain", "dist_sett", "is_coastal", "has_adj_sett",
    "pressure", "sett_count3", "sett_count5",
    "obs_class", "n_obs",
    "nb_ocean", "nb_mountain", "nb_forest", "nb_plains",
    "nb_settlement", "nb_port", "nb_ruin",
    "cs_empty", "cs_sett", "cs_port", "cs_ruin", "cs_forest", "cs_mount",
]

def extract_round(client, round_id, obs=None):
    det = client.get_round_detail(round_id)
    h, w = det.height, det.width
    feats, tgts = [], []
    for si in range(det.seeds_count):
        state = det.initial_states[si]
        sa = SeedAnalysis(state["grid"], state["settlements"])
        gt = np.array(client.get_analysis(round_id, si)["ground_truth"])
        pressure = compute_pressure(state["settlements"], h, w)
        sc3 = np.zeros((h,w), dtype=np.int32)
        sc5 = np.zeros((h,w), dtype=np.int32)
        yy, xx = np.ogrid[0:h, 0:w]
        for s in state["settlements"]:
            d = np.maximum(np.abs(xx-s["x"]), np.abs(yy-s["y"]))
            sc3 += (d <= 3).astype(np.int32)
            sc5 += (d <= 5).astype(np.int32)
        grid = sa.grid
        padded = np.pad(grid, 1, mode="constant", constant_values=TERRAIN_OCEAN)
        if obs and obs.seeds_count > si:
            oc = obs.get_seed_counts(si)
            on = obs.get_seed_obs_counts(si)
            cc = np.zeros((h,w,NUM_CLASSES), dtype=np.float64)
            for oi in range(obs.seeds_count):
                if oi != si:
                    cc += obs.get_seed_counts(oi).astype(np.float64)
        else:
            oc = np.zeros((h,w,NUM_CLASSES))
            on = np.zeros((h,w))
            cc = np.zeros((h,w,NUM_CLASSES))
        for y in range(h):
            for x in range(w):
                t = int(grid[y,x])
                if t in (TERRAIN_OCEAN, TERRAIN_MOUNTAIN): continue
                nt = np.zeros(12, dtype=np.float32)
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        if dy==0 and dx==0: continue
                        tv = int(padded[y+1+dy, x+1+dx])
                        if tv < 12: nt[tv] += 1
                n = on[y,x]
                obsclass = int(np.argmax(oc[y,x])) if n > 0 else -1
                cs = cc[y,x]; css = cs.sum()
                csn = cs/css if css > 0 else np.zeros(NUM_CLASSES)
                feats.append([
                    t, float(sa.dist_to_settlement[y,x]), float(sa.coastal_mask[y,x]),
                    float(sa.dist_to_settlement[y,x]<=1.5), pressure[y,x],
                    float(sc3[y,x]), float(sc5[y,x]), float(obsclass), float(n),
                    nt[TERRAIN_OCEAN], nt[TERRAIN_MOUNTAIN], nt[4], nt[11],
                    nt[1], nt[2], nt[3],
                    csn[0], csn[1], csn[2], csn[3], csn[4], csn[5],
                ])
                tgts.append(gt[y,x].astype(np.float32))
    return np.array(feats, dtype=np.float32), np.array(tgts, dtype=np.float32)

def train_models(X, y, Xv=None, yv=None):
    models = []
    params = {
        "objective": "regression", "metric": "mae",
        "num_leaves": 63, "learning_rate": 0.05,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_child_samples": 50, "verbose": -1, "num_threads": 8,
    }
    for ci in range(NUM_CLASSES):
        td = lgb.Dataset(X, label=y[:,ci], feature_name=FEATURE_NAMES,
                         categorical_feature=["terrain","obs_class"], free_raw_data=False)
        if Xv is not None:
            vd = lgb.Dataset(Xv, label=yv[:,ci], reference=td, free_raw_data=False)
            m = lgb.train(params, td, 500, valid_sets=[vd],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        else:
            m = lgb.train(params, td, 300, callbacks=[lgb.log_evaluation(0)])
        models.append(m)
    return models

def predict_lgbm(models, X, floor=0.001):
    N = X.shape[0]
    pred = np.zeros((N, NUM_CLASSES), dtype=np.float64)
    for ci, m in enumerate(models):
        pred[:,ci] = m.predict(X)
    pred = np.clip(pred, floor, None)
    pred /= pred.sum(axis=1, keepdims=True)
    return pred

def entropy(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p>0, -p*np.log(p), 0).sum(axis=-1)

def kl_div(p, q):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p>0, p*np.log(p/np.maximum(q,1e-10)), 0).sum(axis=-1)

def score(gt, pred):
    e = entropy(gt); kl = kl_div(gt, pred); d = e > 1e-8
    if not d.any(): return 100.0
    wkl = (e[d]*kl[d]).sum() / e[d].sum()
    return max(0, min(100, 100*np.exp(-3*wkl)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    client = AstarClient()
    rounds = sorted([r for r in client.get_my_rounds() if r["status"]=="completed"],
                     key=lambda r: r["round_number"])
    log(f"Found {len(rounds)} completed rounds")
    data_dir = Path("data")
    obs_map = {f.stem.replace("obs_",""): f for f in data_dir.glob("obs_*.npz")}
    rd = {}
    for r in rounds:
        rn, rid = r["round_number"], r["id"]
        prefix = rid[:8]
        log(f"Extracting R{rn}...")
        obs = None
        if prefix in obs_map:
            try:
                obs = ObservationStore.load(str(data_dir/f"obs_{prefix}"))
                if sum(obs.get_seed_obs_counts(i).sum() for i in range(obs.seeds_count)) < 100:
                    obs = None
            except: obs = None
        try:
            X, y = extract_round(client, rid, obs)
            rd[rn] = {"X": X, "y": y, "rid": rid}
            log(f"  {X.shape[0]} cells, {X.shape[1]} features")
        except Exception as e:
            log(f"  SKIP: {e}")
    rns = sorted(rd.keys())
    log(f"\n{len(rns)} rounds ready")
    if args.final:
        log("=== Training FINAL model ===")
        Xa = np.concatenate([rd[r]["X"] for r in rns])
        ya = np.concatenate([rd[r]["y"] for r in rns])
        log(f"Training on {Xa.shape[0]} examples...")
        models = train_models(Xa, ya)
        md = Path("lgbm_models"); md.mkdir(exist_ok=True)
        for i, m in enumerate(models):
            m.save_model(str(md/f"class_{i}.txt"))
        log(f"Saved to {md}/")
        for i, m in enumerate(models):
            imp = sorted(zip(FEATURE_NAMES, m.feature_importance("gain")), key=lambda x:-x[1])[:5]
            log(f"  {['empty','sett','port','ruin','forest','mount'][i]}: {imp}")
        return
    log("\n=== Leave-One-Round-Out CV ===")
    log(f"{'Round':>6} {'LGBM':>6} {'Diff':>6}")
    log("-"*20)
    all_scores = []
    for hold in rns:
        train_rns = [r for r in rns if r != hold]
        Xt = np.concatenate([rd[r]["X"] for r in train_rns])
        yt = np.concatenate([rd[r]["y"] for r in train_rns])
        Xv, yv = rd[hold]["X"], rd[hold]["y"]
        models = train_models(Xt, yt, Xv, yv)
        pred = predict_lgbm(models, Xv)
        s = score(yv, pred)
        all_scores.append(s)
        log(f"R{hold:>4} {s:>6.2f}")
    log("-"*20)
    log(f"{'AVG':>6} {np.mean(all_scores):>6.2f}")
    log(f"{'MED':>6} {np.median(all_scores):>6.2f}")
    log(f"{'MIN':>6} {min(all_scores):>6.2f}")
    log(f"{'MAX':>6} {max(all_scores):>6.2f}")

if __name__ == "__main__":
    main()
