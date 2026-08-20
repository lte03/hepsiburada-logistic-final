import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

T0 = time.time()
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

DEMAND_XLSX = os.path.join(DATA, "teknofest26_gelismis.xlsx")
NETWORK_XLSX = os.path.join(DATA, "sehirler_arasi_lojistik.xlsx")
OUT_XLSX = os.path.join(BASE, "Talep-tahmini.xlsx")
META_JSON = os.path.join(BASE, "forecast_meta.json")

FORECAST_DATES = pd.date_range("2026-06-29", "2026-07-05", freq="D")
SLOTS = ["09:00", "17:00"]

# ---------------------------------------------------------------- takvim
HOLIDAYS = set(pd.to_datetime(x) for x in [
    "2026-01-01",                                # Yılbaşı
    "2026-03-19", "2026-03-20", "2026-03-21",    # Ramazan Bayramı
    "2026-04-23", "2026-05-01", "2026-05-19",    # Resmî tatiller
])
KURBAN_WEEK = set(pd.date_range("2026-05-25", "2026-05-31"))
ANOMALOUS_TRAIN_EXCLUDE = HOLIDAYS | KURBAN_WEEK | {pd.Timestamp("2026-06-28")}
# Lag/rolling pencerelerinden ayrıca ay sonları da düşülür (§2.2 tablo)
HOLIDAY_BLOCKS_END = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-21"),
                      pd.Timestamp("2026-04-23"), pd.Timestamp("2026-05-01"),
                      pd.Timestamp("2026-05-19"), pd.Timestamp("2026-05-31")]

# ---------------------------------------------------------------- korkuluk sabitleri (§10.1–10.3)
# V4: dtme0 override çarpanı sabit DEĞİL — {0.02, 0.05, 0.08} grid'i analog-fold
# backtest'iyle yarışır, kazanan k30* finalde kullanılır (§10.1 kararının veriyle kilidi)
CLIP_29JUN = (0.60, 0.80)       # dtme=1 gün toplamı bandı (saf DOW tabanına oran)
CLIP_1JUL_17 = (1.05, 1.35)     # 1 Tem 17:00 kesiti bandı
CLIP_1JUL_09 = (1.50, 2.00)     # 1 Tem 09:00 kesiti bandı (2.27 değil — kırpılmış)
NZR_ELIGIBLE = 0.30             # 1 Tem 09:00 artışının dağıtılacağı hücre eşiği


def dtme_dfs(d):
    dim = pd.Timestamp(d.year, d.month, 1) + pd.offsets.MonthEnd(0)
    return (dim.day - d.day), (d.day - 1)


def load_panel():
    """Sıfır-dolgulu panel: her (od, slot, tarih) hücresi; tarih ekseni tahmin haftasını da içerir."""
    df = pd.read_excel(DEMAND_XLSX)
    df["tarih"] = pd.to_datetime(df["tarih"])
    s = df["talep_tamamlanma_saati"].astype(str).str.strip()
    df["slot"] = np.where(s.str.match(r"^0?9"), "09:00", "17:00")

    hist_dates = sorted(df["tarih"].unique())
    all_dates = list(hist_dates) + list(FORECAST_DATES)
    od_pairs = sorted(set(zip(df["cikis"], df["varis"])))
    assert len(od_pairs) == 289, f"OD sayısı {len(od_pairs)} != 289"

    idx = pd.MultiIndex.from_product(
        [od_pairs, SLOTS, all_dates], names=["od", "slot", "tarih"])
    panel = pd.DataFrame(index=idx).reset_index()
    panel[["cikis", "varis"]] = pd.DataFrame(panel["od"].tolist(), index=panel.index)
    obs = df.set_index([df["cikis"] + "→" + df["varis"], "slot", "tarih"])["toplam_desi"]
    panel["key"] = panel["cikis"] + "→" + panel["varis"]
    panel["desi"] = obs.reindex(
        pd.MultiIndex.from_arrays([panel["key"], panel["slot"], panel["tarih"]])).values
    panel["is_future"] = panel["tarih"].isin(FORECAST_DATES)
    panel.loc[~panel["is_future"], "desi"] = panel.loc[~panel["is_future"], "desi"].fillna(0.0)
    return panel, hist_dates, all_dates, od_pairs


def ha_prev_sources(all_dates):
    """Her tarih için holiday-aware aynı-DOW geçmiş kaynak listesi (6 adet).
    Maske: tatiller, Kurban haftası, ay sonları (dtme=0), 28 Haziran, gelecek tarihler."""
    hist_last = pd.Timestamp("2026-06-28")
    masked = set(HOLIDAYS) | KURBAN_WEEK | {pd.Timestamp("2026-06-28")}
    for d in all_dates:
        if dtme_dfs(d)[0] == 0:
            masked.add(d)
    date_set = set(all_dates)
    src = {}
    for d in all_dates:
        lst = []
        w = 1
        while len(lst) < 6 and w <= 26:
            c = d - pd.Timedelta(weeks=w)
            if c in date_set and c not in masked and c <= hist_last:
                lst.append(c)
            w += 1
        src[d] = lst
    return src


def build_features(panel, all_dates, od_pairs):
    """§3.2 feature seti v3 (~40 feature). Panel üzerinde vektörize hesap."""
    print("  feature inşası...", flush=True)
    p = panel
    p["dow"] = p["tarih"].dt.dayofweek
    p["day_of_month"] = p["tarih"].dt.day
    p["week_of_year"] = p["tarih"].dt.isocalendar().week.astype(int)
    p["month"] = p["tarih"].dt.month
    p["is_weekend"] = (p["dow"] >= 5).astype(int)
    p["is_sunday"] = (p["dow"] == 6).astype(int)

    cal = pd.DataFrame({"tarih": pd.to_datetime(sorted(set(all_dates)))})
    dd = cal["tarih"].map(dtme_dfs)
    cal["dtme"] = [x[0] for x in dd]
    cal["dfs"] = [x[1] for x in dd]
    cal["dtme_c"] = cal["dtme"].clip(0, 5)
    cal["dfs_c"] = cal["dfs"].clip(0, 5)
    cal["is_month_end"] = (cal["dtme"] == 0).astype(int)
    cal["is_holiday"] = cal["tarih"].isin(HOLIDAYS | KURBAN_WEEK).astype(int)

    def d2h(d):
        vals = [abs((d - h).days) for h in HOLIDAYS | KURBAN_WEEK]
        return min(min(vals), 30)
    cal["days_to_holiday"] = cal["tarih"].map(d2h)

    def bridge(d):
        if d.dayofweek >= 5:
            return 0
        return int(any(d + pd.Timedelta(days=k) in HOLIDAYS for k in (-1, 1)))
    cal["is_bridge_day"] = cal["tarih"].map(bridge)
    cal["is_post_holiday_1_2"] = cal["tarih"].map(
        lambda d: int(any(0 < (d - e).days <= 2 for e in HOLIDAY_BLOCKS_END)))

    p = p.merge(cal, on="tarih", how="left")
    p["slot_17"] = (p["slot"] == "17:00").astype(int)
    p["slot_x_dow"] = p["slot_17"] * 7 + p["dow"]
    p["slot_x_dtme"] = p["slot_17"] * 6 + p["dtme_c"]
    p["slot_x_dfs"] = p["slot_17"] * 6 + p["dfs_c"]

    # --- rota istatistikleri: arındırılmış (saf) günlerden (§2.2 dip not)
    pure = p[(~p["is_future"]) & (p["is_holiday"] == 0) &
             (p["dtme"] >= 3) & (p["dfs"] >= 2) &
             (p["tarih"] != pd.Timestamp("2026-06-28"))]
    g = pure.groupby(["key", "slot"])["desi"]
    rs = g.agg(route_mean="mean", route_std="std", route_median="median").fillna(0)
    q = g.quantile([0.25, 0.75]).unstack()
    rs["route_iqr"] = (q[0.75] - q[0.25]).fillna(0)
    rs["route_cv"] = rs["route_std"] / (rs["route_mean"] + 1.0)
    p = p.merge(rs.reset_index(), on=["key", "slot"], how="left")

    ds = pure.groupby(["key", "slot", "dow"])["desi"].agg(
        dow_slot_route_mean="mean", dow_slot_route_median="median").reset_index()
    p = p.merge(ds, on=["key", "slot", "dow"], how="left")

    share = pure.groupby(["key", "slot"])["desi"].sum().unstack(fill_value=0)
    share_route = (share.get("17:00", 0) / (share.sum(axis=1) + 1e-9)).rename("slot_share_route")
    p = p.merge(share_route.reset_index(), on="key", how="left")

    # --- coğrafi
    net = pd.read_excel(NETWORK_XLSX)
    net["key"] = net["cikis"] + "→" + net["varis"]
    p = p.merge(net[["key", "mesafe_km", "hedef_teslim_gun"]]
                .rename(columns={"mesafe_km": "distance_km", "hedef_teslim_gun": "sla_gun"}),
                on="key", how="left")
    keys = sorted(p["key"].unique())
    p["route_idx"] = p["key"].map({k: i for i, k in enumerate(keys)})
    origins = sorted(p["cikis"].unique()); dests = sorted(p["varis"].unique())
    p["origin_idx"] = p["cikis"].map({o: i for i, o in enumerate(origins)})
    p["dest_idx"] = p["varis"].map({d: i for i, d in enumerate(dests)})

    # --- lag / rolling: (key,slot) x tarih matrisi üzerinde HA kaynak indeksleme
    piv = p[~p["is_future"]].pivot_table(index=["key", "slot"], columns="tarih",
                                         values="desi", aggfunc="first")
    piv = piv.reindex(columns=sorted(set(all_dates)))     # gelecek kolonlar NaN
    M = piv.values
    date_pos = {d: i for i, d in enumerate(piv.columns)}
    row_pos = {rk: i for i, rk in enumerate(piv.index)}
    src = ha_prev_sources(sorted(set(all_dates)))

    n_rows, n_dates = M.shape
    LAG = np.full((n_rows, n_dates, 6), np.nan)
    for j, d in enumerate(piv.columns):
        for k, sd in enumerate(src[d]):
            LAG[:, j, k] = M[:, date_pos[sd]]
    feats = {
        "lag_7": LAG[:, :, 0], "lag_14": LAG[:, :, 1],
        "lag_21": LAG[:, :, 2], "lag_28": LAG[:, :, 3],
        "rmean_4w": np.nanmean(LAG[:, :, :4], axis=2),
        "rmedian_6w": np.nanmedian(LAG, axis=2),
        "rstd_4w": np.nanstd(LAG[:, :, :4], axis=2),
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        feats["momentum_4w"] = (np.nanmean(LAG[:, :, :2], axis=2) /
                                (np.nanmean(LAG[:, :, 2:4], axis=2) + 1.0))

    # çapraz-slot lag7: aynı od, diğer slot
    other = np.empty(n_rows, dtype=int)
    for (k_, s_), i in row_pos.items():
        other[i] = row_pos[(k_, "17:00" if s_ == "09:00" else "09:00")]
    feats["same_day_other_slot_lag7"] = feats["lag_7"][other, :]

    # --- kesiklilik: gün-sırası dizisi üzerinden (28 Haz maskeli)
    hist_cols = [d for d in piv.columns if d <= pd.Timestamp("2026-06-27")]
    Mh = piv[hist_cols].values
    nz = (Mh > 0).astype(float)
    csum = np.cumsum(nz, axis=1)
    pad = np.concatenate([np.zeros((n_rows, 1)), csum], axis=1)
    win = 56
    nzr = np.full_like(Mh, np.nan)
    for j in range(Mh.shape[1]):
        # sıfır/pozitif durumu feature'a sızmasın. Önceden pad[:,j+1] (j dahil) idi.
        lo = max(0, j - win)
        if j - lo > 0:
            nzr[:, j] = (pad[:, j] - pad[:, lo]) / (j - lo)
    dsl = np.zeros_like(Mh); zst = np.zeros_like(Mh)
    run_d = np.zeros(n_rows); run_z = np.zeros(n_rows)
    for j in range(Mh.shape[1]):
        dsl[:, j] = run_d; zst[:, j] = run_z
        pos = Mh[:, j] > 0
        run_d = np.where(pos, 0, run_d + 1)
        run_z = np.where(pos, 0, run_z + 1)
    # gelecek/28-Haz projeksiyonu: son gözlenen güne (27 Haz) KADAR dahil pencere
    # (gelecek için "mevcut gün" yok; son gözlemi kullanmak causal-tutarlı ve doğru)
    nh = Mh.shape[1]
    lo_f = max(0, nh - win)
    state_dsl = run_d; state_zst = run_z
    state_nzr = (pad[:, nh] - pad[:, lo_f]) / max(nh - lo_f, 1)

    disc = {"nonzero_rate_8w": np.full((n_rows, n_dates), np.nan),
            "days_since_last_demand": np.full((n_rows, n_dates), np.nan),
            "zero_streak": np.full((n_rows, n_dates), np.nan)}
    for j, d in enumerate(piv.columns):
        if d in hist_cols:
            jj = hist_cols.index(d)
            disc["nonzero_rate_8w"][:, j] = nzr[:, jj]
            disc["days_since_last_demand"][:, j] = dsl[:, jj]
            disc["zero_streak"][:, j] = zst[:, jj]
        else:  # 28 Haz + tahmin günleri: 27 Haz durumundan ileri say
            ahead = (d - pd.Timestamp("2026-06-27")).days
            disc["nonzero_rate_8w"][:, j] = state_nzr
            disc["days_since_last_demand"][:, j] = state_dsl + ahead
            disc["zero_streak"][:, j] = state_zst + ahead
    feats.update(disc)

    # matris feature'larını panele geri yaz
    ri = p.set_index(["key", "slot"]).index.map(row_pos).values
    ci = p["tarih"].map(date_pos).values
    for name, mat in feats.items():
        p[name] = mat[ri, ci]

    # --- V4-1: origin/dest agregat lag'leri — ablasyonla ELENDİ (bkz. docstring);
    #     yeniden denemek için AGG_FEATURES=True + FEATURES listesine 4 kolon ekle
    if not AGG_FEATURES:
        return p
    hist_p = p[~p["is_future"]]
    for side, col in [("origin", "cikis"), ("dest", "varis")]:
        tot = (hist_p.groupby([col, "tarih"])["desi"].sum()
               .unstack("tarih").reindex(columns=piv.columns))     # gelecek = NaN
        T = tot.values
        tpos = {tm: i for i, tm in enumerate(tot.index)}
        L7 = np.full((T.shape[0], n_dates), np.nan)
        R4 = np.full((T.shape[0], n_dates), np.nan)
        for j, d in enumerate(piv.columns):
            ss = src[d]
            if ss:
                L7[:, j] = T[:, date_pos[ss[0]]]
                cols4 = [date_pos[x] for x in ss[:4]]
                R4[:, j] = np.nanmean(T[:, cols4], axis=1)
        tri = p[col].map(tpos).values
        p[f"{side}_tot_lag7"] = L7[tri, ci]
        p[f"{side}_tot_rmean4w"] = R4[tri, ci]
    return p


FEATURES = [
    "dow", "day_of_month", "week_of_year", "month", "is_weekend", "is_sunday",
    "slot_17", "slot_x_dow", "slot_share_route",
    "dtme_c", "dfs_c", "is_month_end", "slot_x_dtme", "slot_x_dfs",
    "is_holiday", "days_to_holiday", "is_bridge_day", "is_post_holiday_1_2",
    "lag_7", "lag_14", "lag_21", "lag_28", "same_day_other_slot_lag7",
    "rmean_4w", "rmedian_6w", "rstd_4w", "momentum_4w",
    "route_mean", "route_std", "route_median", "route_cv", "route_iqr",
    "dow_slot_route_mean", "dow_slot_route_median",
    "distance_km", "sla_gun", "route_idx", "origin_idx", "dest_idx",
    "nonzero_rate_8w", "days_since_last_demand", "zero_streak",
]
AGG_FEATURES = False   # V4-1 negatif sonuç — ablasyon kanıtıyla kapalı


def lgbm(objective, seed=42):
    import lightgbm as lgb
    params = dict(n_estimators=600, learning_rate=0.05, num_leaves=127,
                  min_child_samples=20, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                  verbose=-1, n_jobs=-1)
    if objective == "l1":
        params.update(objective="quantile", alpha=0.5)
    elif objective == "tweedie":
        params.update(objective="tweedie", tweedie_variance_power=1.3)
    elif isinstance(objective, float):
        params.update(objective="quantile", alpha=objective)
    return lgb.LGBMRegressor(**params)


def wmape(y, yhat):
    return float(np.sum(np.abs(np.asarray(yhat) - np.asarray(y))) /
                 max(np.sum(np.asarray(y)), 1e-9) * 100)


def tsb_forecast(panel, all_dates):
    """TSB (Teunter–Syntetos–Babai) — yalnız 09:00 kesiti için ensemble üyesi (§3.5).
    Her (od,09:00) serisinde gün-sırası dizisi üzerinden olasılık x büyüklük."""
    p9 = panel[panel["slot"] == "09:00"]
    piv = p9[~p9["is_future"]].pivot_table(index="key", columns="tarih",
                                           values="desi", aggfunc="first")
    hist_cols = [d for d in piv.columns if d <= pd.Timestamp("2026-06-27")]
    A, B = 0.10, 0.10
    Y = piv[hist_cols].values
    prob = np.full(Y.shape[0], 0.5); size = np.nanmean(np.where(Y > 0, Y, np.nan), axis=1)
    size = np.nan_to_num(size, nan=0.0)
    for j in range(Y.shape[1]):
        y = Y[:, j]; pos = y > 0
        prob = prob + A * (pos.astype(float) - prob)
        size = np.where(pos, size + B * (y - size), size)
    est = prob * size
    return dict(zip(piv.index, est))


def rule_kind(d):
    dtme, dfs = dtme_dfs(d)
    if dtme == 0:
        return "dtme0"
    if dtme == 1:
        return "dtme1"
    if dfs == 0:
        return "dfs0"
    return None


def build_anchors(panel, cutoff):
    """
    dow_total: saf günlerin son ≤8 gözlem medyanı; dowslot: slot kesitli;
    me_shape: geçmiş ay-sonu günlerinin (key,slot) şekli."""
    hist = panel[(~panel["is_future"]) & (panel["tarih"] < cutoff) &
                 (panel["tarih"] <= pd.Timestamp("2026-06-27"))]
    day_tot = hist.groupby("tarih")["desi"].sum()
    info = pd.DataFrame({"total": day_tot})
    info["dow"] = info.index.dayofweek
    dd = [dtme_dfs(d) for d in info.index]
    info["dtme"] = [x[0] for x in dd]
    info["dfs"] = [x[1] for x in dd]
    info["hol"] = [d in (HOLIDAYS | KURBAN_WEEK) for d in info.index]
    pure = info[(~info["hol"]) & (info["dtme"] >= 3) & (info["dfs"] >= 2)]
    dow_total = {k: float(v["total"].tail(8).median()) for k, v in pure.groupby("dow")}
    slot_tot = hist.groupby(["tarih", "slot"])["desi"].sum().unstack(fill_value=0)
    dowslot = {}
    for dow in range(7):
        days = [d for d in pure.index if d.dayofweek == dow and d in slot_tot.index]
        for s in SLOTS:
            vals = slot_tot.loc[days, s] if days else pd.Series(dtype=float)
            dowslot[(dow, s)] = float(vals.tail(8).median()) if len(vals) else 0.0
    me_days = [d for d in info.index if info.loc[d, "dtme"] == 0]
    me_shape = {}
    if me_days:
        me_shape = (hist[hist["tarih"].isin(me_days)]
                    .groupby(["key", "slot"])["desi"].sum() / len(me_days)).to_dict()
    return {"dow_total": dow_total, "dowslot": dowslot, "me_shape": me_shape}


def apply_month_rules(fc, anchors, k30, meta=None):
    """Ay-sınırı kuralları (§3.3, §10.1–10.3) — hedef haftaya VE analog fold'lara
    uygulanan TEK makine. fc: tarih, slot, key, tahmin, nonzero_rate_8w kolonları."""
    fc = fc.copy()
    for d in sorted(pd.to_datetime(fc["tarih"].unique())):
        kind = rule_kind(d)
        if kind is None:
            continue
        dow = d.dayofweek
        m = fc["tarih"] == d
        base = anchors["dow_total"].get(dow, 0.0)
        if kind == "dtme0" and base > 0:
            target = k30 * base
            shape = fc.loc[m].apply(
                lambda r: anchors["me_shape"].get((r["key"], r["slot"]), 0.0), axis=1).values
            ssum = shape.sum()
            fc.loc[m, "tahmin"] = (shape * target / ssum) if ssum > 0 \
                else target / max(int(m.sum()), 1)
            if meta is not None:
                meta.setdefault("rules", []).append(
                    {"gun": str(d.date()), "kural": f"dtme0 ×{k30}", "hedef": float(target)})
        elif kind == "dtme1" and base > 0:
            tot = fc.loc[m, "tahmin"].sum()
            sc = min(max(tot, CLIP_29JUN[0] * base), CLIP_29JUN[1] * base)
            if tot > 0 and abs(sc - tot) > 1e-6:
                fc.loc[m, "tahmin"] *= sc / tot
            if meta is not None:
                meta.setdefault("rules", []).append(
                    {"gun": str(d.date()), "kural": "dtme1 bant", "once": float(tot),
                     "sonra": float(sc)})
        elif kind == "dfs0":
            for s, band in [("17:00", CLIP_1JUL_17), ("09:00", CLIP_1JUL_09)]:
                anc = anchors["dowslot"].get((dow, s), 0.0)
                if anc <= 0:
                    continue
                ms = m & (fc["slot"] == s)
                tot = fc.loc[ms, "tahmin"].sum()
                sc = min(max(tot, band[0] * anc), band[1] * anc)
                if tot <= 0 or abs(sc - tot) < 1e-6:
                    continue
                if sc > tot and s == "09:00":
                    elig = ms & (fc["nonzero_rate_8w"] > NZR_ELIGIBLE)
                    b = fc.loc[elig, "tahmin"].sum()
                    if b > 0:
                        fc.loc[elig, "tahmin"] *= 1 + (sc - tot) / b
                    else:
                        fc.loc[ms, "tahmin"] *= sc / tot
                else:
                    fc.loc[ms, "tahmin"] *= sc / tot
                if meta is not None:
                    meta.setdefault("rules", []).append(
                        {"gun": str(d.date()), "kural": f"dfs0 {s}", "once": float(tot),
                         "sonra": float(sc)})
    return fc


def main():
    print("=== forecast_v4 — panel inşası ===", flush=True)
    panel, hist_dates, all_dates, od_pairs = load_panel()
    panel = build_features(panel, all_dates, od_pairs)

    train_all = panel[(~panel["is_future"]) &
                      (~panel["tarih"].isin(ANOMALOUS_TRAIN_EXCLUDE))].copy()
    # §10.10: ay-sınırı örnek ağırlığı ×2
    train_all["w"] = np.where((train_all["dtme"] <= 2) | (train_all["dfs"] <= 1), 2.0, 1.0)
    Xcols = FEATURES
    med = train_all[Xcols].median()

    # ------------- V4-2: 6 fold (4 analog + 2 ay-ortası), ağırlıklı seçim + kararlılık
    folds = [("2026-01-26", "2026-02-01", "analog"),   # soğuk başlangıç
             ("2026-02-23", "2026-03-01", "analog"),
             ("2026-03-30", "2026-04-05", "analog"),
             ("2026-04-27", "2026-05-03", "analog"),   # 1 May skor dışı (maske)
             ("2026-06-08", "2026-06-14", "mid"),
             ("2026-06-15", "2026-06-21", "mid")]
    W_ANALOG, W_MID = 0.7, 0.3
    tsb_map = tsb_forecast(panel, all_dates)
    member_names = ["l1", "tw", "ha_med", "ha_lag7", "tsb"]
    cv_rows = {s: {"y": [], "layer": [], "cols": {k: [] for k in member_names}}
               for s in SLOTS}
    fold_scores, guard_rows = [], []
    K30_GRID = (0.02, 0.05, 0.08)
    for fs, fe, layer in folds:
        fs, fe = pd.Timestamp(fs), pd.Timestamp(fe)
        tr = train_all[train_all["tarih"] < fs]
        te = panel[(panel["tarih"] >= fs) & (panel["tarih"] <= fe) &
                   (~panel["tarih"].isin(ANOMALOUS_TRAIN_EXCLUDE)) & (~panel["is_future"])]
        if len(te) == 0 or len(tr) < 5000:
            continue
        Xtr = tr[Xcols].fillna(med); ytr = tr["desi"].values; wtr = tr["w"].values
        Xte = te[Xcols].fillna(med); yte = te["desi"].values
        m1 = lgbm("l1"); m1.fit(Xtr, ytr, sample_weight=wtr)
        m2 = lgbm("tweedie"); m2.fit(Xtr, ytr, sample_weight=wtr)
        p1 = np.maximum(m1.predict(Xte), 0); p2 = np.maximum(m2.predict(Xte), 0)
        ha_med = te["rmedian_6w"].fillna(0).values
        ha_l7 = te["lag_7"].fillna(0).values
        tsb = np.where(te["slot"] == "09:00",
                       te["key"].map(tsb_map).fillna(0), ha_med)
        for s in SLOTS:
            ms = (te["slot"] == s).values
            cv_rows[s]["y"].append(yte[ms])
            cv_rows[s]["layer"].append(layer)
            for k, v in zip(member_names, [p1, p2, ha_med, ha_l7, tsb]):
                cv_rows[s]["cols"][k].append(np.asarray(v)[ms])
        fold_scores.append({"fold": f"{fs.date()}–{fe.date()}", "layer": layer,
                            "L1": wmape(yte, p1), "Tweedie": wmape(yte, p2),
                            "HA_med": wmape(yte, ha_med), "HA_lag7": wmape(yte, ha_l7)})
        # --- V4-3: korkuluk makinesi backtest'i (fold'un kendi ay-sınırı günleri)
        anc_f = build_anchors(panel, fs)
        df_f = te[["tarih", "slot", "key", "nonzero_rate_8w"]].copy()
        df_f["tahmin"] = p1
        gr = {"fold": f"{fs.date()}–{fe.date()}", "layer": layer,
              "kural_yok": wmape(yte, p1)}
        for k30 in K30_GRID:
            adj = apply_month_rules(df_f, anc_f, k30)
            gr[f"k={k30}"] = wmape(yte, adj["tahmin"].values)
        guard_rows.append(gr)

    print("\nCV fold WMAPE'leri (padded panel):")
    for r in fold_scores:
        print("  [{layer:6s}] {fold}: L1 {L1:.1f} | TW {Tweedie:.1f} | "
              "HAmed {HA_med:.1f} | HAlag7 {HA_lag7:.1f}".format(**r))

    # kararlılık raporu: üye bazında mean ± std + ağırlıklı skor
    def layer_score(vals, layers):
        a = [v for v, l_ in zip(vals, layers) if l_ == "analog"]
        m_ = [v for v, l_ in zip(vals, layers) if l_ == "mid"]
        return (W_ANALOG * np.mean(a) if a else 0) + (W_MID * np.mean(m_) if m_ else 0)

    layers_all = [r["layer"] for r in fold_scores]
    print("\nKararlılık raporu (mean ± std | ağırlıklı skor 0.7×analog + 0.3×orta):")
    for mname, col in [("L1", "L1"), ("Tweedie", "Tweedie"),
                       ("HA_med", "HA_med"), ("HA_lag7", "HA_lag7")]:
        vals = [r[col] for r in fold_scores]
        print(f"  {mname:8s}: {np.mean(vals):5.1f} ± {np.std(vals):4.1f} | "
              f"skor {layer_score(vals, layers_all):5.1f}")

    print("\nKorkuluk backtest'i (fold WMAPE — kural yok vs ×k override):")
    for gr in guard_rows:
        print("  [{layer:6s}] {fold}: yok {kural_yok:.1f} | ".format(**gr) +
              " | ".join(f"k={k}: {gr[f'k={k}']:.1f}" for k in K30_GRID))
    analog_g = [gr for gr in guard_rows if gr["layer"] == "analog"]
    k30_star = min(K30_GRID,
                   key=lambda k: np.mean([gr[f"k={k}"] for gr in analog_g])) \
        if analog_g else 0.05
    print(f"  → seçilen k30* = {k30_star} (analog fold ortalamasına göre; "
          f"kural yok ort {np.mean([gr['kural_yok'] for gr in analog_g]):.1f} → "
          f"{np.mean([gr[f'k={k30_star}'] for gr in analog_g]):.1f})")

    # slot bazında çekirdek seçimi (ağırlıklı skor) + ağırlıklı NNLS (§3.5, §3.8)
    from scipy.optimize import nnls
    slot_core, slot_weights = {}, {}
    for s in SLOTS:
        layers_s = cv_rows[s]["layer"]
        n_a = max(sum(1 for l_ in layers_s if l_ == "analog"), 1)
        n_m = max(sum(1 for l_ in layers_s if l_ == "mid"), 1)
        # fold bazlı üye WMAPE'leri → ağırlıklı skorla çekirdek seçimi
        def member_score(k):
            vals = [wmape(y_, c_) for y_, c_ in zip(cv_rows[s]["y"], cv_rows[s]["cols"][k])]
            return layer_score(vals, layers_s)
        core = "l1" if member_score("l1") <= member_score("tw") else "tw"
        slot_core[s] = core
        # satır ağırlıkları: katman bütçesi fold sayısına ve satırına bölünür
        rw = np.concatenate([
            np.full(len(y_), (W_ANALOG / n_a if l_ == "analog" else W_MID / n_m) / len(y_))
            for y_, l_ in zip(cv_rows[s]["y"], layers_s)])
        y = np.concatenate(cv_rows[s]["y"])
        C = {k: np.concatenate(v) for k, v in cv_rows[s]["cols"].items()}
        members = [core, "ha_med", "ha_lag7"] + (["tsb"] if s == "09:00" else [])
        A = np.column_stack([C[k] for k in members])
        sq = np.sqrt(rw)
        w, _ = nnls(A * sq[:, None], y * sq)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(members)) / len(members)

        def wmape_w(yh):
            return float(np.sum(rw * np.abs(yh - y)) / max(np.sum(rw * y), 1e-9) * 100)
        if wmape_w(A @ w) > wmape_w(C["ha_med"]):   # güvenlik eşiği (§3.5)
            w = np.array([0.0] * len(members)); w[members.index("ha_med")] = 1.0
        slot_weights[s] = dict(zip(members, map(float, w)))
        print(f"  slot {s}: çekirdek={core} ağırlıklar=" +
              ", ".join(f"{k}:{v:.2f}" for k, v in slot_weights[s].items()) +
              f" | ağırlıklı CV WMAPE ens {wmape_w(A @ w):.1f}%")

    # ---------------- Final modeller (tüm tarih, anomaliler hariç)
    print("\nFinal eğitim...", flush=True)
    Xtr = train_all[Xcols].fillna(med); ytr = train_all["desi"].values
    wtr = train_all["w"].values
    models = {}
    # V5-B: çekirdek(ler) + Tweedie harmanı için L1 ve Tweedie'yi HER ZAMAN eğit
    for k in ({slot_core[s] for s in SLOTS} | {"l1", "tweedie"}):
        mdl = lgbm(k); mdl.fit(Xtr, ytr, sample_weight=wtr); models[k] = mdl
    q_models = {}
    for q in (0.10, 0.90):   # §3.7 P10/P90 — stres testi/rapor için
        mdl = lgbm(q); mdl.fit(Xtr, ytr, sample_weight=wtr); q_models[q] = mdl

    fut = panel[panel["is_future"]].copy()
    Xf = fut[Xcols].fillna(med)
    pred_core = {k: np.maximum(m.predict(Xf), 0) for k, m in models.items()}
    # WMAPE 31.70→31.49; ensemble bunu L1-çekirdek üyesi olarak kullanır)
    TW_BLEND = 0.25  # ayrıştırma: blend hem WMAPE'yi (31.70→31.49) hem plan
                     # maliyetini (11.374M→11.045M) düşürür → net kazanç, açık
    if "l1" in pred_core and "tweedie" in pred_core:
        pred_core["l1"] = ((1 - TW_BLEND) * pred_core["l1"]
                           + TW_BLEND * pred_core["tweedie"])
    fut["ha_med"] = fut["rmedian_6w"].fillna(0)
    fut["ha_lag7"] = fut["lag_7"].fillna(0)
    tsb_col = fut["key"].map(tsb_forecast(panel, all_dates)).fillna(0)

    tahmin = np.zeros(len(fut))
    for s in SLOTS:
        ms = (fut["slot"] == s).values
        w = slot_weights[s]
        est = np.zeros(ms.sum())
        for k, wk in w.items():
            if wk == 0:
                continue
            col = (pred_core[k][ms] if k in pred_core else
                   fut.loc[ms, "ha_med"].values if k == "ha_med" else
                   fut.loc[ms, "ha_lag7"].values if k == "ha_lag7" else
                   tsb_col[ms].values)
            est = est + wk * np.asarray(col, dtype=float)
        tahmin[ms] = est
    fut["tahmin"] = np.maximum(tahmin, 0)
    fut["p10"] = np.maximum(q_models[0.10].predict(Xf), 0)
    fut["p90"] = np.maximum(q_models[0.90].predict(Xf), 0)

    # ---------------- Çapalar + korkuluklar (V4-3: backtest'le kilitlenen k30*)
    anchors = build_anchors(panel, pd.Timestamp("2026-06-29"))
    meta = {"version": "v5", "slot_core": slot_core, "slot_weights": slot_weights,
            "cv_folds": fold_scores, "guardrail_backtest": guard_rows,
            "k30_star": float(k30_star)}
    fut = apply_month_rules(fut, anchors, k30_star, meta)

    # ---------------- Çıktı — şablon formatı (TALEP TAHMİNİ.xlsx ile birebir)
    fut = fut.sort_values(["tarih", "slot", "cikis", "varis"]).reset_index(drop=True)
    # S-C (OpAI S5): ~0 tahminler de satır olarak sunulur — plan tarafı için 0.01 desi taban
    fut["tahmin"] = fut["tahmin"].clip(lower=0.01)
    fut["Talep ID"] = [f"D{i + 1:05d}" for i in range(len(fut))]
    out = pd.DataFrame({
        "Talep ID": fut["Talep ID"],
        "Tarih": fut["tarih"].dt.strftime("%d.%m.%Y"),
        "Talep Tamamlama Saati": fut["slot"].map({"09:00": "09:00:00", "17:00": "17:00:00"}),
        "Çıkış Transfer Merkezi": fut["cikis"],
        "Varış Transfer Merkezi": fut["varis"],
        "Tahmin Edilen Desi": fut["tahmin"].round(2),
    })
    assert len(out) == 4046, f"satır sayısı {len(out)} != 4046"
    out.to_excel(OUT_XLSX, index=False, engine="openpyxl")

    # optimizer/stres testi için P10/P90 yan dosyası
    fut[["Talep ID", "tarih", "slot", "cikis", "varis", "tahmin", "p10", "p90"]] \
        .assign(tarih=lambda d: d["tarih"].dt.strftime("%Y-%m-%d")) \
        .to_csv(os.path.join(BASE, "forecast_quantiles.csv"), index=False, encoding="utf-8-sig")

    daily = fut.groupby(fut["tarih"].dt.date)["tahmin"].sum()
    dowN = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
    print("\nTahmin haftası gün toplamları (beklenen şekil §2.11):")
    for d, t in daily.items():
        print(f"  {d} ({dowN[pd.Timestamp(d).dayofweek]}): {t / 1e6:.2f}M desi")
    print(f"  HAFTA TOPLAMI: {daily.sum() / 1e6:.2f}M desi ")
    meta["daily_totals"] = {str(k): float(v) for k, v in daily.items()}
    meta["week_total"] = float(daily.sum())
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nYazıldı: {OUT_XLSX} ({len(out)} satır) | süre {time.time() - T0:.0f} sn")


if __name__ == "__main__":
    main()
