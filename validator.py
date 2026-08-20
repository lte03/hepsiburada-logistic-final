# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 — Gelişmiş Çözüm Aşaması — Bağımsız Kural Denetleyicisi (§7)
============================================================================
Talep-tahmini.xlsx + Tasima-plani.xlsx dosyalarını optimizer'dan bağımsız olarak
yeniden okuyup tüm hard kısıtları ve format kurallarını denetler:

  F1. Format: sütunlar şablonla birebir; ID formatları (D00001 / D00001-1 / V0001);
      saat SS:DD; tarih GG.AA.YYYY
  F2. Tahmin: 4.046 satır — 289 OD × 7 gün × 2 slot, eksiksiz ve tekil
  F3. ID eşleşmesi: plan ↔ tahmin birebir; bölünmüş desi toplamı = tahmin desisi
  F4. Kiralık filo: 12 hat × 7 gün zorunlu çıkış; rota sapması/uğrama/dönüş yok;
      çıkış o gün içinde; araç türü doğru
  F5. Araç kapasitesi: bacak yükü ≤ tür kapasitesi
  F6. Elleçleme kotası (hard): TM-gün desi ≤ kota — gece yarısı oransal bölme
      aritmetiğiyle (S-C: AUGM/ByteSis 3K/7K) yeniden hesaplanır
  F7. Tır kotası (hard): TM-gün tekil 'Tır' araç sayısı ≤ v2 kapasitesi
  F8. SLA cezası: talep tamamlanma → nihai varış elleçleme bitişi; üst saate
      yuvarlama; rapor edilen cezayla karşılaştırma
  F9. Maliyet: kullanım süresi (elleçleme+bekleme+seyir) × saatlik kira + km ×
      km-maliyeti; rapor edilen toplamla karşılaştırma
  F10. Hazır-olma: hiçbir talep, tamamlanma saatinden önce elleçlenip yüklenemez

Çalıştırma:  python validator.py
"""
import math
import os
import re
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
EPOCH = pd.Timestamp("2026-06-29")
SLA_TL = 0.4
H_MIN = 0.01
TOL_DESI = 0.51          # yuvarlama toleransı (desi)
TOL_TL = 2.0             # satır başına maliyet toleransı (TL)

FORECAST_COLS = ["Talep ID", "Tarih", "Talep Tamamlama Saati", "Çıkış Transfer Merkezi",
                 "Varış Transfer Merkezi", "Tahmin Edilen Desi"]
PLAN_COLS = ["Araç ID", "Araç Tipi", "Araç türü", "Çıkış Transfer Merkezi",
             "Varış Transfer Merkezi", "Çıkış Tarihi", "Çıkış Saati", "Varış Tarihi",
             "Varış Saati", "Talep ID", "Taşınan Desi", "Yolculuk süresi",
             "Varış elleçleme süresi", "Çıkış Elleçleme süresi", "SLA cezası",
             "Toplam maliyet"]

issues = []


def flag(code, msg):
    issues.append((code, msg))


def to_min(date_str, time_str):
    ts = pd.to_datetime(f"{date_str} {time_str}", format="%d.%m.%Y %H:%M")
    return (ts - EPOCH).total_seconds() / 60.0


def split_by_days(start, dur, desi):
    """Gece yarısı oransal bölme: [(gün, desi payı)] (S-C 3K/7K aritmetiği)."""
    if desi <= 0 or dur <= 0:
        return [(int(start // 1440), 0.0)]
    out, t, end = [], start, start + dur
    while t < end - 1e-9:
        d = int(t // 1440)
        seg = min(end, (d + 1) * 1440.0) - t
        out.append((d, desi * seg / dur))
        t = min(end, (d + 1) * 1440.0)
    return out


def main():
    print("=== validator — bağımsız denetim ===")
    fc = pd.read_excel(os.path.join(BASE, "Talep-tahmini.xlsx"))
    plan = pd.read_excel(os.path.join(BASE, "Tasima-plani.xlsx"))

    # ---------------- F1: format
    if list(fc.columns) != FORECAST_COLS:
        flag("F1", f"Tahmin sütunları şablondan farklı: {list(fc.columns)}")
    if list(plan.columns) != PLAN_COLS:
        flag("F1", f"Plan sütunları şablondan farklı: {list(plan.columns)}")
    for c, pat in [("Talep ID", r"^D\d{5}$")]:
        bad = fc[~fc[c].astype(str).str.match(pat)]
        if len(bad):
            flag("F1", f"Tahmin {c} format hatası: {bad[c].iloc[0]} …({len(bad)})")
    bad = plan[(plan["Talep ID"] != "-") &
               (~plan["Talep ID"].astype(str).str.match(r"^D\d{5}(-\d+)*$"))]
    if len(bad):
        flag("F1", f"Plan Talep ID format hatası: {bad['Talep ID'].iloc[0]} …({len(bad)})")
    bad = plan[~plan["Araç ID"].astype(str).str.match(r"^V\d{4,}$")]
    if len(bad):
        flag("F1", f"Araç ID format hatası: {bad['Araç ID'].iloc[0]}")
    for c in ["Çıkış Saati", "Varış Saati"]:
        bad = plan[~plan[c].astype(str).str.match(r"^\d{2}:\d{2}$")]
        if len(bad):
            flag("F1", f"{c} SS:DD değil: {bad[c].iloc[0]}")
    for c in ["Çıkış Tarihi", "Varış Tarihi"]:
        bad = plan[~plan[c].astype(str).str.match(r"^\d{2}\.\d{2}\.\d{4}$")]
        if len(bad):
            flag("F1", f"{c} GG.AA.YYYY değil: {bad[c].iloc[0]}")

    # ---------------- F2: tahmin kapsamı
    if len(fc) != 4046:
        flag("F2", f"Tahmin satır sayısı {len(fc)} != 4046")
    if fc["Talep ID"].nunique() != len(fc):
        flag("F2", "Talep ID tekrarı var")
    hist = pd.read_excel(os.path.join(DATA, "teknofest26_gelismis.xlsx"))
    od_hist = set(zip(hist["cikis"], hist["varis"]))
    od_fc = set(zip(fc["Çıkış Transfer Merkezi"], fc["Varış Transfer Merkezi"]))
    if od_fc - od_hist:
        flag("F2", f"Geçmişte görülmeyen TM çifti için tahmin: {sorted(od_fc - od_hist)[:3]}")
    if len(od_fc) != 289:
        flag("F2", f"OD sayısı {len(od_fc)} != 289")
    combo = fc.groupby(["Tarih", "Talep Tamamlama Saati"]).size()
    if not (combo == 289).all():
        flag("F2", "Her (gün, slot) kesitinde 289 satır yok")
    if (fc["Tahmin Edilen Desi"] < 0).any():
        flag("F2", "Negatif tahmin var")

    # ---------------- referans veriler
    net = pd.read_excel(os.path.join(DATA, "sehirler_arasi_lojistik.xlsx"))
    dist = {(r["cikis"], r["varis"]): float(r["mesafe_km"]) for _, r in net.iterrows()}
    sla = {(r["cikis"], r["varis"]): int(r["hedef_teslim_gun"]) for _, r in net.iterrows()}
    tcolmap = {"Tır": "Tir_Suresi_Saat", "Kamyon": "Kamyon_Suresi_Saat",
               "Hafif Kamyon": "Hafif_Kamyon_Suresi_Saat", "Kamyonet": "Kamyonet_Suresi_Saat"}
    tt = {(r["cikis"], r["varis"]): {t: float(r[c]) * 60 for t, c in tcolmap.items()}
          for _, r in net.iterrows()}
    veh = pd.read_excel(os.path.join(DATA, "Araç_Kapasite_Maliyet_Saat.xlsx"))
    cap = dict(zip(veh["Araç Adı"], veh["Kapasite (desi)"]))
    spot_hr = dict(zip(veh["Araç Adı"], veh["Spot Araç Saatlik Kira (TL)"]))
    spot_km = dict(zip(veh["Araç Adı"], veh["Spot Kilometre Başına Maliyet (TL)"]))
    rent_hr = dict(zip(veh["Araç Adı"], veh["Kiralık Araç Saatlik Kira (TL)"]))
    rent_km = dict(zip(veh["Araç Adı"], veh["Kiralık Araç Kilometre Başına Maliyet (TL)"]))
    ell = pd.read_excel(os.path.join(DATA, "Ellecleme-kapasite.xlsx"))
    hcap = dict(zip(ell["transfer_merkezi"], ell["ellecleme_kapasite"]))
    tir = pd.read_excel(os.path.join(DATA, "tir_kapasiteleri v2.xlsx"))
    tircap = dict(zip(tir["transfer_merkezi"], tir["tir_kapasitesi"]))
    rentals = pd.read_excel(os.path.join(DATA, "Kiralık_Araclar.xlsx"))

    fc["ready_min"] = fc.apply(
        lambda r: (pd.to_datetime(r["Tarih"], format="%d.%m.%Y") - EPOCH).days * 1440
        + (540 if str(r["Talep Tamamlama Saati"]).startswith("09") else 1020), axis=1)
    dem = fc.set_index("Talep ID")

    # ---------------- F3: ID eşleşmesi + desi bütünlüğü
    p = plan[plan["Talep ID"] != "-"].copy()
    p["base_id"] = p["Talep ID"].astype(str).str.split("-").str[0]
    unknown = set(p["base_id"]) - set(fc["Talep ID"])
    if unknown:
        flag("F3", f"Planda tahminde olmayan Talep ID: {sorted(unknown)[:5]}")
    not_served = set(fc["Talep ID"]) - set(p["base_id"])
    if not_served:
        flag("F3", f"Taşınmayan talep: {len(not_served)} adet, örn {sorted(not_served)[:5]}")
    # bölünmüş parçaların desi toplamı — her parça nihai varışa 1 kez ulaşmalı
    final_rows = p[p.apply(lambda r: dem.loc[r["base_id"], "Varış Transfer Merkezi"]
                           == r["Varış Transfer Merkezi"], axis=1)]
    got = final_rows.groupby("base_id")["Taşınan Desi"].sum()
    for tid, want in dem["Tahmin Edilen Desi"].items():
        have = got.get(tid, 0.0)
        if abs(have - want) > TOL_DESI:
            flag("F3", f"{tid}: taşınan {have:.2f} != tahmin {want:.2f}")

    # ---------------- gezi (trip) rekonstrüksiyonu
    plan["dep_min"] = [to_min(a, b) for a, b in zip(plan["Çıkış Tarihi"], plan["Çıkış Saati"])]
    plan["arr_min"] = [to_min(a, b) for a, b in zip(plan["Varış Tarihi"], plan["Varış Saati"])]
    legs = plan.groupby(["Araç ID", "Araç Tipi", "Araç türü", "Çıkış Transfer Merkezi",
                         "Varış Transfer Merkezi", "dep_min", "arr_min",
                         "Yolculuk süresi", "Varış elleçleme süresi",
                         "Çıkış Elleçleme süresi"], as_index=False).agg(
        desi=("Taşınan Desi", "sum"), maliyet=("Toplam maliyet", "sum"),
        ceza=("SLA cezası", "sum"))

    # F5 + seyir süresi tutarlılığı
    for _, L in legs.iterrows():
        od = (L["Çıkış Transfer Merkezi"], L["Varış Transfer Merkezi"])
        typ = L["Araç türü"]
        if od not in dist:
            flag("F5", f"{L['Araç ID']}: matris dışı bacak {od}")
            continue
        if L["desi"] > cap[typ] + TOL_DESI:
            flag("F5", f"{L['Araç ID']} {od}: yük {L['desi']:.0f} > {typ} kapasitesi")
        exp_tt = tt[od][typ]
        if abs((L["arr_min"] - L["dep_min"]) - exp_tt) > 1.5:   # SS:DD yuvarlama payı
            flag("F5", f"{L['Araç ID']} {od}: varış-çıkış {L['arr_min']-L['dep_min']:.0f} dk"
                       f" != seyir {exp_tt:.0f} dk")

    # ---------------- F4: kiralık filo
    rq = {}
    for _, r in rentals.iterrows():
        rq[(r["Çıkış Transfer Merkezi"], r["Varış Transfer Merkezi"],
            r["Araç Türü"])] = int(r["Araç sayısı"])
    kl = legs[legs["Araç Tipi"] == "Kiralık"].copy()
    kl["day"] = (kl["dep_min"] // 1440).astype(int)
    for day in range(7):
        sub = kl[kl["day"] == day]
        got_ln = sub.groupby(["Çıkış Transfer Merkezi", "Varış Transfer Merkezi",
                              "Araç türü"])["Araç ID"].nunique().to_dict()
        for key, n in rq.items():
            if got_ln.get(key, 0) != n:
                flag("F4", f"Gün {day} ({(EPOCH + pd.Timedelta(days=day)).date()}) "
                           f"kiralık hat {key}: {got_ln.get(key, 0)}/{n} çıkış")
    extra = set(map(tuple, kl[["Çıkış Transfer Merkezi", "Varış Transfer Merkezi",
                               "Araç türü"]].values)) - set(rq)
    if extra:
        flag("F4", f"Tanımsız kiralık hat kullanımı: {sorted(extra)[:3]}")
    dup = kl.groupby(["Araç ID", "day"]).size()
    if (dup > 1).any():
        flag("F4", f"Kiralık araç aynı gün >1 sefer: {dup[dup > 1].index.tolist()[:3]}")

    # ---------------- F6: elleçleme kotası (oransal bölme aritmetiğiyle)
    h_used = {}
    for _, L in legs.iterrows():
        o, d = L["Çıkış Transfer Merkezi"], L["Varış Transfer Merkezi"]
        out_h, in_h = L["Çıkış Elleçleme süresi"], L["Varış elleçleme süresi"]
        out_desi = out_h / H_MIN if out_h > 0 else 0.0
        in_desi = in_h / H_MIN if in_h > 0 else 0.0
        for day, share in split_by_days(L["dep_min"] - out_h, out_h, out_desi):
            h_used[(o, day)] = h_used.get((o, day), 0.0) + share
        for day, share in split_by_days(L["arr_min"], in_h, in_desi):
            h_used[(d, day)] = h_used.get((d, day), 0.0) + share
    for (tm, day), used in sorted(h_used.items()):
        if used > hcap[tm] + 1.0:
            flag("F6", f"{tm} gün {day}: elleçleme {used:,.0f} > kota {hcap[tm]:,.0f}")

    # ---------------- F7: tır kotası
    tir_events = {}
    tl = legs[legs["Araç türü"] == "Tır"]
    for _, L in tl.iterrows():
        for tm, m in [(L["Çıkış Transfer Merkezi"], L["dep_min"]),
                      (L["Varış Transfer Merkezi"], L["arr_min"]),
                      (L["Varış Transfer Merkezi"],
                       L["arr_min"] + L["Varış elleçleme süresi"])]:
            tir_events.setdefault((tm, int(m // 1440)), set()).add(L["Araç ID"])
    for (tm, day), vs in sorted(tir_events.items()):
        if len(vs) > tircap.get(tm, 0):
            flag("F7", f"{tm} gün {day}: {len(vs)} tır > kota {tircap.get(tm, 0)}")

    # ---------------- §10.8: P75 kota stres testi (rapor — hard kısıt DEĞİL)
    # Plan P50 ile kurulur; talep P75 senaryosunda elleçleme kotalarının durumu
    # raporlanır (robustluk sigortası; ihlal değil uyarı üretir).
    qf = os.path.join(BASE, "forecast_quantiles.csv")
    if os.path.exists(qf):
        q = pd.read_csv(qf)
        q["p75"] = q["tahmin"] + 0.526 * (q["p90"] - q["tahmin"]).clip(lower=0)
        rmap = dict(zip(q["Talep ID"],
                        (q["p75"] / q["tahmin"].clip(lower=0.01)).clip(1.0, 3.0)))
        gcols = ["Araç ID", "Çıkış Transfer Merkezi", "Varış Transfer Merkezi", "dep_min"]
        pr = p.copy()
        pr["dep_min"] = [to_min(a, b) for a, b in zip(pr["Çıkış Tarihi"], pr["Çıkış Saati"])]
        pr["ratio"] = pr["base_id"].map(rmap).fillna(1.0)
        pr["wr"] = pr["ratio"] * pr["Taşınan Desi"].clip(lower=0.01)
        lr = (pr.groupby(gcols).apply(
            lambda g_: g_["wr"].sum() / g_["Taşınan Desi"].clip(lower=0.01).sum())
            .to_dict())
        h75 = {}
        for _, L in legs.iterrows():
            key = tuple(L[c] for c in gcols)
            r_ = lr.get(key, 1.0)
            out_h, in_h = L["Çıkış Elleçleme süresi"], L["Varış elleçleme süresi"]
            out_desi = (out_h / H_MIN if out_h > 0 else 0.0) * r_
            in_desi = (in_h / H_MIN if in_h > 0 else 0.0) * r_
            for day, share in split_by_days(L["dep_min"] - out_h, out_h, out_desi):
                h75[(L["Çıkış Transfer Merkezi"], day)] = \
                    h75.get((L["Çıkış Transfer Merkezi"], day), 0.0) + share
            for day, share in split_by_days(L["arr_min"], in_h, in_desi):
                h75[(L["Varış Transfer Merkezi"], day)] = \
                    h75.get((L["Varış Transfer Merkezi"], day), 0.0) + share
        risky = sorted(((tm, day, used / hcap[tm]) for (tm, day), used in h75.items()
                        if used / hcap[tm] > 0.90), key=lambda x: -x[2])
        print(f"  P75 stres testi: %90+ dolan TM-gün sayısı = {len(risky)}"
              + (f" — en kritik: " + ", ".join(
                  f"{tm}@g{day} %{u*100:.0f}" for tm, day, u in risky[:5]) if risky else " ✓"))

    # ---------------- F8: SLA cezası + F10 hazır-olma
    pen_calc = 0.0
    p2 = p.merge(dem[["Varış Transfer Merkezi", "ready_min", "Tahmin Edilen Desi"]],
                 left_on="base_id", right_index=True, suffixes=("", "_dem"))
    p2["dep_min"] = [to_min(a, b) for a, b in zip(p2["Çıkış Tarihi"], p2["Çıkış Saati"])]
    p2["arr_min"] = [to_min(a, b) for a, b in zip(p2["Varış Tarihi"], p2["Varış Saati"])]
    for _, r in p2.iterrows():
        # F10: ilk bacakta elleçleme başlangıcı >= talep tamamlanma
        if r["Çıkış Elleçleme süresi"] > 0:
            h_start = r["dep_min"] - r["Çıkış Elleçleme süresi"]
            if h_start < r["ready_min"] - 1.01:      # SS:DD yuvarlama payı
                flag("F10", f"{r['Talep ID']}: elleçleme {h_start:.0f} < hazır {r['ready_min']:.0f}")
        if r["Varış Transfer Merkezi"] == r["Varış Transfer Merkezi_dem"]:
            od = (dem.loc[r["base_id"], "Çıkış Transfer Merkezi"],
                  r["Varış Transfer Merkezi_dem"])
            deadline = r["ready_min"] + sla[od] * 1440.0
            done = r["arr_min"] + r["Varış elleçleme süresi"]
            late = done - deadline
            want = (r["Taşınan Desi"] * math.ceil(late / 60.0 - 1e-9) * SLA_TL
                    if late > 1e-6 else 0.0)
            pen_calc += want
            if abs(want - r["SLA cezası"]) > 1.0:
                flag("F8", f"{r['Talep ID']}: SLA {r['SLA cezası']:.2f} ≠ hesap {want:.2f}")

    # ---------------- F9: maliyet (araç bazında alt sınır kontrolü)
    cost_calc = 0.0
    for (vid, tip, typ), gl in legs.groupby(["Araç ID", "Araç Tipi", "Araç türü"]):
        gl = gl.sort_values("dep_min")
        hr = rent_hr[typ] if tip == "Kiralık" else spot_hr[typ]
        km_r = rent_km[typ] if tip == "Kiralık" else spot_km[typ]
        usage = 0.0
        km = 0.0
        prev_end, prev_dest = None, None
        for _, L in gl.iterrows():
            usage += L["Çıkış Elleçleme süresi"] + L["Yolculuk süresi"] + L["Varış elleçleme süresi"]
            # zincir içi bekleme: yalnız aynı seferin ardışık bacağı (uğrama) —
            # kiralıkların günler arası boşta durması kullanım süresi DEĞİLDİR
            if (tip != "Kiralık" and prev_end is not None
                    and prev_dest == L["Çıkış Transfer Merkezi"]
                    and L["dep_min"] >= prev_end - 1.5):
                gap = (L["dep_min"] - L["Çıkış Elleçleme süresi"]) - prev_end
                if 1.5 < gap < 720:
                    usage += gap
            km += dist[(L["Çıkış Transfer Merkezi"], L["Varış Transfer Merkezi"])]
            prev_end = L["arr_min"] + L["Varış elleçleme süresi"]
            prev_dest = L["Varış Transfer Merkezi"]
        c = hr * usage / 60.0 + km_r * km
        cost_calc += c
        reported = gl["maliyet"].sum()
        if reported < c - TOL_TL * len(gl):
            flag("F9", f"{vid}: maliyet {reported:,.0f} < hesap {c:,.0f}")

    # ---------------- rapor
    rep_cost = plan["Toplam maliyet"].sum()
    rep_pen = plan["SLA cezası"].sum()
    print(f"  plan satırı: {len(plan)} | araç: {plan['Araç ID'].nunique()}")
    print(f"  rapor edilen maliyet: {rep_cost / 1e6:.3f}M TL | yeniden hesap: {cost_calc / 1e6:.3f}M TL")
    print(f"  rapor edilen SLA: {rep_pen / 1e3:.1f}K TL | yeniden hesap: {pen_calc / 1e3:.1f}K TL")
    hard = [i for i in issues if i[0] in ("F4", "F5", "F6", "F7")]
    print(f"\n  Kural ihlali (hard F4–F7): {len(hard)}")
    print(f"  Diğer bulgular (format/eşleşme/maliyet): {len(issues) - len(hard)}")
    if issues:
        print("\n--- BULGULAR ---")
        for code, msg in issues[:40]:
            print(f"  [{code}] {msg}")
        if len(issues) > 40:
            print(f"  … +{len(issues) - 40} bulgu daha")
        sys.exit(1)
    print("\n  ✅ 0 ihlal — plan ve tahmin dosyaları tüm denetimlerden geçti.")


if __name__ == "__main__":
    main()
