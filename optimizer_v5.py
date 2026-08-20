# -*- coding: utf-8 -*-
"""
Çalıştırma:  python optimizer_v5.py [--budget 120] [--iters 400] [--ablation]
"""
import argparse
import itertools
import math
import os
import random
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
EPOCH = pd.Timestamp("2026-06-29")          # dakika 0
HORIZON_DAYS = 10                            # 29 Haz – 8 Tem
SLA_TL = 0.4
H_MIN_PER_DESI = 0.01
TOUR_CAND_MAX = 5600.0        # tur adayı OD-gün grubu üst sınırı (desi)
TOUR_MAX_STOPS = 6            # çok-duraklı milk-run: en fazla 6 varış
TOUR_MAX_LOAD = 12000.0       # tur aracı üst sınırı (Kamyon; Tır turlara girmez)
TOUR_DETOUR = 0.85            # durak ekleme kriteri: ek km < 0.85 × direkt km
TYPES = ["Tır", "Kamyon", "Hafif Kamyon", "Kamyonet"]


def day_index(m):
    return int(m // 1440)


def wmin(m):
    """Dosyaya yazılan SS:DD değerinin dakika karşılığı (en yakın dakika)."""
    return math.floor(m + 0.5)


def fmt_dt(m):
    ts = EPOCH + pd.Timedelta(minutes=float(wmin(m)))
    return ts.strftime("%d.%m.%Y"), ts.strftime("%H:%M")


# ================================================================ veri
class World:
    def __init__(self):
        veh = pd.read_excel(os.path.join(DATA, "Araç_Kapasite_Maliyet_Saat.xlsx"))
        self.cap = dict(zip(veh["Araç Adı"], veh["Kapasite (desi)"].astype(float)))
        self.spot_hr = dict(zip(veh["Araç Adı"], veh["Spot Araç Saatlik Kira (TL)"]))
        self.spot_km = dict(zip(veh["Araç Adı"], veh["Spot Kilometre Başına Maliyet (TL)"]))
        self.rent_hr = dict(zip(veh["Araç Adı"], veh["Kiralık Araç Saatlik Kira (TL)"]))
        self.rent_km = dict(zip(veh["Araç Adı"], veh["Kiralık Araç Kilometre Başına Maliyet (TL)"]))

        net = pd.read_excel(os.path.join(DATA, "sehirler_arasi_lojistik.xlsx"))
        self.dist, self.tt, self.sla = {}, {}, {}
        tcol = {"Tır": "Tir_Suresi_Saat", "Kamyon": "Kamyon_Suresi_Saat",
                "Hafif Kamyon": "Hafif_Kamyon_Suresi_Saat", "Kamyonet": "Kamyonet_Suresi_Saat"}
        for _, r in net.iterrows():
            k = (r["cikis"], r["varis"])
            self.dist[k] = float(r["mesafe_km"])
            self.sla[k] = int(r["hedef_teslim_gun"])
            self.tt[k] = {t: float(r[c]) * 60.0 for t, c in tcol.items()}

        ell = pd.read_excel(os.path.join(DATA, "Ellecleme-kapasite.xlsx"))
        self.hcap = dict(zip(ell["transfer_merkezi"], ell["ellecleme_kapasite"]))
        tir = pd.read_excel(os.path.join(DATA, "tir_kapasiteleri v2.xlsx"))
        self.tircap = dict(zip(tir["transfer_merkezi"], tir["tir_kapasitesi"].astype(int)))

        rent = pd.read_excel(os.path.join(DATA, "Kiralık_Araclar.xlsx"))
        self.rentals = []                    # V0001..V0014 sabit fiziksel araçlar
        vid = 1
        for _, r in rent.iterrows():
            for _ in range(int(r["Araç sayısı"])):
                self.rentals.append((f"V{vid:04d}", r["Çıkış Transfer Merkezi"],
                                     r["Varış Transfer Merkezi"], r["Araç Türü"]))
                vid += 1
        self.rental_lines = {}
        for v, o, d, t in self.rentals:
            self.rental_lines.setdefault((o, d), []).append((v, t))
        self.n_rentals = len(self.rentals)

        # R1 (§3.4 yorum riski) — tır kapasitesi sıkışık kiralık VARIŞ merkezleri.
        # Bir kiralık Tır gece taşmasıyla varırsa (örn. 20:45 çıkış → ertesi gün 00:14
        # varış), o varış TM'sinde tek takvim gününde ONUN İKİ AYRI seferi görünür.
        # Şartname "aynı araç aynı TM = 1 slot" derken, Q&A bunu "hareketsiz boşalt-
        # yükle" ile sınırlar; jüri dar yorumu uygularsa ayrı seferler ayrı sayılıp
        # kapasite aşılabilir. Kapasitesi kendisine gelen kiralık Tır sayısına eşit/
        # altındaki varışlar 'sıkışık' sayılır ve bu varışlara giden kiralık Tır'lar
        # AYNI GÜN varmaya zorlanır (dispatch'te latest_end ile) — belirsizlik sıfırlanır.
        rent_tir_dest = {}
        for v, o, d, t in self.rentals:
            if t == "Tır":
                rent_tir_dest.setdefault(d, set()).add(v)
        self.tight_rent_dest = {d for d, vs in rent_tir_dest.items()
                                if self.tircap.get(d, 0) <= len(vs)}

        fc = pd.read_excel(os.path.join(BASE, "Talep-tahmini.xlsx"))
        fc["Tarih"] = pd.to_datetime(fc["Tarih"], format="%d.%m.%Y")
        fc["slot"] = fc["Talep Tamamlama Saati"].astype(str).str[:2]
        self.demands = {}
        for _, r in fc.iterrows():
            slot = "09:00" if r["slot"] == "09" else "17:00"
            ready = ((r["Tarih"] - EPOCH).days * 1440.0 +
                     (9 * 60 if slot == "09:00" else 17 * 60))
            od = (r["Çıkış Transfer Merkezi"], r["Varış Transfer Merkezi"])
            desi = max(float(r["Tahmin Edilen Desi"]), 0.01)
            self.demands[r["Talep ID"]] = dict(
                id=r["Talep ID"], o=od[0], d=od[1], desi=desi, ready=ready,
                deadline=ready + self.sla[od] * 1440.0,
                day=(r["Tarih"] - EPOCH).days, slot=slot)

    def travel(self, o, d, typ):
        return self.tt[(o, d)][typ]


# ================================================================ defterler
class Ledgers:
    def __init__(self, w: World):
        self.w = w
        self.h_used = {}       # (tm, gün) -> desi
        self.tir_used = {}     # (tm, gün) -> set(araç_id)

    def h_rem(self, tm, day):
        return self.w.hcap[tm] - self.h_used.get((tm, day), 0.0)

    def h_add(self, tm, allocs):
        for day, share in allocs:
            self.h_used[(tm, day)] = self.h_used.get((tm, day), 0.0) + share

    def tir_ok(self, vid, tm, days):
        for day in days:
            s = self.tir_used.get((tm, day), set())
            if vid not in s and len(s) >= self.w.tircap.get(tm, 0):
                return False
        return True

    def tir_free(self, tm, day):
        return self.w.tircap.get(tm, 0) - len(self.tir_used.get((tm, day), set()))

    def tir_add(self, vid, tm, days):
        for day in days:
            self.tir_used.setdefault((tm, day), set()).add(vid)

    def prereserve_rentals(self):
        """Zorunlu kiralık tırların slotları önceden rezerve edilir (§2.9).
        Aynı araç aynı TM-günde 1 slot saydığından bu rezervasyon kayıpsızdır;
        spot tırların kiralıkların yerini kapmasını yapısal olarak engeller."""
        for vid, o, d, typ in self.w.rentals:
            if typ != "Tır":
                continue
            for day in range(8):            # 7 sefer günü + gün-aşırı varış payı
                self.tir_add(vid, o, [day])
                self.tir_add(vid, d, [day])

    def schedule_handling(self, tm, earliest, desi, latest_end=None):
        """En erken fizıbıl elleçleme: (start, end, [(gün, desi)]).
        Kota dolarsa başlangıç kaydırılır; gece yarısı oransal bölme (S-C 3K/7K)."""
        if desi <= 0:
            return earliest, earliest, []
        dur = desi * H_MIN_PER_DESI
        t = earliest
        for _ in range(3 * HORIZON_DAYS):
            allocs, ok = self._split(tm, t, dur, desi)
            if ok:
                if latest_end is not None and t + dur > latest_end:
                    return None
                return t, t + dur, allocs
            d0 = day_index(t)
            rem0 = self.h_rem(tm, d0)
            first_share = allocs[0][1] if allocs and allocs[0][0] == d0 else 0.0
            nxt = (d0 + 1) * 1440.0
            if first_share > rem0 + 1e-9:
                if rem0 > 1e-9:
                    t2 = nxt - (rem0 / desi) * dur      # bugünkü pay = kalan kota
                    t = t2 if t2 > t + 1e-9 else nxt
                else:
                    t = nxt
            else:                                        # tıkanma sonraki günde
                t = nxt
        return None

    def _split(self, tm, start, dur, desi):
        allocs, ok, t = [], True, start
        end = start + dur
        while t < end - 1e-9:
            d = day_index(t)
            seg = min(end, (d + 1) * 1440.0) - t
            share = desi * seg / dur
            allocs.append((d, share))
            if share > self.h_rem(tm, d) + 1e-9:
                ok = False
            t = min(end, (d + 1) * 1440.0)
        if not allocs:
            allocs = [(day_index(start), desi)]
            ok = desi <= self.h_rem(tm, day_index(start)) + 1e-9
        return allocs, ok


# ================================================================ merdiven (K1-3)
def spot_unit_cost(w, od, typ, load):
    usage_h = (2 * load * H_MIN_PER_DESI + w.travel(*od, typ)) / 60.0
    return w.spot_hr[typ] * usage_h + w.spot_km[typ] * w.dist[od]


_LADDER_CACHE = {}


def ladder_combos(w, od, load, max_tir, k_best=3):
    """Yükü taşıyan maliyet-optimal tamsayı araç kombinasyonları (ilk k_best).
    Aşırı-kapasite budaması: hiçbir araç türünde tam-araçlık israf bırakılmaz."""
    if load <= 0:
        return [[]]
    key = (od, round(load, 1), max_tir)
    if key in _LADDER_CACHE:
        return _LADDER_CACHE[key]
    cT, cK, cH, cM = (w.cap[t] for t in TYPES)
    res = []
    nT_hi = min(max_tir, math.ceil(load / cT))
    for nT in range(nT_hi + 1):
        r1 = max(0.0, load - nT * cT)
        for nK in range(math.ceil(r1 / cK) + 1):
            r2 = max(0.0, r1 - nK * cK)
            if nK * cK - r1 >= cK:          # israf budaması
                break
            for nH in range(0, 3):
                r3 = max(0.0, r2 - nH * cH)
                if nH * cH - r2 >= cH:
                    break
                nM = math.ceil(r3 / cM) if r3 > 1e-9 else 0
                combo = (["Tır"] * nT + ["Kamyon"] * nK +
                         ["Hafif Kamyon"] * nH + ["Kamyonet"] * nM)
                if not combo:
                    continue
                cost, rem = 0.0, load
                for t in sorted(combo, key=lambda x: -w.cap[x]):
                    lv = min(w.cap[t], rem)
                    rem -= lv
                    cost += spot_unit_cost(w, od, t, lv)
                res.append((cost, combo))
    res.sort(key=lambda x: (x[0], len(x[1])))
    seen, out = set(), []
    for c, combo in res:
        k2 = tuple(sorted(combo))
        if k2 in seen:
            continue
        seen.add(k2)
        out.append(combo)
        if len(out) >= k_best:
            break
    out = out or [["Kamyon"] * math.ceil(load / cK)]
    _LADDER_CACHE[key] = out
    return out


def pack_amounts(amounts, caps):
    """K0 ön-bölme: [(base_id, desi)] → caps sıralı araç manifestleri.
    Kiralık-önce politikası: liste sırasına göre doldurulur; talepler bölünebilir."""
    mani = [[] for _ in caps]
    free = [c for c in caps]
    for bid, desi in sorted(amounts, key=lambda x: -x[1]):
        rem = desi
        for i in range(len(free)):
            if rem <= 1e-9:
                break
            take = min(free[i], rem)
            if take <= 1e-9:
                continue
            mani[i].append((bid, take))
            free[i] -= take
            rem -= take
        if rem > 1e-9:
            return None
    return mani


# ================================================================ simülasyon
class Simulator:
    def __init__(self, w: World):
        self.w = w
        self.base_groups = {}                # (o, d, gün) -> [talep_id]
        for dem in w.demands.values():
            self.base_groups.setdefault((dem["o"], dem["d"], dem["day"]), []).append(dem["id"])
        self.group_tot = {k: sum(w.demands[t]["desi"] for t in v)
                          for k, v in self.base_groups.items()}

    def build_jobs(self, dec):
        """Kararları uygula: (gün, dalga) → {(o,d): [talep_id]}"""
        jobs = {}
        for (o, d, day), ids in self.base_groups.items():
            morning = dec.get(("morning", (o, d, day)), False)
            defer = dec.get(("defer", (o, d, day)), 0)
            for tid in ids:
                dem = self.w.demands[tid]
                if dem["slot"] == "09:00" and morning and defer == 0:
                    jd, jw = day, 9
                elif defer > 0:
                    # V4-B: ertelenen grup hedef günde erken dalgaya inebilir —
                    # talepler zaten önceki günden hazır, 09:00 çıkışı meşru
                    jd = min(day + defer, HORIZON_DAYS - 2)
                    jw = 9 if dec.get(("dwave", (o, d, day)), 17) == 9 else 17
                else:
                    jd, jw = day, 17
                jobs.setdefault((jd, jw), {}).setdefault((o, d), []).append(tid)
        return jobs

    def run(self, dec, emit=False):
        w = self.w
        led = Ledgers(w)
        led.prereserve_rentals()
        state = dict(rows=[], veh_cost=0.0, pen=0.0, spot_seq=w.n_rentals,
                     infeasible=0, gpen={})
        jobs = self.build_jobs(dec)
        completion = {}

        # ---------- tek sefer (1 veya 2 bacak) çizelgele + kaydet
        def dispatch(vid, tip, typ, legs, ready, leg_manifests, hourly, per_km,
                     latest_end=None, commit=True):
            """legs: [(o,d)] veya [(o,A),(A,B)]; leg_manifests[i] = o çıkışında
            i. bacağın SONUNDA inecek [(base_id, desi)]. 2 bacakta araç A'da yalnız
            1. bacak yükünü indirir; 2. bacak yükü araçta kalır (uğrama)."""
            loads = [sum(x[1] for x in m) for m in leg_manifests]
            total_load = sum(loads)
            o0 = legs[0][0]
            sh = led.schedule_handling(o0, ready, total_load, latest_end=latest_end)
            if sh is None:
                return None
            h0, dep, alloc_o = sh
            plan_legs, t_cur, commits = [], dep, [(o0, alloc_o)]
            for i, (lo, ld) in enumerate(legs):
                travel = w.travel(lo, ld, typ)
                arr = t_cur + travel
                unload = loads[i]
                sh2 = led.schedule_handling(ld, arr, unload)
                if sh2 is None:
                    return None
                hs, he, alloc_d = sh2
                commits.append((ld, alloc_d))
                plan_legs.append(dict(o=lo, d=ld, dep=t_cur, arr=arr, travel=travel,
                                      h_end=he, unload=unload))
                t_cur = he
            # tır slotları (yalnız Tır türü; 2 bacaklı seferlerde Tır kullanılmaz)
            if typ == "Tır":
                dys_o = sorted({day_index(h0), day_index(dep)})
                L = plan_legs[0]
                dys_d = sorted({day_index(L["arr"]), day_index(L["h_end"])})
                if not (led.tir_ok(vid, legs[0][0], dys_o) and
                        led.tir_ok(vid, legs[0][1], dys_d)):
                    return None
                if commit:
                    led.tir_add(vid, legs[0][0], dys_o)
                    led.tir_add(vid, legs[0][1], dys_d)
            if not commit:
                return plan_legs
            for tm, al in commits:
                led.h_add(tm, al)
            usage = (dep - h0)
            for L in plan_legs:
                usage += L["travel"] + (L["h_end"] - L["arr"])
            km = sum(w.dist[(L["o"], L["d"])] for L in plan_legs)
            cost = hourly * usage / 60.0 + per_km * km
            state["veh_cost"] += cost

            # satırlar + SLA
            out_h_first = total_load * H_MIN_PER_DESI
            wsum = sum(x[1] * w.dist[(L["o"], L["d"])]
                       for i, L in enumerate(plan_legs)
                       for j in range(i, len(plan_legs))
                       for x in leg_manifests[j] if True) or 1.0
            # ağırlık: her satır (bacak, talep) → desi × bacak_km
            row_specs = []
            for i, L in enumerate(plan_legs):
                onboard = [x for j in range(i, len(leg_manifests)) for x in leg_manifests[j]]
                for bid, desi_s in onboard:
                    row_specs.append((i, L, bid, desi_s,
                                      desi_s * w.dist[(L["o"], L["d"])]))
            wtot = sum(x[4] for x in row_specs) or 1.0
            for i, L, bid, desi_s, wt in row_specs:
                dem = w.demands[bid]
                pen = 0.0
                if L["d"] == dem["d"]:
                    completion[bid] = max(completion.get(bid, 0.0), L["h_end"])
                    # ceza, dosyaya yazılan (yuvarlanmış) zamanlarla hesaplanır —
                    # jüri/validator dosyadan aynı değeri türetir
                    done_w = wmin(L["arr"]) + round(L["unload"] * H_MIN_PER_DESI, 2)
                    late = done_w - dem["deadline"]
                    if late > 1e-6:
                        pen = desi_s * math.ceil(late / 60.0 - 1e-9) * SLA_TL
                        state["pen"] += pen
                        gk = (dem["o"], dem["d"], dem["day"])
                        state["gpen"][gk] = state["gpen"].get(gk, 0.0) + pen
                if emit:
                    dd1, tt1 = fmt_dt(L["dep"]); dd2, tt2 = fmt_dt(L["arr"])
                    state["rows"].append({
                        "Araç ID": vid, "Araç Tipi": tip, "Araç türü": typ,
                        "Çıkış Transfer Merkezi": L["o"], "Varış Transfer Merkezi": L["d"],
                        "Çıkış Tarihi": dd1, "Çıkış Saati": tt1,
                        "Varış Tarihi": dd2, "Varış Saati": tt2,
                        "Talep ID": bid, "Taşınan Desi": round(desi_s, 2),
                        "Yolculuk süresi": round(L["travel"], 1),
                        "Varış elleçleme süresi": round(L["unload"] * H_MIN_PER_DESI, 2),
                        "Çıkış Elleçleme süresi": round(out_h_first if i == 0 else 0.0, 2),
                        "SLA cezası": round(pen, 2),
                        "Toplam maliyet": cost * wt / wtot,
                    })
            if emit and not row_specs:                   # zorunlu boş kiralık seferi
                L = plan_legs[0]
                dd1, tt1 = fmt_dt(L["dep"]); dd2, tt2 = fmt_dt(L["arr"])
                state["rows"].append({
                    "Araç ID": vid, "Araç Tipi": tip, "Araç türü": typ,
                    "Çıkış Transfer Merkezi": L["o"], "Varış Transfer Merkezi": L["d"],
                    "Çıkış Tarihi": dd1, "Çıkış Saati": tt1,
                    "Varış Tarihi": dd2, "Varış Saati": tt2,
                    "Talep ID": "-", "Taşınan Desi": 0.0,
                    "Yolculuk süresi": round(L["travel"], 1),
                    "Varış elleçleme süresi": 0.0, "Çıkış Elleçleme süresi": 0.0,
                    "SLA cezası": 0.0, "Toplam maliyet": cost,
                })
            return plan_legs

        def next_spot_id():
            state["spot_seq"] += 1
            return f"V{state['spot_seq']:04d}"

        # ---------- tek OD işini spot merdivenle sevk et
        def dispatch_spot(o, d, ready, amounts, gkey):
            load = sum(x[1] for x in amounts)
            if load <= 0:
                return
            mt = max(0, min(led.tir_free(o, day_index(ready)),
                            max(led.tir_free(d, day_index(ready)),
                                led.tir_free(d, day_index(ready) + 1))))
            cands = ladder_combos(w, (o, d), load, max_tir=mt, k_best=3)
            combo = cands[min(dec.get(("combo", gkey), 0), len(cands) - 1)]
            mani = pack_amounts(amounts, [w.cap[t] for t in combo])
            if mani is None:
                combo = combo + ["Kamyonet"]
                mani = pack_amounts(amounts, [w.cap[t] for t in combo])
            for typ, mv in zip(combo, mani):
                if not mv:
                    continue
                vid = next_spot_id()
                r = dispatch(vid, "Spot", typ, [(o, d)], ready, [mv],
                             w.spot_hr[typ], w.spot_km[typ])
                if r is None:                    # tır slot reddi → kamyonlara böl
                    n_k = max(1, math.ceil(sum(x[1] for x in mv) / w.cap["Kamyon"]))
                    sub = pack_amounts(mv, [w.cap["Kamyon"]] * n_k)
                    for mv2 in (sub or []):
                        r2 = dispatch(next_spot_id(), "Spot", "Kamyon", [(o, d)],
                                      ready, [mv2], w.spot_hr["Kamyon"],
                                      w.spot_km["Kamyon"])
                        if r2 is None:
                            state["infeasible"] += 1

        # ---------- kronolojik dalga döngüsü
        rental_pending = {(line, day) for line in w.rental_lines for day in range(7)}
        line_carry = {}          # V4-C: kiralık hattına ertesi güne devredilen taşma
        for day in range(HORIZON_DAYS):
            for wave in (9, 17):
                wave_jobs = jobs.get((day, wave), {})
                # +1 dk tampon: SS:DD yuvarlaması hazır-olma kuralını bozmasın
                ready = day * 1440.0 + wave * 60.0 + 1.0
                latest = (day + 1) * 1440.0 - 5
                # 1) kiralık hatlar (yalnız 17 dalgası, gün<7): kiralık-önce doldur
                if wave == 17 and day < 7:
                    for line, vehicles in w.rental_lines.items():
                        ids = wave_jobs.pop(line, [])
                        amounts = line_carry.pop(line, []) + \
                            [(t, w.demands[t]["desi"]) for t in ids]
                        caps = [w.cap[t] for _, t in vehicles]
                        rent_cap = sum(caps)
                        total = sum(x[1] for x in amounts)
                        spot_amounts = []
                        if total > rent_cap:                 # taşan kısım spota
                            mani_all = pack_amounts(amounts, caps + [total - rent_cap + 1])
                            rent_mani = mani_all[:-1]
                            spot_amounts = mani_all[-1]
                        else:
                            rent_mani = pack_amounts(amounts, caps) or [[] for _ in caps]
                        for (vid, typ), mv in zip(vehicles, rent_mani):
                            # R1: sıkışık varışa giden kiralık Tır AYNI GÜN varmalı
                            # (görünür varış = çıkış-elleçleme bitişi + seyir < gün sonu).
                            # Bu, gece taşan varışın (örn. 00:14) o TM'de tek günde iki
                            # sefer görünmesini engeller — §3.4 dar yorum riski sıfırlanır.
                            le = latest
                            mv_use = mv
                            if typ == "Tır" and line[1] in w.tight_rent_dest:
                                travel = w.travel(line[0], line[1], typ)
                                le = min(latest, (day + 1) * 1440.0 - 6.0 - travel)
                                # R1-opt: tümüyle boşaltmak yerine AYNI-GÜN varabilecek
                                # MAKSİMUM yükü kiralıkta tut, fazlasını spota devret —
                                # bedava kiralık kapasitesi boşa gitmesin. (Bölünen talep
                                # D-1/D-2 olur; validator desi bütünlüğünü teyit eder.)
                                max_load = (le - ready) / H_MIN_PER_DESI
                                if 0 < max_load < sum(x[1] for x in mv):
                                    keep, spill, acc = [], [], 0.0
                                    for bid, dd in sorted(mv, key=lambda x: -x[1]):
                                        room = max_load - acc
                                        if room <= 1e-9:
                                            spill.append((bid, dd))
                                        elif dd <= room:
                                            keep.append((bid, dd)); acc += dd
                                        else:
                                            keep.append((bid, room))
                                            spill.append((bid, dd - room)); acc = max_load
                                    mv_use = keep
                                    spot_amounts = spot_amounts + spill
                            r = dispatch(vid, "Kiralık", typ, [line], ready, [mv_use],
                                         w.rent_hr[typ], w.rent_km[typ],
                                         latest_end=le)
                            if r is None:        # aynı-gün varamadı → boş öğle çıkışı, yük spota
                                dispatch(vid, "Kiralık", typ, [line],
                                         day * 1440.0 + 12 * 60, [[]],
                                         w.rent_hr[typ], w.rent_km[typ],
                                         latest_end=latest)
                                spot_amounts = spot_amounts + mv_use
                        rental_pending.discard((line, day))
                        if spot_amounts:
                            # V4-C: taşma, karar açıksa ertesi günün boş kiralık
                            # kapasitesine devredilir (bedava kapasite)
                            if dec.get(("rspill", (line, day)), False) and day < 6:
                                line_carry[line] = (line_carry.get(line, [])
                                                    + spot_amounts)
                            else:
                                dispatch_spot(line[0], line[1], ready, spot_amounts,
                                              (line[0], line[1], day, wave))
                # 2) milk-run turları (K1-4 genişletilmiş): aynı çıkışın küçük
                #    OD-gün gruplarını çok-duraklı tek araçta topla.
                #    Maliyet anatomisi kanıtı: dolu<%50 spot bacaklar 11.3M TL ve
                #    aynı (çıkış, gün)de ortalama 6.5 farklı küçük varış var.
                if dec.get("policy_tours", True):     # V4: turlar iki dalgada da
                    cand = {od: ids for od, ids in wave_jobs.items()
                            if od not in w.rental_lines
                            and self._tot(ids) <= TOUR_CAND_MAX}
                    by_origin = {}
                    for (o, d) in cand:
                        by_origin.setdefault(o, []).append(d)
                    for o, dests in by_origin.items():
                        pool = sorted(dests, key=lambda x: -w.dist[(o, x)])
                        while pool:
                            anchor = pool.pop(0)          # en uzak varış = çapa
                            stops = [anchor]
                            load = self._tot(cand[(o, anchor)])
                            while len(stops) < TOUR_MAX_STOPS:
                                best = None
                                base_km = self._route_km(o, self._order(o, stops))
                                for d2 in pool:
                                    nl = load + self._tot(cand[(o, d2)])
                                    if nl > TOUR_MAX_LOAD:
                                        continue
                                    inc = (self._route_km(o, self._order(o, stops + [d2]))
                                           - base_km)
                                    if inc < TOUR_DETOUR * w.dist[(o, d2)]:
                                        if best is None or inc < best[0]:
                                            best = (inc, d2)
                                if best is None:
                                    break
                                stops.append(best[1])
                                pool.remove(best[1])
                                load += self._tot(cand[(o, best[1])])
                            if len(stops) < 2:
                                continue                   # solo kalsın (normal akış)
                            # durak-başına ceza tavanı: cezası kendi solo maliyetini
                            # aşan durak turdan düşürülür (tekrar solo/merdivene döner)
                            def tour_eval(stops_, order_=None):
                                if order_ is None:
                                    order_ = self._order(o, stops_)
                                load_ = sum(self._tot(cand[(o, s)]) for s in stops_)
                                typ_ = next((t for t in ["Kamyonet", "Hafif Kamyon", "Kamyon"]
                                             if w.cap[t] >= load_), "Kamyon")
                                legs_ = [(o, order_[0])] + list(zip(order_[:-1], order_[1:]))
                                km_ = sum(w.dist[l_] for l_ in legs_)
                                t_c = ready + load_ * H_MIN_PER_DESI
                                usage_ = load_ * H_MIN_PER_DESI
                                pen_stop = {}
                                for (lo, ld) in legs_:
                                    t_c += w.travel(lo, ld, typ_)
                                    unload = self._tot(cand[(o, ld)])
                                    t_c += unload * H_MIN_PER_DESI
                                    usage_ += w.travel(lo, ld, typ_) + unload * H_MIN_PER_DESI
                                    ps = 0.0
                                    for tid in cand[(o, ld)]:
                                        late = t_c - w.demands[tid]["deadline"]
                                        if late > 0:
                                            ps += (w.demands[tid]["desi"]
                                                   * math.ceil(late / 60.0) * SLA_TL)
                                    pen_stop[ld] = ps
                                cost_ = w.spot_hr[typ_] * usage_ / 60.0 + w.spot_km[typ_] * km_
                                return order_, typ_, legs_, cost_, pen_stop

                            def solo_of(s):
                                t_ = self._tot(cand[(o, s)])
                                return min(spot_unit_cost(w, (o, s), t2, t_)
                                           for t2 in TYPES if w.cap[t2] >= t_)

                            while len(stops) >= 2:
                                order_, typ, legs_t, tour_cost, pen_stop = tour_eval(stops)
                                bad = [s for s in stops if pen_stop[s] > 0.8 * solo_of(s)]
                                if not bad:
                                    break
                                worst = max(bad, key=lambda s: pen_stop[s])
                                stops.remove(worst)        # solo akışına geri döner
                            if len(stops) < 2:
                                continue
                            # V4-A: durak sırası araması — ≤4 durak tam permütasyon,
                            # 5-6 durakta 2-opt; amaç = maliyet + SLA-ceza toplamı
                            base_ord = self._order(o, stops)
                            if len(stops) <= 4:
                                cands_ord = [list(t_) for t_ in itertools.permutations(stops)]
                            else:
                                cands_ord = [base_ord]
                                for i_ in range(len(base_ord) - 1):
                                    for j_ in range(i_ + 1, len(base_ord)):
                                        alt = list(base_ord)
                                        alt[i_], alt[j_] = alt[j_], alt[i_]
                                        cands_ord.append(alt)
                            best_t = None
                            for ord2 in cands_ord:
                                o2, t2, l2, c2, ps2 = tour_eval(stops, ord2)
                                obj2 = c2 + sum(ps2.values())
                                if best_t is None or obj2 < best_t[0]:
                                    best_t = (obj2, o2, t2, l2, c2, ps2)
                            _, order_, typ, legs_t, tour_cost, pen_stop = best_t
                            pen_est = sum(pen_stop.values())
                            solo_cost = sum(solo_of(s) for s in stops)
                            if tour_cost + pen_est >= solo_cost:
                                continue                   # çapa solo kalır
                            manis = [[(t, w.demands[t]["desi"]) for t in cand[(o, s)]]
                                     for s in order_]
                            r = dispatch(next_spot_id(), "Spot", typ, legs_t, ready,
                                         manis, w.spot_hr[typ], w.spot_km[typ])
                            if r is not None:
                                for s in order_:
                                    wave_jobs.pop((o, s), None)
                            # r None ise gruplar wave_jobs'ta kalır → solo sevk
                # 3) kalan işler: merdivenle
                for (o, d), ids in sorted(wave_jobs.items(),
                                          key=lambda kv: -self._tot(kv[1])):
                    amounts = [(t, w.demands[t]["desi"]) for t in ids]
                    dispatch_spot(o, d, ready, amounts, (o, d, day, wave))
            # 4) o gün hiç talebi olmayan kiralık hatlar: zorunlu boş çıkış
            if day < 7:
                for line, vehicles in w.rental_lines.items():
                    if (line, day) not in rental_pending:
                        continue
                    rental_pending.discard((line, day))
                    for vid, typ in vehicles:
                        dispatch(vid, "Kiralık", typ, [line], day * 1440.0 + 12 * 60,
                                 [[]], w.rent_hr[typ], w.rent_km[typ],
                                 latest_end=(day + 1) * 1440.0 - 5)

        total = state["veh_cost"] + state["pen"]
        if emit:
            return state, completion
        return total, state["veh_cost"], state["pen"], state["infeasible"], state["gpen"]

    def _tot(self, ids):
        return sum(self.w.demands[t]["desi"] for t in ids)

    def _order(self, o, stops):
        """Tur durak sırası: çıkışa yakından uzağa (çapa en sonda bırakılır)."""
        return sorted(stops, key=lambda x: self.w.dist[(o, x)])

    def _route_km(self, o, order_):
        km, cur = 0.0, o
        for s in order_:
            km += self.w.dist[(cur, s)]
            cur = s
        return km


# ================================================================ K2 — ALNS-lite
def improve(sim, dec, budget_s, seed=42, verbose=True, max_iters=None):
    """K2 ALNS-lite. max_iters verilirse (R2) durma ölçütü wall-clock yerine sabit
    iterasyon sayısıdır → sonuç makineden bağımsız ve tam tekrarlanabilir olur
    (seed zaten sabit; döngü gövdesinde saat bağımlılığı yok)."""
    rng = random.Random(seed)
    best_cost, bv, bp, _, gpen = sim.run(dec)
    best, cur, cur_cost = dict(dec), dict(dec), best_cost
    T = max(best_cost * 0.001, 1.0)
    t0, it = time.time(), 0
    groups = list(sim.base_groups.keys())
    hot = [g for g, _ in sorted(gpen.items(), key=lambda kv: -kv[1])[:60]]
    while (it < max_iters) if max_iters is not None else (time.time() - t0 < budget_s):
        it += 1
        cand = dict(cur)
        op = rng.random()
        # %40: ceza-yoğun gruba hedefli hamle (erteleme kısalt / sabaha al)
        if op < 0.40 and hot:
            g = rng.choice(hot)
            o, d, day = g
            k = ("defer", g)
            if cand.get(k, 0) > 0 and rng.random() < 0.7:
                cand[k] = cand[k] - 1
            else:
                k = ("morning", g)
                cand[k] = not cand.get(k, False)
        else:
            g = rng.choice(groups)
            o, d, day = g
            if op < 0.60:
                k = ("combo", (o, d, day, 17))
                cand[k] = (cand.get(k, 0) + 1) % 3
            elif op < 0.78:
                if day < 6:
                    k = ("defer", g)
                    step = 1 if rng.random() < 0.5 else -1
                    cand[k] = max(0, min(6 - day, cand.get(k, 0) + step))
            elif op < 0.88:                            # V4-B: erken dalga aç/kapa
                if cand.get(("defer", g), 0) > 0:
                    k = ("dwave", g)
                    cand[k] = 9 if cand.get(k, 17) == 17 else 17
            elif op < 0.95:
                k = ("morning", g)
                cand[k] = not cand.get(k, False)
            else:                                      # V4-C: rspill aç/kapa
                rl = list(sim.w.rental_lines)
                if rl:
                    line = rng.choice(rl)
                    k = ("rspill", (line, rng.randrange(6)))
                    cand[k] = not cand.get(k, False)
        c, v, p, inf, gp = sim.run(cand)
        dlt = c - cur_cost
        if dlt < 0 or rng.random() < math.exp(-dlt / max(T, 1e-9)):
            cur, cur_cost = cand, c
            if c < best_cost:
                best, best_cost = dict(cand), c
                hot = [g for g, _ in sorted(gp.items(), key=lambda kv: -kv[1])[:60]]
        T *= 0.997
    if verbose:
        print(f"  K2 ALNS-lite: {it} iterasyon → {best_cost / 1e6:.3f}M TL")
    return best, best_cost


# ================================================================ bölünmüş ID adlandırma
def assign_split_ids(plan):
    """Bir talep birden çok araca bölünmüşse D00001-1, D00001-2 … (S-C formatı).
    Aynı aracın uğramalı seferinde talep birden çok bacakta görünür — bu bölme değildir."""
    plan = plan.copy()
    carrier = plan[plan["Talep ID"] != "-"].groupby("Talep ID")["Araç ID"].nunique()
    for tid, n in carrier.items():
        if n <= 1:
            continue
        m = plan["Talep ID"] == tid
        sub = plan.loc[m].sort_values(["Çıkış Tarihi", "Çıkış Saati", "Araç ID"])
        vmap = {v: i + 1 for i, v in enumerate(dict.fromkeys(sub["Araç ID"]))}
        plan.loc[m, "Talep ID"] = [f"{tid}-{vmap[v]}" for v in plan.loc[m, "Araç ID"]]
    return plan


# ================================================================ ana akış
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=60.0,
                    help="K2 ALNS zaman bütçesi (saniye); 0 = yalnız K1")
    ap.add_argument("--iters", type=int, default=0,
                    help="K2 ALNS iterasyon bütçesi (restart başına). >0 ise wall-"
                         "clock yerine kullanılır → makineden bağımsız, tekrarlanabilir "
                         "(R2). Teslim koşusu için önerilir.")
    ap.add_argument("--ablation", action="store_true",
                    help="politika anahtarları için leave-one-out K1 tablosu (§7)")
    args = ap.parse_args()

    t0 = time.time()
    print("=== optimizer_v5 — veri yükleme ===")
    w = World()
    print(f"  talep: {len(w.demands)} | kiralık araç: {w.n_rentals} | OD: {len(w.dist)}")
    sim = Simulator(w)

    # ---- K1: kural-tabanlı başlangıç kararları — çok-günlü havuzlama (chain-pooling)
    # Küçük OD-gün grupları, ceza < tasarruf kaldıkça sonraki günlere taşınıp
    # havuzlanır: SLA-2'de ilk gün bedava (48 saat slack), SLA-1'de gün başına
    # ceza desi × 24 sa × 0.4 TL. Havuz yeterince büyüyünce (veya 3 gün / hafta
    # sonu sınırında) topluca sevk edilir; mikro talepler araç üretmeye zorlamaz.
    dec = {}
    od_days = {}
    for (o, d, day), ids in sim.base_groups.items():
        od_days.setdefault((o, d), {})[day] = sim.group_tot[(o, d, day)]
    for (o, d), dm in od_days.items():
        if (o, d) in w.rental_lines:
            continue
        sla_g = w.sla[(o, d)]
        solo_k = lambda t_: spot_unit_cost(w, (o, d), "Kamyonet", t_)
        def pen_at(tot_, age_):                        # age_ gün beklemenin cezası
            free = 1 if sla_g == 2 else 0
            return tot_ * 24 * SLA_TL * max(0, age_ - free)

        carried = []                                   # (gün, tot)
        for day in range(7):
            tot_today = dm.get(day, 0.0)
            pool = sum(t_ for _, t_ in carried) + tot_today
            # yaş kontrolü: bir gün daha beklemek herhangi bir grubun ceza
            # bütçesini (0.6 × solo) aşacaksa şimdi boşalt
            age_break = any(pen_at(t_, day + 1 - d_) >= 0.6 * solo_k(max(t_, 50))
                            for d_, t_ in carried)
            flush = (day == 6 or tot_today >= 2000.0 or pool >= TOUR_CAND_MAX
                     or age_break or (carried and day - carried[0][0] >= 3))
            if flush:
                for d_, t_ in carried:
                    dec[("defer", (o, d, d_))] = day - d_
                carried = []
                continue                               # bugünün grubu yerinde kalır
            if tot_today <= 0:
                continue
            if (tot_today <= TOUR_CAND_MAX
                    and pen_at(tot_today, 1) < 0.6 * solo_k(max(tot_today, 50))):
                carried.append((day, tot_today))
        for d_, t_ in carried:                         # emniyet: kalan varsa 6. güne
            dec[("defer", (o, d, d_))] = 6 - d_

    # V4-B: dwave (ertelenen grubu 09:00 dalgasına indirme) K1'de TOHUMLANMAZ —
    # battaniye uygulama 17:00 havuzuyla birleşmeyi bozup araç maliyetini artırıyor
    # (ablasyonla görüldü). Karar grup bazında ALNS mutasyonuna bırakılır.

    # V4-C tohumu: kiralık hat taşması ertesi günün boş kiralığına devredilsin
    for line, vehicles in w.rental_lines.items():
        rent_cap = sum(w.cap[t] for _, t in vehicles)
        for day in range(6):
            gt = sim.group_tot.get((line[0], line[1], day), 0.0)
            nt = sim.group_tot.get((line[0], line[1], day + 1), 0.0)
            over = gt - rent_cap
            if over <= 0 or nt + over > rent_cap:
                continue
            if w.sla[line] == 2 or over * 24 * SLA_TL < 0.6 * spot_unit_cost(
                    w, line, "Kamyonet", max(over, 50)):
                dec[("rspill", (line, day))] = True
    c0, v0, p0, inf0, _ = sim.run(dec)
    print(f"  K1: toplam {c0 / 1e6:.3f}M TL (araç {v0 / 1e6:.3f}M + SLA {p0 / 1e3:.1f}K)"
          f" | infeasible={inf0} | {time.time() - t0:.0f} sn")

    # ---- ablasyon (§7): politika anahtarları leave-one-out
    abl_rows = []
    if args.ablation:
        no_defer = {k: v for k, v in dec.items() if k[0] != "defer"}
        variants = [
            ("tam K1 (tur + erteleme)", dec),
            ("tur KAPALI", {**dec, "policy_tours": False}),
            ("erteleme KAPALI", no_defer),
            ("ikisi de KAPALI", {**no_defer, "policy_tours": False}),
        ]
        print("\n  --- ABLASYON (K1, ALNS'siz) ---")
        for name, dv in variants:
            ca, va, pa, _, _ = sim.run(dv)
            abl_rows.append((name, ca, va, pa))
            print(f"    {name}: {ca / 1e6:.3f}M TL (araç {va / 1e6:.3f}M + SLA {pa / 1e3:.1f}K)")

    # ---- K2: ALNS-lite (anytime) — V4-D: çoklu-restart
    # R2: --iters verilirse durma ölçütü sabit iterasyon → sonuç makineden bağımsız
    # ve tekrarlanabilir. Aksi halde wall-clock (--budget) anytime davranışı korunur.
    c_best = c0
    use_iters = args.iters > 0
    if use_iters or args.budget > 0:
        best_dec = dict(dec)
        seg = args.budget / 3.0
        iters_seg = args.iters if use_iters else None
        for sd in (42, 1337, 2026):
            d_i, c_i = improve(sim, dict(dec), seg, seed=sd, verbose=False,
                               max_iters=iters_seg)
            if c_i < c_best:
                best_dec, c_best = d_i, c_i
        dec = best_dec
        if use_iters:
            print(f"  K2 ALNS-lite (3 restart × {args.iters} iterasyon, "
                  f"deterministik): → {c_best / 1e6:.3f}M TL")
        else:
            print(f"  K2 ALNS-lite (3 restart × {seg:.0f} sn): → {c_best / 1e6:.3f}M TL")

    # ---- final plan + K3 (dakika-düzeyi zamanlar simülasyondan)
    state, completion = sim.run(dec, emit=True)
    plan = pd.DataFrame(state["rows"])
    plan = assign_split_ids(plan)
    plan["Toplam maliyet"] = plan["Toplam maliyet"].round(2)
    cols = ["Araç ID", "Araç Tipi", "Araç türü", "Çıkış Transfer Merkezi",
            "Varış Transfer Merkezi", "Çıkış Tarihi", "Çıkış Saati", "Varış Tarihi",
            "Varış Saati", "Talep ID", "Taşınan Desi", "Yolculuk süresi",
            "Varış elleçleme süresi", "Çıkış Elleçleme süresi", "SLA cezası",
            "Toplam maliyet"]
    plan = plan[cols]
    out = os.path.join(BASE, "Tasima-plani.xlsx")
    plan.to_excel(out, index=False, engine="openpyxl")

    # ---- rapor
    base_served = {t.split("-")[0] for t in plan["Talep ID"] if t != "-"}
    missing = set(w.demands) - base_served
    n_spot = plan.loc[plan["Araç Tipi"] == "Spot", "Araç ID"].nunique()
    print("\n=== SONUÇ ===")
    print(f"  plan satırı: {len(plan)} | spot araç: {n_spot} | kiralık: {w.n_rentals}")
    print(f"  araç maliyeti: {state['veh_cost'] / 1e6:.3f}M TL | "
          f"SLA cezası: {state['pen'] / 1e3:.1f}K TL | "
          f"TOPLAM: {(state['veh_cost'] + state['pen']) / 1e6:.3f}M TL")
    print(f"  taşınmayan talep: {len(missing)}" +
          (f" — UYARI: {sorted(missing)[:8]}" if missing else " ✓"))
    daily = plan.groupby("Çıkış Tarihi")["Taşınan Desi"].sum()
    for k_, v_ in sorted(daily.items(), key=lambda x: pd.to_datetime(x[0], format="%d.%m.%Y")):
        print(f"    {k_}: {v_ / 1e6:.2f}M desi sevk")
    print(f"  yazıldı: {out} | toplam süre {time.time() - t0:.0f} sn")

    # ---- Sonuç raporu (markdown): Sonuc_Raporu.md
    n_tour = (plan[plan["Araç Tipi"] == "Spot"].groupby("Araç ID")
              [["Çıkış Transfer Merkezi", "Varış Transfer Merkezi"]]
              .nunique().max(axis=1) > 1).sum()
    md = []
    md.append("# Taşıma Planı — Koşu Sonucu\n")
    md.append(f"- **Koşu zamanı:** {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}")
    if use_iters:
        md.append(f"- **ALNS bütçesi:** 3 × {args.iters} iterasyon (deterministik, R2)")
    else:
        md.append(f"- **ALNS bütçesi:** {args.budget:.0f} sn (wall-clock)")
    md.append(f"- **Toplam süre:** {time.time() - t0:.0f} sn\n")
    md.append("## Maliyet Özeti\n")
    md.append("| Kalem | Değer |")
    md.append("|---|---|")
    md.append(f"| K1 başlangıç (kural-tabanlı) | {c0 / 1e6:.3f}M TL |")
    if use_iters or args.budget > 0:
        md.append(f"| K2 ALNS sonrası | {c_best / 1e6:.3f}M TL |")
    md.append(f"| Araç maliyeti | {state['veh_cost'] / 1e6:.3f}M TL |")
    md.append(f"| SLA cezası | {state['pen'] / 1e3:.1f}K TL |")
    md.append(f"| **TOPLAM** | **{(state['veh_cost'] + state['pen']) / 1e6:.3f}M TL** |\n")
    md.append("## Plan Özeti\n")
    md.append("| Metrik | Değer |")
    md.append("|---|---|")
    md.append(f"| Plan satırı | {len(plan)} |")
    md.append(f"| Spot araç | {n_spot} (çok-duraklı tur: {n_tour}) |")
    md.append(f"| Kiralık araç | {w.n_rentals} (7 gün zorunlu çıkış) |")
    md.append(f"| Taşınmayan talep | {len(missing)} |\n")
    md.append("## Günlük Sevk (çıkış günü, desi)\n")
    md.append("| Gün | Sevk |")
    md.append("|---|---|")
    for k_, v_ in sorted(daily.items(), key=lambda x: pd.to_datetime(x[0], format="%d.%m.%Y")):
        md.append(f"| {k_} | {v_ / 1e6:.2f}M |")
    if abl_rows:
        md.append("\n## Ablasyon (K1, ALNS'siz — §7 politika anahtarları)\n")
        md.append("| Varyant | Toplam | Araç | SLA |")
        md.append("|---|---|---|---|")
        for name, ca, va, pa in abl_rows:
            md.append(f"| {name} | {ca / 1e6:.3f}M TL | {va / 1e6:.3f}M | {pa / 1e3:.1f}K |")
    md.append("\n> Doğrulama için `python validator.py` çalıştırın (0-ihlal + P75 stres raporu).\n")
    rpath = os.path.join(BASE, "Sonuc_Raporu.md")
    with open(rpath, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"  rapor: {rpath}")


if __name__ == "__main__":
    main()
