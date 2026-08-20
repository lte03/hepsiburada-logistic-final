# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 — Final Backtest giriş noktası.

    python main.py

Ek argüman, ortam kurulumu veya kullanıcı etkileşimi GEREKTİRMEZ (Bölüm 3).
Yalnızca OPTİMİZASYON adımını çalıştırır; talep tahmini modülü (forecast_v5.py)
çağrılmaz (Bölüm 2).

Girdi (Bölüm 4) — sırayla denenir:
    1) TEKNOFEST_INPUT_FILE ortam değişkenindeki mutlak yol
    2) data/one_week_backtest.xlsx
    3) Talep-tahmini.xlsx  (yalnız yerel geliştirme yedeği)

Çıktı (Bölüm 5): <cikti_klasoru>/Tasima-plani.xlsx — klasör teknofest_manifest.json
içindeki "cikti_klasoru" alanından okunur.
"""
import json
import os
import sys
import time

import optimizer_v5

BASE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(BASE, "teknofest_manifest.json")

ENV_VAR = "TEKNOFEST_INPUT_FILE"
FALLBACK_INPUT = os.path.join("data", "one_week_backtest.xlsx")
DEV_INPUT = "Talep-tahmini.xlsx"

# K2 ALNS bütçesi: teslim koşusuyla birebir aynı (python optimizer_v5.py --iters 1200).
# Deterministik → sonuç makineden bağımsız ve tekrarlanabilir.
ALNS_ITERS = 1200
# Zaman aşımı 100 dk (Bölüm 10). Emniyet freni bunun çok altında tutulur: beklenen
# hacimde ALNS ~7 dk sürer ve fren HİÇ devreye girmez; yalnız girdi beklenenden çok
# büyükse arama kesilir ve geçerli plan yine de yazılır.
ALNS_DEADLINE_S = 60 * 60


def _manifest_outdir():
    try:
        with open(MANIFEST, encoding="utf-8") as f:
            return (json.load(f).get("cikti_klasoru") or "out").strip().rstrip("/\\")
    except (OSError, ValueError):
        return "out"


def _resolve_input():
    """Bölüm 4'teki iki erişim yöntemi + yerel geliştirme yedeği."""
    env = os.environ.get(ENV_VAR, "").strip().strip('"')
    if env:
        if os.path.isfile(env):
            return env, f"{ENV_VAR} ortam değişkeni"
        print(f"UYARI: {ENV_VAR}={env} bulunamadı; yedek yola düşülüyor.", flush=True)
    for rel, etiket in ((FALLBACK_INPUT, "sabit göreli yol"),
                        (DEV_INPUT, "yerel geliştirme yedeği")):
        yol = os.path.join(BASE, rel)
        if os.path.isfile(yol):
            return yol, etiket
    return None, None


def main():
    t0 = time.time()
    if hasattr(sys.stdout, "reconfigure"):     # Türkçe çıktı boru hattında bozulmasın
        sys.stdout.reconfigure(encoding="utf-8")

    girdi, kaynak = _resolve_input()
    if girdi is None:
        print("HATA: talep dosyası bulunamadı.\n"
              f"  {ENV_VAR} ortam değişkenini ayarlayın ya da dosyayı şuraya koyun:\n"
              f"  {os.path.join(BASE, FALLBACK_INPUT)}", file=sys.stderr)
        return 1

    outdir = os.path.join(BASE, _manifest_outdir())
    os.makedirs(outdir, exist_ok=True)

    print("=== TEKNOFEST final backtest — optimizasyon ===")
    print(f"  girdi : {girdi}  ({kaynak})")
    print(f"  çıktı : {os.path.join(outdir, 'Tasima-plani.xlsx')}")

    optimizer_v5.main(["--input", girdi,
                       "--outdir", outdir,
                       "--iters", str(ALNS_ITERS),
                       "--deadline", str(ALNS_DEADLINE_S)])

    hedef = os.path.join(outdir, "Tasima-plani.xlsx")
    if not os.path.isfile(hedef):
        print(f"HATA: çıktı dosyası üretilemedi: {hedef}", file=sys.stderr)
        return 1
    print(f"\nTAMAMLANDI — {time.time() - t0:.0f} sn | {hedef}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
