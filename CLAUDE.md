# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proje

TEKNOFEST 2026 "Yapay Zeka Destekli Lojistik Anahat Optimizasyonu" — Gelişmiş Çözüm Aşaması teslimi.
İki teslim dosyası üretilir: **`Talep-tahmini.xlsx`** (29 Haz – 5 Tem 2026 talep tahmini) ve
**`Tasima-plani.xlsx`** (dakika çözünürlüklü araç/sefer planı).

Kod, yorumlar, konsol çıktısı ve dokümanlar **Türkçe**dir; bu dili koru. Excel sütun adları
Türkçe ve aksanlıdır (`Çıkış Transfer Merkezi`, `Araç türü` — küçük "t" bilinçli) ve şablonla
birebir eşleşmek zorundadır; `validator.py` F1 bunu denetler.

## Komutlar

```bash
pip install -r requirements.txt

python forecast_v5.py                 # 1) tahmin → Talep-tahmini.xlsx (+ quantiles/meta), ~1-2 dk
python optimizer_v5.py --iters 1200   # 2) plan   → Tasima-plani.xlsx + Sonuc_Raporu.md, ~8 dk
python validator.py                   # 3) bağımsız denetim; ihlal varsa exit 1
```

- `--iters N` → durma ölçütü sabit iterasyon (3 restart × N, seed 42/1337/2026): makineden
  bağımsız, **tekrarlanabilir** teslim koşusu. Teslim çıktısı üretirken bunu kullan.
- `--budget SANIYE` (varsayılan 60) → wall-clock anytime mod; sonuç makineye göre değişir.
  `--budget 0` (ve `--iters` verilmemişse) = yalnız K1 kural-tabanlı başlangıç, hızlı duman testi.
- `--ablation` → K1 politika anahtarları için leave-one-out tablosu (tur / erteleme aç-kapa).

Ayrı test paketi yoktur: **`validator.py` bu projenin test paketidir** (F1–F10, "0 ihlal" hedefi).
Sadece optimizer'ı değiştirdiysen `forecast_v5.py` yeniden koşulmaz — `Talep-tahmini.xlsx`
diskte kalır ve girdi olarak okunur.

## Boru hattı ve dosya bağımlılıkları

```
data/*.xlsx ─┬─> forecast_v5.py ─> Talep-tahmini.xlsx ───┬─> optimizer_v5.py ─> Tasima-plani.xlsx
             │                     forecast_quantiles.csv │                     Sonuc_Raporu.md
             │                     forecast_meta.json     │
             └─────────────────────────────────────────> validator.py <─────────┘
```

`data/` girdileri: `teknofest26_gelismis.xlsx` (66.024 satırlık geçmiş talep),
`sehirler_arasi_lojistik.xlsx` (mesafe + araç türü başına seyir saati + SLA günü),
`Araç_Kapasite_Maliyet_Saat.xlsx`, `Kiralık_Araclar.xlsx`, `Ellecleme-kapasite.xlsx`,
`tir_kapasiteleri v2.xlsx`. Üç script de yolları `BASE`/`DATA` üzerinden mutlak kurar.

`README_Gelismis_Cozum_FINAL.md` metodoloji referansıdır; koddaki `(§2.9)`, `(§10.1)` gibi
atıflar bu dosyanın bölüm numaralarına gider. Bir sabiti veya politikayı değiştirmeden önce
ilgili bölümü oku — çoğu "sihirli sayı"nın orada veriyle gerekçesi var.

## Problem sabitleri (üç dosyada da aynı olmalı)

- `EPOCH = 2026-06-29`, zaman **dakika** ofseti; ufuk 10 gün (7 sevk günü + taşma).
- Talep sadece iki slotta hazır olur: 09:00 ve 17:00. Tahmin ızgarası **289 OD × 7 gün × 2 slot
  = 4.046 satır** (hem forecast hem validator bunu assert eder).
- Elleçleme: `0.01 dk/desi`, çıkışta ve varışta ayrı ayrı sayılır; TM-gün desi **kotası** hard
  kısıttır ve gece yarısını aşan elleçleme **oransal** olarak iki güne bölünür.
- SLA cezası: `desi × ceil(gecikme_saat) × 0.4 TL`; gecikme = varış elleçlemesinin bitişi −
  (hazır olma + SLA günü).
- Araç maliyeti = `saatlik_kira × kullanım_saati + km_maliyeti × km`; kullanım = elleçleme +
  seyir + (spot için) zincir içi bekleme. Kiralık aracın günler arası boşta beklemesi **sayılmaz**.
- Tır kotası: TM-gün başına tekil Tır aracı sayısı; kiralık tırların slotları önden rezerve edilir.
- Araç ID: kiralıklar `V0001`–`V0014` (sabit fiziksel araçlar, 7 gün **zorunlu** çıkış),
  spot araçlar `V0015`'ten devam eder. Bölünen talep `D00001-1`, `D00001-2` … olur;
  aynı aracın çok-duraklı turunda talebin birden çok bacakta görünmesi bölme **değildir**.

## forecast_v5.py mimarisi

- **Sıfır-dolgulu panel**: her (OD, slot, tarih) hücresi satır olur; gözlenmeyen geçmiş hücre
  0'dır (panelin ~%36'sı örtük sıfır). Padding kaldırılırsa model sistematik aşırı tahmin yapar.
- **Takvim geometrisi** ana sinyaldir: `dtme` (ay sonuna kalan gün) / `dfs` (ay başından geçen
  gün). Ay sonu talebi normalin ~%2'sine çöker, ay başı sıçrar. `HOLIDAYS`/`KURBAN_WEEK`/`28-06`
  eğitimden dışlanır ve lag/rolling kaynaklarından maskelenir (`ha_prev_sources` —
  holiday-aware aynı-DOW geçmiş).
- **Ensemble**: slot başına çekirdek LGBM (L1-quantile vs Tweedie, ağırlıklı CV ile seçilir)
  + HA-medyan-6w + HA-lag7 + (yalnız 09:00) TSB. Ağırlıklar **NNLS** ile foldlar üzerinden
  çözülür; ensemble HA-medyanı geçemezse güvenlik eşiği devreye girip tüm ağırlığı ona verir.
  CV foldları 4 "analog" (ay sınırı kesen) + 2 "ay-ortası"; skor `0.7×analog + 0.3×orta`.
- **Korkuluklar** (`apply_month_rules`): ay-sınırı override çarpanı `k30` sabit değil —
  `{0.02, 0.05, 0.08}` gridi analog foldlarda backtest edilir, kazanan `k30*` finalde kullanılır.
  `CLIP_*` sabitleri override'ların bant sınırlarıdır.
- Çıktı `Tahmin Edilen Desi` alt sınırı 0.01 (plan tarafında sıfıra bölme olmasın diye).

## optimizer_v5.py mimarisi

Dört katman, tek dosyada:

- **`World`** — tüm Excel girdilerini sözlüklere okur (kapasite, maliyet, mesafe, seyir süresi,
  SLA, elleçleme/tır kotaları, kiralık hatlar, talepler). `tight_rent_dest`: tır kotası kiralık
  filoyla dolan varış TM'leri; oraya giden kiralık tırlar **aynı gün** varmaya zorlanır
  (gece taşan varışın tek takvim gününde iki sefer sayılma riskini sıfırlar).
- **`Ledgers`** — elleçleme kotası ve tır slotu defterleri. `schedule_handling` kota dolduğunda
  başlangıcı ileri kaydırarak en erken fizıbıl elleçleme penceresini bulur; hard kısıtlar
  **inşa sırasında** korunur, sonradan onarılmaz.
- **`Simulator`** — kararları tam plana çevirir. `run(dec)` deterministiktir ve
  `(toplam, araç_maliyeti, ceza, infeasible, grup_bazlı_ceza)` döner; `emit=True` ile plan
  satırlarını üretir. Akış: kronolojik (gün × dalga 9/17) döngüsünde önce **kiralık hatlar**
  doldurulur, sonra **milk-run turları** kurulur (aynı çıkıştan ≤6 durak, sapma
  < `TOUR_DETOUR` × direkt km), kalanlar **maliyet merdiveniyle** (`ladder_combos` — tamsayı
  araç kombinasyonları, israf budamalı) spot araçlara verilir.
- **Karar sözlüğü `dec`** — arama uzayının tamamı; anahtar biçimleri:
  `("defer", (o,d,gün)) -> kaç gün ertele`, `("morning", (o,d,gün)) -> bool` (09:00 dalgasında çık),
  `("dwave", (o,d,gün)) -> 9|17` (ertelenen grup hedef günde hangi dalgada),
  `("combo", (o,d,gün,dalga)) -> merdiven adayı indeksi 0..2`,
  `("rspill", ((o,d),gün)) -> bool` (kiralık hat taşmasını ertesi günün boş kiralığına devret),
  `"policy_tours" -> bool` (ablasyon anahtarı).
- **K1** kural-tabanlı tohum (çok-günlü havuzlama: ceza < tasarruf kaldıkça küçük OD-gün grupları
  ileri taşınıp birleştirilir), **K2** `improve()` = ALNS-lite (SA kabul, ceza-yoğun `hot`
  gruplara %40 hedefli hamle, her adayda tam simülasyon), **K3** dakika-düzeyi zamanlar zaten
  simülasyondan çıkar.

Yeni bir politika/hamle eklerken: kararı `dec` anahtarı olarak modelle, `Simulator.run` içinde
uygula, `improve()`'a mutasyonunu ekle — böylece ALNS onu otomatik arar ve `--ablation` ile
katkısı ölçülebilir kalır.

## Çalışma notları

- Windows + PowerShell ortamı (VS Code'da conda env). Scriptler
  `sys.stdout.reconfigure(encoding="utf-8")` yapar; Türkçe/emoji çıktı bozulursa konsol kod
  sayfasına bak, koda ASCII kaçışı ekleme.
- Dosya adlarında Türkçe karakter var (`Kiralık_Araclar.xlsx`, `çalıştırma rehberi.txt`);
  git bunları escape'li gösterir, normaldir.
- Excel çıktıları Excel'de açıkken (`~$*.xlsx` kilit dosyası varsa) yazma başarısız olur.
- `optimizer_v5.py` her koşuda `Sonuc_Raporu.md`'yi **üzerine yazar** — elle düzenleme yapma.
- `v3`/`v4` sürümleri artık yok (yalnız `__pycache__` kalıntıları); `v5` dosyaları geçerli koddur.
  README ve rehber metinlerinde hâlâ `forecast_v3.py`/`optimizer_v3.py` geçer, güncel değildir.
- Depodaki ilk-aşama dosyaları (`demand_forecast_v2.py`, `vehicle_optimizer.py` vb.) çalışma
  ağacında silinmiştir; yalnız git geçmişinde durur.
