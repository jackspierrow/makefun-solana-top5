# make.fun — Predict top 5 new Solana coins (pumpfun-top5-forecast-01)

Program forecaster untuk bounty **$200** (1st $50, 2nd $30, 3rd $24, 4th $20,
5th $16, 6th $12, 7–10th $12). Deadline submit **Kamis 12 Nov 2026 21:00 WIB**
(14:00 UTC). Penilaian: log score atas 7 read per jam, 20:00 UTC 12 Nov →
02:00 UTC 13 Nov; harus kalahkan baseline house DAN model AI house.

> **Bukan prediksi teks.** Entri adalah program ini (`main.py`) + repo GitHub
> publik + transkrip sesi. Program TIDAK pernah menyarankan membeli koin apa pun.

## File

| File | Isi |
|---|---|
| `main.py` | Forecaster (Python 3.13, stdlib saja, tanpa network, deterministik) |
| `dev/backtest.py` | Harness backtest lokal (bukan bagian submission) |
| `prediksi.md` | Riset + top 5 draft dari data DexScreener (bahan referensi saja) |

## Cara kerja

`main.py` membaca `$MAKEFUN_INPUT_DIR/task.json` + `history.ndjson` (snapshot
10-menitan house atas semua koin pump.fun sejak opening 02:00 UTC 12 Nov),
lalu menulis `$MAKEFUN_OUTPUT_DIR/forecast.json`:

```json
{"probabilities": [{"mint", "place", "p"}...],
 "top5": [mint x5, tempat 1 dulu],
 "sizes": [{"place", "bin", "p"}...]}
```

- `probabilities`: peluang tiap koin di tiap tempat 1–5 (maks 100 baris/tempat).
  Koin hanya dinamai kalau peluang MC-nya ≥ 0,006; sisanya dibiarkan
  **unnamed** — scorer membagi massa unnamed itu rata ke semua koin di chart.
- `sizes`: peluang bin market cap (log10 SOL) per tempat, 82 bin (tepi dari
  `task.json`, 2,00–6,00 langkah 0,05). Semua 82 bin selalu terisi
  (Laplace smoothing), jadi tidak ada bin yang nol.
- `top5`: 5 koin dengan expected place terendah.

### Model: Monte Carlo jump-diffusion (4.000 simulasi, seed tetap)

Per koin yang dikenal (top 80 by market cap terakhir + pemegang top 5
12 mark terakhir), log10 market cap (SOL) disimulasikan maju ke tiap read:

1. **Drift (momentum yang cepat mati).** Slope OLS 1 jam terakhir, di-shrink
   ke rata-rata cross-sectional, dibatasi ±0,25 log10/jam, meluruh ke lantai
   −0,12 log10/jam dengan konstanta waktu 1 jam. Alasan: di rehearsal, top 5
   berganti hampir total tiap 30 menit (hanya 1 dari 5 bertahan) — momentum
   tidak bertahan berjam-jam.
2. **Aturan puncak (peak rule).** Koin yang sedang di puncak tertingginya dan
   berumur > 15 menit hampir pasti sudah mencapai puncak — slope naiknya
   diganti lantai fade (−0,12/jam). Tanpa ini, koin seperti AIB diproyeksikan
   tumbuh sampai 11.000 SOL, padahal kenyataannya ia keluar dari top 5.
3. **Volatilitas** dari realised return 10-menitan per koin (di-shrink ke
   sd global, lantai 0,08, atap 0,30 per 10 menit).
4. **Crash hazard** (rug/dump): 0,10/jam untuk koin curve, 0,05/jam untuk
   pool, rata-rata jatuh 0,6 log10.
5. **Graduation jump** sekali untuk koin curve yang cadangannya ≥ 40 SOL
   (mendekati ambang selesai ~85 SOL): +0,10–0,35 log10.
6. Koin yang jatuh di bawah log10(25 SOL) keluar dari chart.

### Koin pendatang (phantoms) — bagian terpenting

Sebagian besar top 5 di read jawaban adalah koin yang **lahir setelah
history berakhir** (di practice, 2 dari 5 top-5 read terakhir lahir setelah
history; dari 9 koin yang sempat masuk top 5, 3 lahir setelahnya). Koin
seperti itu tidak bisa dinamai, jadi disimulasikan sebagai "phantom":

- **Laju lahir:** diukur dari history — aliran koin muda (umur < 2 jam)
  yang masuk top 5 per jam (rehearsal: 15/jam; backtest K=2–4: 18–24/jam).
- **Puncak:** di-resample dari distribusi empiris puncak koin-koin yang
  mencapai top 5 di history (median ~2,92 log10 SOL, maks 3,56 di rehearsal)
  — skalanya mengikuti pasar itu sendiri.
- **Bentuk hidup:** mencapai puncak ~20 menit setelah lahir, lalu fade
  0,10 log10 per 10 menit (top 5 berganti hampir total dalam 30 menit).
- Phantom ikut bersaing memperebutkan peringkat di tiap simulasi; massa
  peluang yang mereka ambil adalah massa yang dibiarkan unnamed.

Satu forecast dipakai untuk semua 7 read, jadi peluang yang ditulis adalah
**frekuensi MC yang dirata-rata atas semua read** (mixture atas read adalah
forecast tunggal yang optimal untuk log score), begitu juga histogram size.

### Angka yang diukur dari rehearsal (bukan asumsi)

| Statistik | Nilai |
|---|---|
| Return log10 10-menitan | mean −0,0137, sd 0,1208 |
| Top 5 bertahan 30 menit kemudian | 1 dari 5 |
| Koin baru masuk top 100 per 10 menit | 41–59 (menurun) |
| Koin muda masuk top 5 per jam | ~15–24 |
| Puncak top 5 | median 2,92, maks 3,56 log10 SOL |
| Ambang selesai curve | ~85 SOL cadangan |

## Hasil uji (lokal, data rehearsal 11 Nov)

Runtime ~4–9 detik di data rehearsal (5 mark). Estimasi data final
(66 mark): ~15–30 detik — jauh di bawah batas 300 detik.

### Backtest truncation (skor log per read, makin tinggi makin baik)

Potong history di mark ke-K, skor lawan mark berikutnya, bandingkan dengan
**starter baseline house** (dijalankan dengan Node di input yang sama).
Skor memakai port Python dari scorer house (`grader/forecast/ranked.ts`).

| Uji | Kami | Baseline | Selisih |
|---|---|---|---|
| K=2 (hist s/d 20:20, 3 read) | −54,80 | −49,96 | **−4,84** |
| K=3 (hist s/d 20:30, 2 read) | −29,32 | −33,48 | **+4,16** |
| K=4 (hist s/d 20:40, 1 read) | −24,17 | −29,87 | **+5,70** |

- K=3 dan K=4: menang jelas, didorong skor **sizes** (−11,6 vs −14,9 dan
  −12,3 vs −14,6) — distribusi size lebar kami lebih tepat daripada normal
  sempit baseline (sd 0,35).
- K=2 kalah, **seluruhnya di sizes** (−30,1 vs −23,7; order kami malah
  lebih baik, −24,7 vs −26,2). Penyebab: dengan history cuma 2 mark, pool
  puncak phantom hanya berisi puncak-puncak kecil awal, sementara pasar
  nyata meledak ke puncak baru (Google/DeepSeek/AIB/CHAT lahir setelah cut).
  Ini artefak horizon pendek + history mini — bukan rezim final.
- **Caveat jujur:** semua backtest horizonnya 10–50 menit, sedangkan final
  7–13 jam. Backtest membuktikan mesinnya benar (format, determinisme,
  logika order), bukan kalibrasi horizon panjang. Kalibrasi panjang
  bersandar pada statistik churn/fade/entrant yang diukur di atas.

### Validasi format dengan parser + scorer ASLI house (Node)

```
parseRanked OK: places 5 rows 49 top5 5
parseSizes OK: bins 82 rows 410
VALIDATION PASSED (real house parser + scorer)
```

Format dicek dengan `parseRanked`/`parseSizes` dari
`/tmp/harness/grader/forecast/ranked.ts` (kode scorer house sendiri):
tepat 3 key, ≤ 100 baris/tempat, jumlah per tempat ≤ 1, `top5` 5 mint
berbeda, bin 0–81, jumlah size per tempat ≤ 1, file 18.020 byte
(< 1.048.576).

### Determinisme

Dua run pada input yang sama menghasilkan file **byte-identik**
(`cmp` lolos). Seed tetap 20261112, tanpa jam dinding, tanpa network.

### Pangsa phantom (newcomer) pada run rehearsal penuh

`[0,128, 0,231, 0,282, 0,330, 0,383]` per tempat 1–5 — sejalan dengan
pengamatan practice (2 dari 5 top-5 read terakhir adalah pendatang baru,
~0,33–0,40). Sebelum perbaikan model (drift + peak rule + pool puncak
top-5), pangsa tempat 1 hanya 0,005 — model lama hampir tidak memberi
ruang untuk pendatang baru.

### Top 5 pada run rehearsal penuh (ilustrasi, BUKAN prediksi final)

AIB 0,418 di tempat 1; top5 = AIB, HErn, Google, MIHAO VI, CHAT.
Angka ini berubah total saat dijalankan dengan history final (66 mark,
cut 13:00 UTC) — jangan pakai daftar ini untuk submit.

## Checklist pengiriman (belum dilakukan — keputusan di tangan pradana)

- [ ] Buat **repo GitHub publik** baru (akun jackspierrow) berisi `main.py`
      + lisensi OSI, dibuat setelah kompetisi dibuka.
- [ ] Siapkan **transkrip sesi** coding-agent (sesi ini) untuk dilampirkan.
- [ ] Buat **akun make.fun** (kalau belum ada) + hubungkan wallet Solana
      untuk klaim USDC.
- [ ] Jalankan `main.py` dengan **history final** (snapshot sampai 13:00 UTC /
      20:00 WIB) — JANGAN pakai output rehearsal di atas.
- [ ] Cek ulang output final dengan `dev/backtest.py` (validasi format)
      dan `node /tmp/validate.mjs` (parser house).
- [ ] Submit sebelum **21:00 WIB hari ini (12 Nov 2026)**. Maks 10
      submission; yang terakhir sebelum deadline yang dinilai.

## Yang TIDAK dilakukan

- Tidak submit, tidak buat akun, tidak sign in, tidak push ke GitHub.
- Tidak ada network di `main.py` (stdlib saja).
- Tidak ada saran membeli koin apa pun.
