# Download Data

This guide covers the supported ways to download market data in this repo.

Use `dukascopy` when you need long historical FX data for research and backtests.
Use `mt5_ftmo` when you need bars from a live MetaTrader 5 terminal or broker-specific feed.

## 1. Pick A Provider

### Dukascopy

Best for:

- long FX history
- reproducible backtests
- bulk research datasets

### MT5 / FTMO

Best for:

- broker-specific symbols
- data that should match the MT5 terminal feed
- quick pulls from a connected MT5 instance

Important:

- MT5 can only return history that the connected terminal and broker actually expose.
- Range mode uses UTC and treats `--end` as exclusive: `[start, end)`.
- If MT5 history starts late, the downloader now reports the actual history floor directly.

## 2. Common Output

All download paths write:

- a dataset file under `data/processed/` by default
- a sidecar metadata file with the same name plus `.meta.json`

Examples:

- `data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv`
- `data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv.meta.json`

You can choose `csv` or `parquet` with `--file-ext`.

## 3. Streamlit UI

Start the app:

```bash
.venv/bin/streamlit run streamlit_app.py
```

Then open the `Download data` page.

### Dukascopy in UI

1. Set `Provider` to `Dukascopy FX`.
2. Pick `Symbol`, `Timeframe`, `Start date`, and `End date`.
3. Set `File extension` and `Save directory`.
4. Click `Download data`.

### MT5 in UI

1. Set `Provider` to `MT5 / FTMO`.
2. Set `MT5 backend`.
   On macOS with the Docker bridge, use `silicon`.
3. Set `MT5 host` and `MT5 port`.
   Default local bridge is usually `localhost:8001`.
4. Enter `Symbol` and `Timeframe`.
5. Set `Start date` and `End date`.
6. Optionally set `Batch size`, `Max backfill batches`, and `Progress every N batches`.
7. Enable `Allow incomplete` only if you are okay saving partial MT5 history.
8. Click `Download data`.

## 4. CLI Quick Start

The unified downloader is:

```bash
python3 scripts/download_data.py
```

Installed editable entrypoint:

```bash
quantbt-download-data
```

### List Dukascopy Timeframes

```bash
python3 scripts/download_data.py --provider dukascopy --list-timeframes
```

### Download Dukascopy EURUSD 1H

```bash
python3 scripts/download_data.py \
  --provider dukascopy \
  --symbol EURUSD \
  --timeframe 1H \
  --start 2010-01-01 \
  --end 2013-01-01 \
  --save-dir data/processed \
  --file-ext csv
```

### Download MT5 EURUSD M5

```bash
python3 scripts/download_data.py \
  --provider mt5_ftmo \
  --symbol EURUSD \
  --timeframe M5 \
  --start 2025-01-01 \
  --end 2025-02-01 \
  --save-dir data/processed \
  --file-ext csv \
  --mt5-backend silicon \
  --mt5-host localhost \
  --mt5-port 8001
```

### Save Partial MT5 History Anyway

```bash
python3 scripts/download_data.py \
  --provider mt5_ftmo \
  --symbol EURUSD \
  --timeframe M15 \
  --start 2025-01-01 \
  --end 2026-01-01 \
  --save-dir data/processed \
  --file-ext csv \
  --mt5-backend silicon \
  --mt5-host localhost \
  --mt5-port 8001 \
  --mt5-allow-incomplete
```

## 5. MT5 On macOS With siliconmetatrader5

Bootstrap the local MT5 bridge:

```bash
bash scripts/setup_silicon_mt5_local.sh --with-python-deps
```

Check that the MT5 container is running:

```bash
docker ps
```

You should see a `siliconmt5` container exposing port `8001`.

If you need to inspect the compose service directly:

```bash
cd .third_party/silicon-metatrader5/docker
docker-compose ps
```

Rebuild the MT5 bridge container after bootstrap changes:

```bash
cd .third_party/silicon-metatrader5/docker
docker-compose up -d --build --force-recreate
```

Verify the startup chart-bar setting inside the running container:

```bash
docker exec siliconmt5 bash -lc 'env | grep MT5_MAX_BARS'
```

```bash
docker exec siliconmt5 bash -lc 'iconv -f UTF-16LE -t UTF-8 "/opt/wineprefix/drive_c/Program Files/MetaTrader 5/mt5cfg.ini" | grep MaxBars'
```

Expected after the current patch:

- `MT5_MAX_BARS=100000000`
- `MaxBars=100000000`

## 6. Troubleshooting MT5 History

### Symptom: history starts much later than requested

Example:

- requested start: `2010-01-01`
- MT5 reports history starts at `2025-12-31T03:45:00+00:00`

What this means:

- the MT5 terminal or broker feed does not currently expose older bars to the bridge
- the downloader cannot manufacture missing history

What to check:

1. Confirm the correct symbol and timeframe in MT5.
2. Open the MT5 terminal over VNC and load the exact chart you want, such as `EURUSD` `M15`.
3. Scroll far left or use `Home` repeatedly to force older history to download.
4. Make sure the MT5 Docker volume is preserved across restarts.
5. Re-run the download after the terminal history finishes loading.

Important:

- Increasing `MaxBars` helps prevent an artificially low chart cap on fresh MT5 containers.
- It does not guarantee that the broker will provide 10+ years of history.
- If you need deep research history and MT5 does not have it, use `dukascopy`.

### Symptom: `docker compose` does not work

Some environments use the standalone Compose binary instead of the Docker plugin.

Use:

```bash
docker-compose up -d --build --force-recreate
```

Check which Compose flavor you have:

```bash
docker compose version
docker-compose version
```

## 7. Useful MT5 Commands

List MT5-exposed timeframes:

```bash
python3 scripts/download_data.py \
  --provider mt5_ftmo \
  --mt5-backend silicon \
  --mt5-host localhost \
  --mt5-port 8001 \
  --list-timeframes
```

List MT5 symbols:

```bash
python3 scripts/download_data.py \
  --provider mt5_ftmo \
  --mt5-backend silicon \
  --mt5-host localhost \
  --mt5-port 8001 \
  --list-symbols
```

## 8. Practical Recommendation

Use `dukascopy` for long backtest datasets.

Use `mt5_ftmo` when you need the feed to match a live MT5 environment, and expect to manage terminal history depth explicitly.
