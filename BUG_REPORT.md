# Bug Report — Portfolio Optimization Tool

**Date:** 2026-11-22 · **Original pass:** report-only audit · **Fix pass:** 2026-11-22 — ALL findings fixed
**Test evidence:** `tests/` pytest suite (64 tests, all passing) + live GUI session against `streamlit_app.py` (Streamlit 1.62, Python 3.13.7, pandas 3.0.5, numpy 2.5.2).

Severity = impact on a real user of the app, assuming normal use.

---

## Summary

| Severity | Count | One-line gist |
|---|---|---|
| 🔴 Critical | 2 | Silent data corruption paths that poison every number the app shows |
| 🟠 High | 5 | Full-page crashes or destroyed local data |
| 🟡 Medium | 6 | Wrong/awkward behavior users will hit regularly |
| ⚪ Low | 7 | Fragile code, cosmetic and hardening issues |

The theme of the Critical/High findings is **silence**: the data pipeline almost never validates what it receives (downloads, tickers, prices), so bad input turns into confidently-displayed garbage instead of an error.

## ✅ Fix status (2026-11-22) — all 20 findings fixed

| ID | Finding | Fix | Fix-guard test |
|---|---|---|---|
| C1 | Bogus/delisted ticker wipes all data | Dead tickers dropped with a warning; history truncation announced (`calculate_log_returns`) + app-level warnings | `test_pipeline_edge_cases.py` |
| C2 | `result.success` never checked | Failed SLSQP solves raise `RuntimeError` (`true_optimization.py`) | `test_optimizer_robustness.py` |
| H1 | 1-ticker page crash | `np.atleast_1d(axes)` in `plot_log_returns` — GUI-verified renders | `test_resource_leaks.py` |
| H2 | Failed download destroys good CSV | Empty download raises `ValueError`, file untouched | `test_pipeline_edge_cases.py` |
| H3 | Empty downloads flow silently | App guards: empty tickers / bad dates / no data → clear `st.error` + stop | GUI-verified |
| H4 | Stale-CSV race | App works from the cached download, never re-reads disk mid-session | `test_pipeline_edge_cases.py` |
| H5 | Non-positive prices → ±inf | Masked to missing data + warning | `test_data_utils.py` |
| M1 | Order-sensitive ticker compare | Set-based checks + canonical (sorted) ticker order + `align_weight_vector` | `test_simulation_mc.py` |
| M2 | Hardcoded plot axes | Data-driven ranges incl. custom portfolio — GUI-verified visible | GUI screenshot |
| M3 | Bare filename crash | `os.makedirs` skipped when dirname empty | `test_pipeline_edge_cases.py` |
| M4 | Figures never closed | `plt.close(fig)` after every matplotlib render — 0 leaks over 20 reruns | `test_resource_leaks.py` |
| M5 | CORS/XSRF disabled | Both re-enabled in `.streamlit/config.toml` | `test_environment.py` |
| M6 | README `cd` mismatch | Fixed + "Running the Tests" section added | `test_environment.py` |
| L1 | Weight section goes silent | Warning added for simulated-but-not-custom state | GUI-verified |
| L2 | Zero-std Sharpe = ±inf | NaN + warning instead | `test_data_utils.py` |
| L3 | Positional weight slice | Name-based resolution + optional `weight_columns` param (app passes tickers) | `test_simulation_mc.py` |
| L4 | Broken covariance silent NaN | `validate_cov_matrix` raises clear `ValueError` up front | `test_optimizer_robustness.py` |
| L5 | Pipeline reruns every drag | Cached `get_pipeline_outputs` in the app | `test_performance_caching.py` |
| L6 | Confusing `idxmax` ValueError | Actionable `ValueError` messages in `get_mc_optimal_portfolio` | `test_simulation_mc.py` |
| L7 | 50k portfolios ~40s | Vectorized Monte Carlo: **39.8s → 0.02s** (identical seeded results) | `test_performance_caching.py` |

Performance after fixes: 64/64 tests green · 50k portfolios 0.02s / 6.8 MB · optimizer 2.7s → 6.8ms cached · per-rerun pipeline now cache-hit instant.

---

## 🔴 Critical

### C1. One bogus or delisted ticker silently wipes ALL data for every ticker
- **Where:** `core/data_utils.py:66` — `np.log(close_prices / close_prices.shift(1)).dropna()`
- **What happens:** `dropna()` is applied to the *whole* DataFrame. One ticker that is NaN from day 0 (bogus symbol, never-listed) makes **every row** NaN → log returns become an **empty** frame → covariance/correlation become **all-NaN** → Monte Carlo produces **NaN Sharpe ratios** → the app finally crashes far downstream with `ValueError` in `get_mc_optimal_portfolio` (`core/mc_optimization.py:18`), or worse, displays NaN-based charts if the failure lands differently. No warning is ever shown at the point of the real problem.
- **Variant:** a ticker that *delists halfway* through the range silently **truncates the whole panel** — 59 of 99 usable rows kept in the probe; 40 days of good data for the healthy tickers thrown away with no message.
- **Tests:** `tests/test_pipeline_edge_cases.py::test_bogus_ticker_from_start_wipes_all_rows`, `::test_bogus_ticker_full_chain_ends_in_valueerror`, `::test_half_delisted_ticker_silently_truncates_history`

### C2. Optimizer failures are never checked — garbage weights presented as "optimal"
- **Where:** `core/true_optimization.py:66` (`find_true_optimal_portfolio`) and `:92` (`get_min_variance_portfolio`) — neither checks `result.success`. The efficient-frontier loop *does* check (`:135`), which proves the authors knew about the flag.
- **What happens:** a forced-failure experiment shows `compute_all_true_portfolios` completes without error and hands back the solver's raw failed guess as the "True Optimal" and "GMV" portfolios. In the UI these are drawn on the frontier chart and named in the comparison tables. Any numerical trouble (bad covariance, near-singular inputs, unlucky start) can therefore produce a confidently-labeled wrong answer.
- **Tests:** `tests/test_optimizer_robustness.py::test_failed_optimizer_result_flows_through_silently`, `::test_real_slsqp_baseline_succeeds_on_good_data` (shows success *is* reported on healthy data — the flag exists and is simply ignored)

---

## 🟠 High

### H1. Selecting exactly one ticker crashes the entire page
- **Where:** `core/plotting.py:125` + `:128` — `plt.subplots(1, 1)` returns a bare `Axes` object (not a list), so `axes[i]` raises `TypeError: 'Axes' object is not subscriptable`.
- **GUI-confirmed:** deselecting 4 of the 5 tickers turns the whole page into an error box; the app's own traceback points at `streamlit_app.py:152` → `core/plotting.py:128`. Recovery requires re-adding tickers.
- **Tests:** `tests/test_resource_leaks.py::test_single_ticker_plot_crashes_with_typeerror` (+ live GUI screenshot)

### H2. One failed download destroys the previously-good local CSV
- **Where:** `core/data_utils.py:30` — `data.to_csv(filename)` runs unconditionally, even when yfinance returned an empty frame (which is how it reports a blocked/failed request — it does **not** raise).
- **What happens:** in the probe, a 24,299-byte good CSV was replaced by a **4-byte** file. The next rerun then crashes at `streamlit_app.py:139` (`ParserError` loading the gutted file) — the user's only recourse is a successful re-download.
- **Tests:** `tests/test_pipeline_edge_cases.py::test_failed_download_overwrites_good_csv_with_empty_file`, `::test_empty_ticker_list_produces_empty_csv_then_crash` (empty ticker selection → unloadable CSV → crash)

### H3. Empty/zero-row downloads flow through the whole pipeline silently
- **Where:** the chain `streamlit_app.py:138-143` performs **no validation** of the download before computing on it.
- **What happens:** a zero-row download loads as an empty frame, log returns are empty, covariance is all-NaN — **no error at any step** until the Monte-Carlo optimum. Same family as C1 but caused by dates instead of tickers (e.g. `start_date >= end_date`).
- **Tests:** `tests/test_pipeline_edge_cases.py::test_zero_row_download_propagates_silently`

### H4. Stale-CSV race: the disk file can disagree with the cache key
- **Where:** `streamlit_app.py:138-139` — `get_stock_data(...)` is cached by `(tickers, start, end)`, but the pipeline **always reads `data/stock_data.csv` from disk**, with no check that the file matches what was requested.
- **What happens:** if another process/session overwrites the shared file (or a failed download gutted it, see H2), a cache hit serves the *wrong tickers' data* as if it were correct. In the test, a request for one ticker set silently returns a different set from disk.
- **Tests:** `tests/test_pipeline_edge_cases.py::test_cached_fetch_can_silently_read_another_sessions_csv`

### H5. Zero/negative prices poison the covariance matrix via ±inf
- **Where:** `core/data_utils.py:66` — `log(0)` yields `-inf`, and `dropna()` removes only `NaN`, never `inf`.
- **What happens:** one zero close in 50 days → `-inf` rows survive → `cov()` returns an **all-NaN matrix** with no error anywhere. (Splits, data errors, or bad CSV edits can produce zeros.)
- **Tests:** `tests/test_data_utils.py::test_log_returns_zero_price_inf_survives_dropna`

---

## 🟡 Medium

### M1. Ticker comparison is order-sensitive → false "changed" resets (known gap #15)
- **Where:** `streamlit_app.py:89` (state invalidation) and the three warning sites `:253`, `:298`, `:332` — all compare ticker **lists** with `!=`.
- **GUI-confirmed:** after removing a ticker and re-adding it (final selection = the *same set* that was simulated, order changed), the app wiped the simulation state and demanded a re-run. A set-comparison (`set(a) == set(b)`) or sorted comparison would recognize nothing changed.
- **Tests:** GUI session + code inspection (a pure unit repro lives in the invalidation logic; UI flow documented here)

### M2. Hardcoded plot axis ranges clip real portfolios
- **Where:** `core/plotting.py:101-102` — xaxis `range=[0.16, 0.30]`, yaxis `range=[0.08, 0.30]`, with only a text note "Axes are zoomed in…".
- **GUI-confirmed:** a custom portfolio with 7.55% return renders **outside the visible plot area** (its star is clipped at the edge) because 7.55% < the hardcoded 8% y-minimum. Any low-return/high-vol combination is invisible exactly where the user is told to look for it.
- **Tests:** live GUI screenshot (star outside axes grid) + code inspection

### M3. Bare filename crashes the data fetcher
- **Where:** `core/data_utils.py:21` — `os.makedirs(os.path.dirname(filename))`; for `filename="stock_data.csv"` the dirname is `''` → `FileNotFoundError` before any download.
- **Tests:** `tests/test_pipeline_edge_cases.py::test_bare_filename_crashes_on_empty_dirname`

### M4. Matplotlib figures are never closed → memory leak
- **Where:** `core/plotting.py:143` and `:171` — `st.pyplot(fig)` with no `plt.close(fig)` anywhere.
- **Measured:** 20 simulated reruns left **40 figures alive**; matplotlib itself started warning "More than 20 figures have been opened". Every slider drag on a live session adds 2 more leaked figures (log-returns + heatmap render on the page).
- **Tests:** `tests/test_resource_leaks.py::test_matplotlib_figures_accumulate_over_20_reruns`

### M5. Security hardening disabled in Streamlit config
- **Where:** `.streamlit/config.toml` — `enableCORS = false`, `enableXsrfProtection = false`.
- **Impact:** fine for localhost personal use; if this app is ever exposed beyond localhost (the Deploy button makes that tempting), both flags protect against cross-site requests to a running instance. Recorded as-is; decision belongs to the user.
- **Tests:** `tests/test_environment.py::test_streamlit_config_toml_is_valid`

### M6. README instructions don't survive verbatim use
- **Where:** `README.md:26-27` — `git clone …/port-opt-tool.git` then `cd port_opt_tool` (underscore vs the hyphenated repo name; the cloned folder will be `port-opt-tool`).
- **Tests:** `tests/test_environment.py::test_readme_run_instructions_reference_real_files` (run command and file references verified; the cd mismatch documented here)

---

## ⚪ Low

### L1. Weight-comparison section goes silent in one state
- `streamlit_app.py:290-299` — when a simulation exists but no custom portfolio was generated, this section renders **nothing** (the frontier and metrics sections both show a "generate the custom portfolio first" warning). GUI-confirmed. Cosmetic/UX.

### L2. Zero-variance asset yields ±inf Sharpe with no guard
- `core/data_utils.py:151` — flat prices (std = 0) → `inf` Sharpe ratio shown as a normal number. `tests/test_data_utils.py::test_individual_sharpe_zero_std_gives_inf`

### L3. MC-optimal weights taken by *position*, not by name
- `core/mc_optimization.py:27` — `opt_port[portfolios_df.columns[3:]]`. Any extra column inserted after position 3 leaks into the "weights" vector (demonstrated with an intruder column). `tests/test_simulation_mc.py::test_mc_optimal_weight_slice_is_positional_not_named`

### L4. Broken covariance inputs produce NaN/inf silently
- `core/true_optimization.py:19` — non-PSD covariance → negative variance → `sqrt` → NaN vol, no raise (`tests/test_optimizer_robustness.py::test_non_psd_covariance_produces_nan_vol_silently`); zero variance → division by zero → inf Sharpe (`::test_single_asset_zero_variance_divides_by_zero`). Combined with C2, these flow straight into "optimal" results.

### L5. The whole data pipeline re-runs on every widget interaction
- `streamlit_app.py:138-143` — CSV read, log returns, descriptive stats, cov/corr recompute on *every* rerun (each writes CSVs to disk too). Measured cost: **65 ms per slider drag** — tolerable today, but it grows with history length and wastes disk writes.
- `tests/test_performance_caching.py::test_every_rerun_pipeline_cost`

### L6. All-NaN Sharpe or empty simulation crashes `idxmax`
- `core/mc_optimization.py:18` — `idxmax` on an all-NaN or empty column raises `ValueError`. This is the *loud* end of the silent chains in C1/H3 — the only error the user ever sees is this misleading one. `tests/test_simulation_mc.py::test_mc_optimal_all_nan_sharpe_raises`, `::test_mc_optimal_empty_dataframe_raises`

### L7. Max-settings simulation takes ~40s with no progress detail
- 50,000 portfolios: **39.8 s / 31.2 MB peak** (informational — the app's own spinner covers it, and 10k default is ~2.4 s). `tests/test_performance_caching.py::test_max_portfolios_50000_timing_and_memory`

---

## ✅ Verified-good findings (regression protection now in place)

These previously-fixed bugs are confirmed fixed and now guarded by permanent tests:
- **Errno 22 savefig crash** — best-effort disk saves survive simulated file locks (`tests/test_regressions.py`).
- **`mc_result` AttributeError** — guarded at `streamlit_app.py:193`.
- **NameError on bad custom weights** — metrics computed only inside the success branch (`streamlit_app.py:224-233`).
- **Duplicate log-return computation** — exactly one call site (`streamlit_app.py:140`).
- **`simulation.py` signature** — no stale `output_csv` parameter.

Also verified working in the live GUI:
- Full happy path end-to-end (load → simulate → sliders → custom portfolio → 3 chart sections render with correct data — the Custom bar was 100% HDFCBANK for a 100% weight).
- The weight-sum validation (all-zero and 0.99 sums both rejected with the 🚫 error).
- The **new** param-change invalidation: changing the risk-free rate wipes stale simulation state (works as designed).
- Reset to Default (restores everything), 🔄 Refresh Data (clears cache and re-downloads), cache behavior (2nd identical fetch = no download; optimizer 2,700 ms → 7.6 ms cached).
- All math verified against hand-computed values; annualization identical across all 4 sites; MC never beats the true optimizer; GMV is the global volatility minimum; frontier upper branch is monotonic.

## 📊 Performance measurements

| What | Result |
|---|---|
| 50,000-portfolio simulation | 39.8 s, 31.2 MB peak |
| Default 10k (10-ticker synthetic) | MC 2.4 s / optimizer 4.7 s |
| Per-rerun pipeline cost | 65 ms |
| Optimizer uncached → cached | 2,700 ms → 7.6 ms |
| Full test suite | 63 tests, ~2 min |

## Fix-pass notes

Every finding above was fixed on 2026-11-22. Each bug-documenting test in `tests/` was upgraded into a permanent guard for the fixed behavior, so none of these 20 regressions can silently return. The findings' original write-ups are kept above as documentation of *why* each guard exists.
