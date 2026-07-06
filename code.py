# Install packages not preinstalled on Colab
!pip install -q yfinance arch


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf

pd.set_option("display.width", 120)


TICKER = "^GSPC"          # S&P 500 index. Try "^NSEI" for Nifty 50 if you prefer.
START  = "2012-01-01"
END    = "2026-07-01"

TEST_FRACTION       = 0.20   # chronological hold-out, most recent 20% of days
GARCH_REFIT_EVERY   = 21     # trading days (~monthly) — expanding window
XGB_REFIT_EVERY     = 63     # trading days (~quarterly) — expanding window
LSTM_SEQ_LEN        = 20     # days of history the LSTM looks back on
LSTM_EPOCHS         = 30
LSTM_N_SEEDS        = 5      # ensemble size -- see section 6 for why this matters


import yfinance as yf

raw = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
raw = raw[["Close"]].copy()
raw.columns = ["Close"]
raw["log_ret"] = np.log(raw["Close"] / raw["Close"].shift(1))
raw = raw.dropna()

print(f"{len(raw)} trading days, {raw.index[0].date()} to {raw.index[-1].date()}")
raw.head()


fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(raw.index, raw["Close"])
ax.set_title(f"{TICKER} closing price")
plt.show()


def build_dataset(df, ewma_lambda=0.94):
    out = df.copy()
    r = out["log_ret"]
    r_lag1 = r.shift(1)  # r[t-1]: the most recent return known at prediction time

    out["target_var"] = r ** 2
    out["target_vol"] = np.sqrt(out["target_var"] * 252)

    for lag in range(1, 6):
        out[f"ret_lag{lag}"] = r.shift(lag)
        out[f"sqret_lag{lag}"] = r.shift(lag) ** 2

    for w in (5, 10, 21):
        out[f"roll_std_{w}"] = r_lag1.rolling(w).std()
        out[f"roll_absmean_{w}"] = r_lag1.abs().rolling(w).mean()

    # RiskMetrics-style EWMA volatility, built from r[t-1] backwards
    rs = r_lag1.fillna(0.0).values
    ewma_var = np.zeros(len(r))
    ewma_var[0] = rs[0] ** 2
    for t in range(1, len(r)):
        ewma_var[t] = ewma_lambda * ewma_var[t - 1] + (1 - ewma_lambda) * rs[t] ** 2
    out["ewma_vol"] = np.sqrt(ewma_var * 252)

    return out.dropna()


FEATURE_COLS = (
    [f"ret_lag{i}" for i in range(1, 6)]
    + [f"sqret_lag{i}" for i in range(1, 6)]
    + [f"roll_std_{w}" for w in (5, 10, 21)]
    + [f"roll_absmean_{w}" for w in (5, 10, 21)]
    + ["ewma_vol"]
)

ds = build_dataset(raw)
print(ds.shape)
ds[["target_vol", "ewma_vol"] + FEATURE_COLS[:3]].head()


def chrono_split(df, test_frac=0.2):
    n_test = int(len(df) * test_frac)
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()

train, test = chrono_split(ds, TEST_FRACTION)
print(f"train={len(train)} rows ({train.index[0].date()} to {train.index[-1].date()})")
print(f"test ={len(test)} rows ({test.index[0].date()} to {test.index[-1].date()})")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ds.index, ds["target_vol"], linewidth=0.6)
ax.axvline(test.index[0], color="red", linestyle="--", label="train / test split")
ax.set_title("Realized volatility proxy (annualized)")
ax.legend()
plt.show()


from arch import arch_model

def garch_walk_forward(full_df, train, test, refit_every=21):
    returns_pct = full_df["log_ret"] * 100
    test_idx = test.index
    forecasts = pd.Series(index=test_idx, dtype=float)
    fitted_params = None

    for i, t in enumerate(test_idx):
        loc = full_df.index.get_loc(t)
        history = returns_pct.iloc[:loc]

        if fitted_params is None or i % refit_every == 0:
            am = arch_model(history, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off")
            fitted_params = res.params
            fc = res.forecast(horizon=1, reindex=False)
        else:
            am = arch_model(history, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res_fixed = am.fix(fitted_params)
            fc = res_fixed.forecast(horizon=1, reindex=False)

        var_pct2 = fc.variance.values[-1, 0]
        daily_var = var_pct2 / (100 ** 2)
        forecasts.loc[t] = np.sqrt(daily_var * 252)

    return forecasts

garch_fc = garch_walk_forward(ds, train, test, refit_every=GARCH_REFIT_EVERY)
print("GARCH walk-forward done.")
garch_fc.head()


import xgboost as xgb

def xgb_walk_forward(train, test, refit_every=63):
    test_idx = test.index
    forecasts = pd.Series(index=test_idx, dtype=float)
    combined = pd.concat([train, test])
    train_end_pos = len(train)
    model = None

    for i, t in enumerate(test_idx):
        cur_pos = train_end_pos + i
        if model is None or i % refit_every == 0:
            hist = combined.iloc[:cur_pos]
            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=0,
            )
            model.fit(hist[FEATURE_COLS], np.sqrt(hist["target_var"] * 252))

        row = combined.iloc[[cur_pos]][FEATURE_COLS]
        forecasts.loc[t] = model.predict(row)[0]

    return forecasts

xgb_fc = xgb_walk_forward(train, test, refit_every=XGB_REFIT_EVERY)
print("XGBoost walk-forward done.")
xgb_fc.head()


from sklearn.preprocessing import StandardScaler

def make_sequences(feature_arr, target_arr, seq_len):
    X, y = [], []
    for i in range(seq_len, len(feature_arr)):
        X.append(feature_arr[i - seq_len:i])
        y.append(target_arr[i])
    return np.array(X), np.array(y)

def lstm_forecast(train, test, seq_len=20, epochs=30, seed=0):
    tf.keras.utils.set_random_seed(seed)
    scaler = StandardScaler().fit(train[FEATURE_COLS])
    combined = pd.concat([train, test])
    feat_scaled = scaler.transform(combined[FEATURE_COLS])
    target_vol = np.sqrt(combined["target_var"].values * 252)

    X_all, y_all = make_sequences(feat_scaled, target_vol, seq_len)
    n_train_seq = len(train) - seq_len
    X_train, y_train = X_all[:n_train_seq], y_all[:n_train_seq]
    X_test = X_all[n_train_seq:]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, X_all.shape[2])),
        tf.keras.layers.LSTM(32, activation="tanh"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="softplus"),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0,
              validation_split=0.1,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

    preds = model.predict(X_test, verbose=0).flatten()
    return pd.Series(preds, index=test.index)

def lstm_forecast_ensemble(train, test, seq_len=20, epochs=30, n_seeds=5):
    per_seed = []
    for seed in range(n_seeds):
        fc = lstm_forecast(train, test, seq_len=seq_len, epochs=epochs, seed=seed)
        per_seed.append(fc.rename(f"seed_{seed}"))
    stacked = pd.concat(per_seed, axis=1)
    return stacked.mean(axis=1), stacked.std(axis=1), stacked

lstm_fc, lstm_std, lstm_stack = lstm_forecast_ensemble(
    train, test, seq_len=LSTM_SEQ_LEN, epochs=LSTM_EPOCHS, n_seeds=LSTM_N_SEEDS)
print(f"LSTM ensemble of {LSTM_N_SEEDS} seeds done. "
      f"Mean seed-to-seed std across test days: {lstm_std.mean():.4f} "
      f"(compare to the typical vol level of ~{test['target_vol'].mean():.2f} to judge materiality)")
lstm_fc.head()


def qlike(y_true_var, y_pred_var):
    ratio = y_true_var / y_pred_var
    return np.mean(ratio - np.log(ratio) - 1)

def evaluate(name, y_true_vol, y_pred_vol):
    y_true_vol = np.asarray(y_true_vol)
    y_pred_vol = np.clip(np.asarray(y_pred_vol), 1e-4, None)
    y_true_var, y_pred_var = (y_true_vol ** 2) / 252, (y_pred_vol ** 2) / 252
    rmse = np.sqrt(np.mean((y_true_vol - y_pred_vol) ** 2))
    mae = np.mean(np.abs(y_true_vol - y_pred_vol))
    ql = qlike(np.clip(y_true_var, 1e-12, None), np.clip(y_pred_var, 1e-12, None))
    ss_res = np.sum((y_true_vol - y_pred_vol) ** 2)
    ss_tot = np.sum((y_true_vol - np.mean(y_true_vol)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"model": name, "RMSE": rmse, "MAE": mae, "QLIKE": ql, "R2": r2}

def diebold_mariano(err1, err2):
    d = np.asarray(err1) ** 2 - np.asarray(err2) ** 2
    n = len(d)
    dm_stat = d.mean() / np.sqrt(d.var(ddof=0) / n)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

true_vol = test["target_vol"]
forecasts = {"GARCH(1,1)": garch_fc, "XGBoost": xgb_fc, "LSTM": lstm_fc}

results_df = pd.DataFrame([evaluate(name, true_vol, fc) for name, fc in forecasts.items()])
print(results_df)

dm_table = []
names = list(forecasts.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        stat, p = diebold_mariano(true_vol - forecasts[a], true_vol - forecasts[b])
        dm_table.append((a, b, stat, p))
        sig = "significant at 5%" if p < 0.05 else "not significant at 5%"
        print(f"{a} vs {b}: DM={stat:.3f}, p={p:.3f} ({sig})")


fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
for ax, (name, fc) in zip(axes, forecasts.items()):
    ax.plot(true_vol.index, true_vol.values, label="Realized (actual)", linewidth=0.7, alpha=0.7)
    ax.plot(fc.index, fc.values, label=f"{name} forecast", linewidth=1.0)
    ax.set_title(name)
    ax.legend()
plt.tight_layout()
plt.show()


def regime_breakdown(true_vol, forecasts, n_bins=3):
    labels = ["Low", "Medium", "High"][:n_bins]
    regime = pd.qcut(true_vol, n_bins, labels=labels)
    rows = []
    for name, fc in forecasts.items():
        for lbl in labels:
            mask = (regime == lbl).values
            if mask.sum() < 5:
                continue
            m = evaluate(name, true_vol.values[mask], fc.values[mask])
            m["regime"] = lbl
            m["n_days"] = int(mask.sum())
            rows.append(m)
    return pd.DataFrame(rows)

def spike_capture(true_vol, forecasts, top_pct=0.05):
    n_top = max(1, int(len(true_vol) * top_pct))
    top_idx = true_vol.sort_values(ascending=False).index[:n_top]
    rows = []
    for name, fc in forecasts.items():
        ratio = (fc.loc[top_idx] / true_vol.loc[top_idx]).mean()
        rows.append({"model": name, "n_spike_days": n_top, "avg_pred_over_actual": ratio})
    return pd.DataFrame(rows)

regime_df = regime_breakdown(true_vol, forecasts, n_bins=3)
spike_df = spike_capture(true_vol, forecasts, top_pct=0.05)

print(regime_df[["model", "regime", "n_days", "RMSE", "QLIKE"]])
print()
print(spike_df)


from matplotlib.backends.backend_pdf import PdfPages

def build_report(pdf_path, meta, raw_df, ds, train, test, forecasts, results_df, dm_table,
                  regime_df=None, spike_df=None, lstm_stability=None):
    true_vol = test["target_vol"]

    def text_page(pdf, title, lines, fontsize=11, line_spacing=0.035, start_y=0.88, bottom_margin=0.05):
        # Auto-paginate: a fixed-height page can only fit so many lines, and
        # silently overflowing past the bottom margin clips content rather
        # than raising an error, so it's worth guarding against explicitly.
        max_lines = max(1, int((start_y - bottom_margin) / line_spacing))
        chunks = [lines[i:i + max_lines] for i in range(0, len(lines), max_lines)] or [[]]
        for idx, chunk in enumerate(chunks):
            page_title = title if idx == 0 else f"{title} (cont'd)"
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.94, page_title, fontsize=16, fontweight="bold")
            y = start_y
            for line in chunk:
                fig.text(0.08, y, line, fontsize=fontsize, wrap=True, va="top")
                y -= line_spacing
            plt.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(pdf_path) as pdf:
        text_page(pdf, "Volatility Forecasting Shootout", [
            "GARCH(1,1)  vs.  XGBoost  vs.  LSTM", "",
            f"Instrument:        {meta['ticker']}",
            f"Sample period:     {meta['start']}  to  {meta['end']}",
            f"Total observations (after feature construction): {len(ds)}",
            f"Train / Test split: {len(train)} / {len(test)}  (chronological, no shuffling)", "",
            "Target definition:",
            "  target_var[t] = r[t]^2          (realized-variance proxy for day t)",
            "  target_vol[t] = sqrt(target_var[t] * 252)   (annualized)", "",
            "Every feature at row t uses ONLY r[t-1], r[t-2], ... -- never r[t] itself --",
            "so all three models forecast the same one-day-ahead quantity from the same",
            "information set. An off-by-one error here is a classic source of inflated,",
            "non-reproducible backtest results.", "",
            f"GARCH refit cadence:   every {meta['refit_every_garch']} trading days (expanding window)",
            f"XGBoost refit cadence: every {meta['refit_every_xgb']} trading days (expanding window)",
            f"LSTM: {meta.get('n_seeds', 1)}-seed ensemble average, sequence length {meta['seq_len']} days", "",
            "Evaluation metrics: RMSE and MAE (vol units), QLIKE (variance units -- the",
            "standard scoring rule for volatility forecasts), and the Diebold-Mariano test",
            "for whether accuracy differences are statistically significant.",
        ])

        fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
        axes[0].plot(raw_df.index, raw_df["Close"])
        axes[0].set_title(f"{meta['ticker']} price level")
        axes[1].plot(ds.index, ds["target_vol"], linewidth=0.6)
        axes[1].set_title("Realized volatility proxy (annualized)")
        axes[1].axvline(test.index[0], color="red", linestyle="--", linewidth=1, label="train/test split")
        axes[1].legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for name, fc in forecasts.items():
            fig, ax = plt.subplots(figsize=(8.27, 5))
            ax.plot(true_vol.index, true_vol.values, label="Realized (actual)", linewidth=0.8, alpha=0.7)
            ax.plot(fc.index, fc.values, label=f"{name} forecast", linewidth=1.0)
            ax.set_title(f"{name}: forecast vs. realized volatility (test period)")
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        for ax, metric in zip(axes, ["RMSE", "MAE", "QLIKE"]):
            ax.bar(results_df["model"], results_df[metric], color=["#4C72B0", "#DD8452", "#55A868"])
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=20)
        fig.suptitle("Model comparison (lower is better on all three)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Calibration: predicted vs actual, points below the diagonal = under-prediction
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        for ax, (name, fc) in zip(axes, forecasts.items()):
            ax.scatter(true_vol.values, fc.values, s=6, alpha=0.35)
            lim = max(true_vol.max(), fc.max())
            ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="perfect calibration")
            ax.set_xlabel("Actual realized vol")
            ax.set_ylabel("Predicted vol")
            ax.set_title(name)
            ax.legend(fontsize=8)
        fig.suptitle("Calibration: points below the red line are under-predictions")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if regime_df is not None and spike_df is not None:
            lines = [
                "A single pooled RMSE/QLIKE number can hide a model that performs well",
                "on ordinary days but quietly fails exactly when volatility spikes --",
                "which is the scenario that actually matters for risk management and",
                "options trading. So the test period is split into Low / Medium / High",
                "realized-volatility terciles and every model is re-scored within each.", "",
                f"{'Model':<13}{'Regime':<9}{'N':>5}{'RMSE':>10}{'QLIKE':>10}",
            ]
            for _, row in regime_df.iterrows():
                lines.append(f"{row['model']:<13}{row['regime']:<9}{row['n_days']:>5}{row['RMSE']:>10.4f}{row['QLIKE']:>10.4f}")
            high = regime_df[regime_df["regime"] == "High"].sort_values("QLIKE")
            lines.append("")
            if len(high):
                lines.append(f"Best QLIKE in the High-volatility regime: {high.iloc[0]['model']}")
            lines += ["", f"Spike capture -- average (predicted / actual) on the worst "
                      f"{spike_df.iloc[0]['n_spike_days']} test days by realized vol:",
                      f"{'Model':<13}{'pred / actual':>15}"]
            for _, row in spike_df.iterrows():
                flag = "  <-- under-predicts spikes" if row["avg_pred_over_actual"] < 0.85 else ""
                lines.append(f"{row['model']:<13}{row['avg_pred_over_actual']:>15.3f}{flag}")
            lines += ["", "A ratio well below 1.0 means the model systematically under-calls",
                      "volatility on exactly the days where under-calling it is costliest --",
                      "e.g. an under-hedged options book right before a vol spike."]
            text_page(pdf, "Regime Breakdown & Tail-Spike Capture", lines, fontsize=10)

        lines = ["Metric table (pooled, whole test period):", "",
                 f"{'Model':<14}{'RMSE':>10}{'MAE':>10}{'QLIKE':>10}{'R2':>10}"]
        for _, row in results_df.iterrows():
            lines.append(f"{row['model']:<14}{row['RMSE']:>10.4f}{row['MAE']:>10.4f}{row['QLIKE']:>10.4f}{row['R2']:>10.4f}")
        lines += ["", "Diebold-Mariano tests (squared-error loss differential):"]
        for a, b, stat, p in dm_table:
            sig = "significant at 5%" if p < 0.05 else "not significant at 5%"
            lines.append(f"  {a} vs {b}: DM={stat:.3f}, p={p:.3f}  ({sig})")

        best_rmse = results_df.sort_values("RMSE").iloc[0]["model"]
        best_qlike = results_df.sort_values("QLIKE").iloc[0]["model"]
        lines += ["", f"Lowest pooled RMSE:   {best_rmse}", f"Lowest pooled QLIKE:  {best_qlike}"]
        if lstm_stability is not None:
            lines += ["", f"LSTM ensemble stability: mean seed-to-seed std = {lstm_stability:.4f}",
                      "  (GARCH and XGBoost are deterministic given the data; this is the",
                      "   price of the LSTM's flexibility -- controlled here by averaging seeds)"]
        lines += ["", "Recommendation:"]
        if best_rmse != best_qlike:
            lines += [f"  RMSE and QLIKE disagree on the winner ({best_rmse} vs {best_qlike}).",
                      "  This is a real, informative disagreement, not noise: RMSE is dominated",
                      "  by the many ordinary days, while QLIKE weights large-variance days far",
                      "  more heavily. See the regime/spike-capture page for which model actually",
                      "  degrades during volatility spikes -- that model is the riskier deploy",
                      "  choice even if it wins on pooled RMSE."]
        else:
            lines += [f"  {best_rmse} wins on both pooled RMSE and QLIKE -- check the regime",
                      "  breakdown to confirm that holds specifically in the High-vol regime too."]
        lines += ["", "  In practice: GARCH is the cheapest to retrain and the most interpretable",
                  "  -- a reasonable production default or sanity-check floor. An ML/DL model",
                  "  should only replace it if it demonstrably wins where GARCH is weakest",
                  "  WITHOUT losing GARCH's responsiveness to fresh shocks -- which is exactly",
                  "  what the regime table above checks."]
        text_page(pdf, "Results & Discussion", lines, fontsize=10)

    return pdf_path

meta = dict(ticker=TICKER, start=START, end=END,
            refit_every_garch=GARCH_REFIT_EVERY, refit_every_xgb=XGB_REFIT_EVERY,
            seq_len=LSTM_SEQ_LEN, n_seeds=LSTM_N_SEEDS)

pdf_path = build_report("quantbox_volatility_report.pdf", meta, raw, ds, train, test,
                         forecasts, results_df, dm_table, regime_df=regime_df, spike_df=spike_df,
                         lstm_stability=float(lstm_std.mean()))
print("Saved:", pdf_path)


# Download the PDF (Colab only -- comment this out if running locally)
try:
    from google.colab import files
    files.download(pdf_path)
except ImportError:
    print("Not running in Colab -- find the file at:", pdf_path)
