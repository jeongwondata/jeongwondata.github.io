# =============================================================
# CALLBACKS — All tabs
# =============================================================
from dashboard import (app, df, df_raw, NUM_COLS, CAT_COLS,
                        PALETTE, C_F, C_M, TEMPLATE, apply_layout)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import shapiro, normaltest, zscore, skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

from dash import Input, Output, State, dash_table, dcc
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# TAB 1: Load Data
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output('load-table-div', 'children'),
    Output('load-stats-div', 'children'),
    Output('load-status', 'children'),
    Input('btn-load', 'n_clicks'),
    State('load-rows', 'value'),
    State('load-col-filter', 'value'),
    prevent_initial_call=False,
)
def load_data(n, rows, col_filter):
    rows = int(rows) if rows else 10
    display = df.head(rows)
    if col_filter and col_filter != 'All':
        display = display[[col_filter]]

    table = dash_table.DataTable(
        data=display.astype(str).to_dict('records'),
        columns=[{'name': c, 'id': c} for c in display.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '6px',
                    'fontFamily': 'Arial', 'fontSize': '13px'},
        style_header={'backgroundColor': '#2c3e50', 'color': 'white',
                      'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}],
        page_size=rows,
    )

    desc = df[NUM_COLS].describe().round(2).reset_index()
    stats_table = dash_table.DataTable(
        data=desc.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in desc.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'padding': '5px', 'fontSize': '12px'},
        style_header={'backgroundColor': '#34495e', 'color': 'white',
                      'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#ecf0f1'}],
    )

    status = dbc.Alert('✅ Data loaded successfully!', color='success',
                       duration=3000)
    return table, stats_table, status


# ─────────────────────────────────────────────────────────────
# TAB 2: Data Cleaning
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output('clean-graph', 'figure'),
    Output('clean-alert', 'children'),
    Output('clean-alert', 'color'),
    Output('led-missing-before', 'value'),
    Output('led-missing-after', 'value'),
    Input('btn-clean', 'n_clicks'),
    State('clean-method', 'value'),
    State('clean-feature', 'value'),
    prevent_initial_call=False,
)
def clean_data(n, method, feature):
    missing_before = int(df_raw[feature].isnull().sum()) if feature in df_raw.columns else 0

    tmp = df_raw[[feature]].copy()
    if method == 'drop':
        tmp = tmp.dropna()
    elif method == 'mean':
        tmp[feature] = tmp[feature].fillna(tmp[feature].mean())
    elif method == 'median':
        tmp[feature] = tmp[feature].fillna(tmp[feature].median())
    elif method == 'mode':
        tmp[feature] = tmp[feature].fillna(tmp[feature].mode()[0])
    elif method == 'ffill':
        tmp[feature] = tmp[feature].ffill()

    missing_after = int(tmp[feature].isnull().sum())

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Before Cleaning', 'After Cleaning'])
    fig.add_trace(go.Histogram(x=df_raw[feature], name='Before',
                               marker_color=C_F, opacity=0.7,
                               nbinsx=40), row=1, col=1)
    fig.add_trace(go.Histogram(x=tmp[feature], name='After',
                               marker_color=C_M, opacity=0.7,
                               nbinsx=40), row=1, col=2)
    fig.update_layout(title=f'Data Cleaning: {feature} — Method: {method.title()}',
                      template=TEMPLATE, showlegend=True,
                      font=dict(family='Arial', size=13))

    method_desc = {
        'drop': 'Rows with missing values dropped.',
        'mean': f'Missing values filled with mean ({df_raw[feature].mean():.2f}).',
        'median': f'Missing values filled with median ({df_raw[feature].median():.2f}).',
        'mode': f'Missing values filled with mode.',
        'ffill': 'Missing values forward-filled from previous row.',
    }
    msg   = f'Method: {method.title()} | {method_desc.get(method,"")} | Missing before: {missing_before} → after: {missing_after}'
    color = 'success' if missing_after == 0 else 'warning'

    return fig, msg, color, str(missing_before), str(missing_after)


# ─────────────────────────────────────────────────────────────
# TAB 3: Outlier Detection
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output('outlier-box', 'figure'),
    Output('outlier-hist', 'figure'),
    Output('outlier-alert', 'children'),
    Output('outlier-alert', 'color'),
    Output('gauge-outlier-pct', 'value'),
    Output('led-outlier-n', 'value'),
    Output('led-clean-n', 'value'),
    Input('btn-outlier', 'n_clicks'),
    State('outlier-method', 'value'),
    State('outlier-feature', 'value'),
    State('outlier-k', 'value'),
    prevent_initial_call=False,
)
def detect_outliers(n, method, feature, k):
    series = df[feature].dropna()

    if method == 'iqr':
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        mask = (series < Q1 - k*IQR) | (series > Q3 + k*IQR)
        bounds = (Q1 - k*IQR, Q3 + k*IQR)
    elif method == 'zscore':
        z = np.abs(zscore(series))
        mask = z > 3
        bounds = (series.mean() - 3*series.std(), series.mean() + 3*series.std())
    else:  # mod_z
        med = series.median()
        mad = np.median(np.abs(series - med))
        mod_z = 0.6745 * np.abs(series - med) / (mad + 1e-9)
        mask = mod_z > 3.5
        bounds = None

    n_out   = int(mask.sum())
    n_clean = len(series) - n_out
    pct     = round(100 * n_out / len(series), 2)

    # Box plot
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=series[~mask], name='Normal',
                             marker_color=C_M, boxmean=True))
    fig_box.add_trace(go.Box(y=series[mask], name='Outlier',
                             marker_color=C_F, boxmean=True))
    apply_layout(fig_box, f'Box Plot: {feature} (outliers highlighted)',
                 '', feature)

    # Histogram with outlier shading
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=series[~mask], name='Normal',
                                    marker_color=C_M, opacity=0.7, nbinsx=50))
    fig_hist.add_trace(go.Histogram(x=series[mask], name='Outlier',
                                    marker_color=C_F, opacity=0.7, nbinsx=50))
    if bounds:
        fig_hist.add_vline(x=bounds[0], line_dash='dash', line_color='red',
                           annotation_text=f'Lower: {bounds[0]:.2f}')
        fig_hist.add_vline(x=bounds[1], line_dash='dash', line_color='red',
                           annotation_text=f'Upper: {bounds[1]:.2f}')
    apply_layout(fig_hist, f'Distribution: {feature} — Outlier Detection',
                 feature, 'Count')
    fig_hist.update_layout(barmode='overlay')

    msg   = (f'Method: {method.upper()} | Feature: {feature} | '
             f'Outliers: {n_out:,} ({pct:.2f}%) | Cleaned rows: {n_clean:,}')
    color = 'danger' if pct > 10 else ('warning' if pct > 5 else 'success')

    return fig_box, fig_hist, msg, color, pct, str(n_out), str(n_clean)


# ─────────────────────────────────────────────────────────────
# TAB 4: PCA
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output('pca-scree', 'figure'),
    Output('pca-biplot', 'figure'),
    Output('pca-cumvar', 'figure'),
    Output('grad-bar-pca', 'value'),
    Output('pca-cond-alert', 'children'),
    Input('btn-pca', 'n_clicks'),
    State('pca-features', 'value'),
    State('pca-n', 'value'),
    State('pca-color', 'value'),
    prevent_initial_call=False,
)
def run_pca(n, features, n_comp, color_by):
    features = features or NUM_COLS
    X = df[features].dropna()
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pca    = PCA()
    pca.fit(X_sc)

    evr     = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(evr)
    cond_n  = np.linalg.cond(X_sc)

    # Scree
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(x=[f'PC{i+1}' for i in range(len(evr))],
                               y=evr, marker_color=PALETTE[0],
                               text=[f'{v:.1f}%' for v in evr],
                               textposition='outside', name='Var %'))
    apply_layout(fig_scree, 'Scree Plot — Explained Variance per PC',
                 'Principal Component', 'Explained Variance (%)')

    # Biplot
    pca2   = PCA(n_components=2)
    X_2d   = pca2.fit_transform(X_sc)
    samp   = np.random.choice(len(X_2d), min(2000, len(X_2d)), replace=False)
    color_vals = df[color_by].iloc[X.index].iloc[samp].values if color_by else None

    fig_bi = px.scatter(x=X_2d[samp, 0], y=X_2d[samp, 1],
                        color=color_vals,
                        labels={'x': f'PC1 ({evr[0]:.1f}%)',
                                'y': f'PC2 ({evr[1]:.1f}%)',
                                'color': color_by},
                        opacity=0.5, template=TEMPLATE,
                        color_discrete_sequence=PALETTE)
    # Loadings arrows
    scale = 3
    for i, feat in enumerate(features):
        fig_bi.add_annotation(
            x=pca2.components_[0, i]*scale,
            y=pca2.components_[1, i]*scale,
            ax=0, ay=0,
            xanchor='center', yanchor='bottom',
            text=feat, font=dict(size=10, color='red'),
            arrowcolor='red', arrowwidth=1.5, arrowhead=2,
        )
    apply_layout(fig_bi, 'PCA Biplot (PC1 vs PC2)',
                 f'PC1 ({evr[0]:.1f}%)', f'PC2 ({evr[1]:.1f}%)')

    # Cumulative variance
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=list(range(1, len(cum_var)+1)), y=cum_var,
                                 mode='lines+markers+text',
                                 text=[f'{v:.1f}%' for v in cum_var],
                                 textposition='top center',
                                 marker_color=PALETTE[1], line_width=2.5,
                                 name='Cumulative Var'))
    fig_cum.add_hline(y=95, line_dash='dash', line_color='red',
                      annotation_text='95% threshold')
    fig_cum.add_hline(y=80, line_dash='dash', line_color='green',
                      annotation_text='80% threshold')
    apply_layout(fig_cum, 'Cumulative Explained Variance',
                 'Number of Components', 'Cumulative Variance (%)')

    n_for_95  = int(np.argmax(cum_var >= 95)) + 1
    cond_msg  = (f'Condition Number: {cond_n:.2f} | '
                 f'PCs needed for 95% variance: {n_for_95} | '
                 f'Singular values range: {pca.singular_values_.min():.2f} – {pca.singular_values_.max():.2f}')

    grad_val = round(float(cum_var[min(int(n_comp)-1, len(cum_var)-1)]), 1)
    return fig_scree, fig_bi, fig_cum, grad_val, cond_msg


# ─────────────────────────────────────────────────────────────
# TAB 5: Normality Test
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output('norm-hist', 'figure'),
    Output('norm-qq', 'figure'),
    Output('norm-result', 'children'),
    Output('norm-result', 'color'),
    Output('thermo-skew', 'value'),
    Output('thermo-kurt', 'value'),
    Output('norm-switch', 'on'),
    Output('knob-pval', 'value'),
    Input('btn-norm', 'n_clicks'),
    State('norm-feature', 'value'),
    State('norm-method', 'value'),
    State('norm-color', 'value'),
    prevent_initial_call=False,
)
def run_normality(n, feature, method, color_val):
    series  = df[feature].dropna()
    hex_col = color_val.get('hex', '#3498db') if isinstance(color_val, dict) else '#3498db'
    sk_val  = float(skew(series))
    kt_val  = float(kurtosis(series))

    # Histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=series, nbinsx=50,
                                    marker_color=hex_col, opacity=0.7,
                                    histnorm='probability density', name='Data'))
    x_range = np.linspace(series.min(), series.max(), 200)
    mu, sig = series.mean(), series.std()
    fig_hist.add_trace(go.Scatter(x=x_range,
                                  y=stats.norm.pdf(x_range, mu, sig),
                                  mode='lines', line=dict(color='red', width=2.5),
                                  name='Normal PDF'))
    fig_hist.add_vline(x=mu, line_dash='dash', line_color='orange',
                       annotation_text=f'Mean: {mu:.2f}')
    apply_layout(fig_hist, f'Distribution of {feature}', feature, 'Density')

    # QQ Plot
    qq = sm.ProbPlot(series)
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=qq.theoretical_quantiles,
                                y=qq.sample_quantiles,
                                mode='markers',
                                marker=dict(color=hex_col, size=3, opacity=0.5),
                                name='Sample'))
    lim = max(abs(qq.theoretical_quantiles.min()),
              abs(qq.theoretical_quantiles.max()))
    fig_qq.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim],
                                mode='lines', line=dict(color='red', width=2),
                                name='Normal line'))
    apply_layout(fig_qq, f'QQ Plot: {feature}',
                 'Theoretical Quantiles', 'Sample Quantiles')

    # Tests
    samp_n = series.sample(min(5000, len(series)), random_state=42)
    results = []
    min_p   = 1.0

    if method in ['shapiro', 'both']:
        sw_s, sw_p = shapiro(samp_n.sample(min(5000, len(samp_n)), random_state=42))
        results.append(f'Shapiro-Wilk: W={sw_s:.4f}, p={sw_p:.4f} '
                       f'→ {"Normal ✅" if sw_p>0.05 else "Not Normal ❌"}')
        min_p = min(min_p, sw_p)

    if method in ['dagostino', 'both']:
        da_s, da_p = normaltest(samp_n)
        results.append(f"D'Agostino K²: stat={da_s:.4f}, p={da_p:.4f} "
                       f'→ {"Normal ✅" if da_p>0.05 else "Not Normal ❌"}')
        min_p = min(min_p, da_p)

    results.append(f'Skewness: {sk_val:.2f} | Kurtosis: {kt_val:.2f}')
    is_normal = min_p > 0.05
    msg_color = 'success' if is_normal else 'danger'

    sk_clamped = float(np.clip(sk_val, -5, 5))
    kt_clamped = float(np.clip(kt_val, -3, 10))
    p_display  = float(np.clip(round(min_p, 4), 0, 1))

    return (fig_hist, fig_qq, ' | '.join(results), msg_color,
            sk_clamped, kt_clamped, is_normal, p_display)


# ─────────────────────────────────────────────────────────────
# TAB 6: Data Transformation
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output('trans-before', 'figure'),
    Output('trans-after', 'figure'),
    Output('transform-alert', 'children'),
    Output('led-skew-before', 'value'),
    Output('led-skew-after', 'value'),
    Output('led-mean-before', 'value'),
    Output('led-mean-after', 'value'),
    Input('btn-transform', 'n_clicks'),
    State('trans-feature', 'value'),
    State('trans-method', 'value'),
    prevent_initial_call=False,
)
def transform_data(n, feature, method):
    series = df[feature].dropna()

    if method == 'log':
        transformed = np.log1p(series)
        label = 'Log (log1p)'
    elif method == 'sqrt':
        transformed = np.sqrt(np.abs(series))
        label = 'Square Root'
    elif method == 'zscore':
        transformed = pd.Series(zscore(series))
        label = 'Z-Score'
    elif method == 'minmax':
        mn, mx = series.min(), series.max()
        transformed = (series - mn) / (mx - mn + 1e-9)
        label = 'Min-Max'
    elif method == 'boxcox':
        shifted = series - series.min() + 1
        transformed, _ = stats.boxcox(shifted)
        transformed = pd.Series(transformed)
        label = 'Box-Cox'
    else:
        transformed = series
        label = 'Original'

    sk_before = round(float(skew(series)), 2)
    sk_after  = round(float(skew(transformed)), 2)
    mu_before = round(float(series.mean()), 2)
    mu_after  = round(float(transformed.mean()), 2)

    x_b = np.linspace(series.min(), series.max(), 200)
    x_a = np.linspace(transformed.min(), transformed.max(), 200)

    fig_b = go.Figure()
    fig_b.add_trace(go.Histogram(x=series, nbinsx=50,
                                  marker_color=C_F, opacity=0.7,
                                  histnorm='probability density', name='Original'))
    fig_b.add_trace(go.Scatter(x=x_b,
                                y=stats.norm.pdf(x_b, series.mean(), series.std()),
                                mode='lines', line=dict(color='black', dash='dash'),
                                name='Normal PDF'))
    apply_layout(fig_b, f'Before: {feature} (Skew={sk_before:.2f})',
                 feature, 'Density')

    fig_a = go.Figure()
    fig_a.add_trace(go.Histogram(x=transformed, nbinsx=50,
                                  marker_color=C_M, opacity=0.7,
                                  histnorm='probability density', name=label))
    fig_a.add_trace(go.Scatter(x=x_a,
                                y=stats.norm.pdf(x_a, transformed.mean(), transformed.std()),
                                mode='lines', line=dict(color='black', dash='dash'),
                                name='Normal PDF'))
    apply_layout(fig_a, f'After: {label} (Skew={sk_after:.2f})',
                 f'{feature} ({label})', 'Density')

    msg = (f'Transformation: {label} | '
           f'Skewness: {sk_before:.2f} → {sk_after:.2f} | '
           f'Mean: {mu_before:.2f} → {mu_after:.2f}')

    return (fig_b, fig_a, msg,
            f'{sk_before:.2f}', f'{sk_after:.2f}',
            f'{mu_before:.2f}', f'{mu_after:.2f}')


@app.callback(
    Output('download-transformed', 'data'),
    Input('btn-download-transform', 'n_clicks'),
    State('trans-feature', 'value'),
    State('trans-method', 'value'),
    prevent_initial_call=True,
)
def download_transformed(n, feature, method):
    series = df[feature].dropna()
    if method == 'log':
        out = np.log1p(series)
    elif method == 'sqrt':
        out = np.sqrt(np.abs(series))
    elif method == 'zscore':
        out = pd.Series(zscore(series), index=series.index)
    elif method == 'minmax':
        mn, mx = series.min(), series.max()
        out = (series - mn) / (mx - mn + 1e-9)
    elif method == 'boxcox':
        shifted = series - series.min() + 1
        out_arr, _ = stats.boxcox(shifted)
        out = pd.Series(out_arr, index=series.index)
    else:
        out = series
    export = df.copy()
    export[f'{feature}_transformed'] = out
    return dcc.send_data_frame(export.to_csv, f'{feature}_{method}_transformed.csv',
                               index=False)


# ─────────────────────────────────────────────────────────────
# TAB 7: Numerical Plots
# ─────────────────────────────────────────────────────────────
# ── DropdownMenu: single callback with all 13 inputs ──
@app.callback(
    Output('num-plot-type', 'data'),
    Input('dd-hist',      'n_clicks'),
    Input('dd-kde',       'n_clicks'),
    Input('dd-box',       'n_clicks'),
    Input('dd-violin',    'n_clicks'),
    Input('dd-scatter',   'n_clicks'),
    Input('dd-qq',        'n_clicks'),
    Input('dd-dist',      'n_clicks'),
    Input('dd-area',      'n_clicks'),
    Input('dd-rug',       'n_clicks'),
    Input('dd-hexbin',    'n_clicks'),
    Input('dd-scatter3d', 'n_clicks'),
    Input('dd-contour',   'n_clicks'),
    Input('dd-joint',     'n_clicks'),
    prevent_initial_call=True,
)
def update_plot_type(*args):
    from dash import ctx
    triggered = ctx.triggered_id
    mapping = {
        'dd-hist':'hist', 'dd-kde':'kde', 'dd-box':'box',
        'dd-violin':'violin', 'dd-scatter':'scatter', 'dd-qq':'qq',
        'dd-dist':'dist', 'dd-area':'area', 'dd-rug':'rug',
        'dd-hexbin':'hexbin', 'dd-scatter3d':'scatter3d',
        'dd-contour':'contour', 'dd-joint':'joint',
    }
    return mapping.get(triggered, 'hist')

@app.callback(
    Output('num-graph', 'figure'),
    Output('num-stats-alert', 'children'),
    Input('btn-num-plot', 'n_clicks'),
    State('num-x', 'value'),
    State('num-y', 'value'),
    State('num-hue', 'value'),
    State('num-plot-type', 'data'),
    State('num-bins', 'value'),
    prevent_initial_call=False,
)
def num_plot(n, x, y, hue, ptype, bins_range):
    hue_col = None if (not hue or hue == 'None') else hue
    samp    = df.sample(min(5000, len(df)), random_state=42)
    # RangeSlider returns [min, max] — use max value as bin count
    bins = bins_range[1] if isinstance(bins_range, list) else (bins_range or 40)

    # helper: build kwargs only when hue_col is set
    def hkw():
        return {'color': hue_col, 'color_discrete_sequence': PALETTE} if hue_col else {}

    if ptype == 'hist':
        fig = go.Figure()
        series_x = df[x].dropna()
        if hue_col:
            for i, (cat, grp) in enumerate(df.groupby(hue_col)):
                fig.add_trace(go.Histogram(
                    x=grp[x].dropna(), name=str(cat),
                    nbinsx=bins, opacity=0.6,
                    marker_color=PALETTE[i % len(PALETTE)]))
        else:
            fig.add_trace(go.Histogram(
                x=series_x, nbinsx=bins,
                opacity=0.7, marker_color=C_M, name=x))
        # KDE overlay
        kde_fn = stats.gaussian_kde(series_x)
        xs = np.linspace(series_x.min(), series_x.max(), 300)
        scale = len(series_x) * (series_x.max() - series_x.min()) / bins
        fig.add_trace(go.Scatter(
            x=xs, y=kde_fn(xs) * scale,
            mode='lines', line=dict(color='red', width=2.5),
            name='KDE'))
        fig.update_layout(barmode='overlay', template=TEMPLATE)
        apply_layout(fig, f'Histogram + KDE: {x}', x, 'Count')

    elif ptype == 'kde':
        fig = go.Figure()
        if hue_col:
            for cat, grp in df.groupby(hue_col):
                kde = stats.gaussian_kde(grp[x].dropna())
                xs  = np.linspace(grp[x].min(), grp[x].max(), 300)
                fig.add_trace(go.Scatter(x=xs, y=kde(xs), mode='lines',
                                         fill='tozeroy', opacity=0.5,
                                         name=str(cat)))
        else:
            kde = stats.gaussian_kde(df[x].dropna())
            xs  = np.linspace(df[x].min(), df[x].max(), 300)
            fig.add_trace(go.Scatter(x=xs, y=kde(xs), mode='lines',
                                     fill='tozeroy', line_color=C_M))
        apply_layout(fig, f'KDE Plot: {x}', x, 'Density')

    elif ptype == 'box':
        fig = px.box(df, y=x, template=TEMPLATE, points='outliers', **hkw())
        if not hue_col:
            fig.update_traces(marker_color=C_M)
        apply_layout(fig, f'Box Plot: {x}', '', x)

    elif ptype == 'violin':
        fig = px.violin(df, y=x, box=True, template=TEMPLATE, **hkw())
        if not hue_col:
            fig.update_traces(marker_color=C_M)
        apply_layout(fig, f'Violin Plot: {x}', '', x)

    elif ptype == 'scatter':
        fig = px.scatter(samp, x=x, y=y, trendline='ols', opacity=0.5,
                         template=TEMPLATE, **hkw())
        if not hue_col:
            fig.update_traces(marker_color=C_M, selector=dict(mode='markers'))
        r = df[[x, y]].corr().iloc[0, 1]
        apply_layout(fig, f'Scatter + Regression: {x} vs {y} (r={r:.2f})', x, y)

    elif ptype == 'qq':
        qq  = sm.ProbPlot(df[x].dropna())
        lim = max(abs(qq.theoretical_quantiles).max(), abs(qq.sample_quantiles).max())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=qq.theoretical_quantiles,
                                 y=qq.sample_quantiles,
                                 mode='markers',
                                 marker=dict(color=C_M, size=3, opacity=0.5),
                                 name='Sample'))
        fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim],
                                 mode='lines', line=dict(color='red', width=2),
                                 name='Normal line'))
        apply_layout(fig, f'QQ Plot: {x}',
                     'Theoretical Quantiles', 'Sample Quantiles')

    elif ptype == 'dist':
        hist_data = [df[x].dropna().values]
        fig = ff.create_distplot(hist_data, [x], colors=[C_M],
                                 bin_size=(df[x].max()-df[x].min())/bins)
        apply_layout(fig, f'Dist Plot: {x}', x, 'Density')

    elif ptype == 'area':
        monthly = df.groupby('Month')[x].mean().reset_index()
        fig = px.area(monthly, x='Month', y=x,
                      color_discrete_sequence=PALETTE, template=TEMPLATE)
        apply_layout(fig, f'Area Plot: Monthly Avg {x}', 'Month', f'Avg {x}')

    elif ptype == 'rug':
        fig = go.Figure()
        if hue_col:
            for i, (cat, grp) in enumerate(df.groupby(hue_col)):
                samp_rug = grp[x].sample(min(2000, len(grp)), random_state=42)
                kde      = stats.gaussian_kde(grp[x].dropna())
                xs       = np.linspace(grp[x].min(), grp[x].max(), 300)
                fig.add_trace(go.Scatter(x=xs, y=kde(xs), mode='lines',
                                         name=str(cat),
                                         line_color=PALETTE[i % len(PALETTE)]))
                fig.add_trace(go.Scatter(
                    x=samp_rug, y=[-0.002*(i+1)] * len(samp_rug),
                    mode='markers', marker=dict(symbol='line-ns-open',
                                                color=PALETTE[i % len(PALETTE)],
                                                size=8, opacity=0.4),
                    showlegend=False))
        else:
            kde = stats.gaussian_kde(df[x].dropna())
            xs  = np.linspace(df[x].min(), df[x].max(), 300)
            fig.add_trace(go.Scatter(x=xs, y=kde(xs), mode='lines',
                                     line_color=C_M, name='KDE'))
            samp_rug = df[x].sample(2000, random_state=42)
            fig.add_trace(go.Scatter(
                x=samp_rug, y=[-0.002]*len(samp_rug),
                mode='markers', marker=dict(symbol='line-ns-open',
                                            color=C_F, size=8, opacity=0.3),
                name='Rug'))
        apply_layout(fig, f'Rug Plot: {x}', x, 'Density')

    elif ptype == 'hexbin':
        fig = px.density_heatmap(samp, x=x, y=y, nbinsx=30, nbinsy=30,
                                 color_continuous_scale='YlOrRd',
                                 template=TEMPLATE)
        apply_layout(fig, f'Hexbin / Density Heatmap: {x} vs {y}', x, y)

    elif ptype == 'scatter3d':
        z_col = [c for c in NUM_COLS if c not in [x, y]][0]
        fig   = px.scatter_3d(samp, x=x, y=y, z=z_col, opacity=0.5,
                              template=TEMPLATE, **hkw())
        if not hue_col:
            fig.update_traces(marker=dict(color=C_M))
        fig.update_layout(title=f'3D Scatter: {x} × {y} × {z_col}',
                          font=dict(family='Arial', size=13))

    elif ptype == 'contour':
        fig = go.Figure()
        fig.add_trace(go.Histogram2dContour(x=samp[x], y=samp[y],
                                             colorscale='Blues',
                                             contours_coloring='fill',
                                             showscale=True))
        fig.add_trace(go.Histogram2dContour(x=samp[x], y=samp[y],
                                             colorscale='Blues',
                                             contours_coloring='lines',
                                             showscale=False,
                                             line_width=1))
        apply_layout(fig, f'Contour Plot: {x} vs {y}', x, y)

    elif ptype == 'joint':
        fig = make_subplots(rows=2, cols=2,
                            column_widths=[0.8, 0.2],
                            row_heights=[0.2, 0.8])
        samp2 = df[[x, y]].sample(min(3000, len(df)), random_state=42)
        fig.add_trace(go.Histogram(x=samp2[x], marker_color=C_M,
                                   opacity=0.7, showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=samp2[x], y=samp2[y], mode='markers',
                                 marker=dict(color=C_M, size=3, opacity=0.4),
                                 name='Data'), row=2, col=1)
        fig.add_trace(go.Histogram2dContour(x=samp2[x], y=samp2[y],
                                             colorscale='Blues',
                                             showscale=False,
                                             line_width=1), row=2, col=1)
        fig.add_trace(go.Histogram(y=samp2[y], marker_color=C_F,
                                   opacity=0.7, showlegend=False), row=2, col=2)
        fig.update_layout(title=f'Joint Plot: {x} vs {y}',
                          template=TEMPLATE, font=dict(family='Arial', size=13))

    else:
        fig = go.Figure()
        apply_layout(fig, 'Select a plot type')

    stats_msg = (f'{x}: Mean={df[x].mean():.2f}, '
                 f'Std={df[x].std():.2f}, '
                 f'Skew={skew(df[x].dropna()):.2f}')
    return fig, stats_msg


# ─────────────────────────────────────────────────────────────
# TAB 8: Categorical Plots
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output('cat-graph', 'figure'),
    Output('cat-info-alert', 'children'),
    Input('btn-cat-plot', 'n_clicks'),
    State('cat-x', 'value'),
    State('cat-y', 'value'),
    State('cat-hue', 'value'),
    State('cat-plot-type', 'value'),
    State('cat-agg', 'value'),
    prevent_initial_call=False,
)
def cat_plot(n, x, y, hue, ptype, agg):
    hue_col = None if (not hue or hue == 'None') else hue

    def hkw():
        return {'color': hue_col, 'color_discrete_sequence': PALETTE} if hue_col else {}

    if ptype == 'bar':
        grp = df.groupby(x)[y].agg(agg).sort_values(ascending=False).reset_index()
        fig = px.bar(grp, x=x, y=y, text=y,
                     color=x, color_discrete_sequence=PALETTE,
                     template=TEMPLATE)
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        apply_layout(fig, f'Bar Plot: {agg.title()} {y} by {x}', x, y)

    elif ptype == 'bar_stack':
        if hue_col:
            grp = df.groupby([x, hue_col])[y].agg(agg).reset_index()
            fig = px.bar(grp, x=x, y=y, barmode='stack',
                         template=TEMPLATE, **hkw())
        else:
            grp = df.groupby(x)[y].agg(agg).reset_index()
            fig = px.bar(grp, x=x, y=y, template=TEMPLATE,
                         color_discrete_sequence=PALETTE)
        apply_layout(fig, f'Stacked Bar: {y} by {x}', x, y)

    elif ptype == 'bar_group':
        if hue_col:
            grp = df.groupby([x, hue_col])[y].agg(agg).reset_index()
            fig = px.bar(grp, x=x, y=y, barmode='group',
                         template=TEMPLATE, text=y, **hkw())
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        else:
            grp = df.groupby(x)[y].agg(agg).reset_index()
            fig = px.bar(grp, x=x, y=y, template=TEMPLATE,
                         color_discrete_sequence=PALETTE)
        apply_layout(fig, f'Grouped Bar: {y} by {x} & {hue_col}', x, y)

    elif ptype == 'count':
        fig = px.histogram(df, x=x, barmode='group',
                           template=TEMPLATE, text_auto=True, **hkw())
        apply_layout(fig, f'Count Plot: {x}', x, 'Count')

    elif ptype == 'pie':
        vc  = df[x].value_counts().reset_index()
        fig = px.pie(vc, names=x, values='count',
                     color_discrete_sequence=PALETTE, template=TEMPLATE,
                     hole=0.3)
        fig.update_traces(textposition='inside',
                          texttemplate='%{label}<br>%{percent:.2%}')
        fig.update_layout(title=f'Pie Chart: {x} Distribution',
                          font=dict(family='Arial', size=13))

    elif ptype == 'box':
        fig = px.box(df, x=x, y=y, points='outliers',
                     template=TEMPLATE, **hkw())
        apply_layout(fig, f'Box Plot: {y} by {x}', x, y)

    elif ptype == 'violin':
        fig = px.violin(df, x=x, y=y, box=True,
                        template=TEMPLATE, **hkw())
        apply_layout(fig, f'Violin Plot: {y} by {x}', x, y)

    elif ptype == 'strip':
        samp = df.sample(min(3000, len(df)), random_state=42)
        fig  = px.strip(samp, x=x, y=y, template=TEMPLATE,
                        stripmode='overlay', **hkw())
        apply_layout(fig, f'Strip Plot: {y} by {x}', x, y)

    elif ptype == 'swarm':
        samp = df.sample(min(800, len(df)), random_state=42)
        fig  = px.strip(samp, x=x, y=y, template=TEMPLATE, **hkw())
        apply_layout(fig, f'Swarm Plot: {y} by {x}', x, y)

    elif ptype == 'boxen':
        fig = px.box(df, x=x, y=y, points=False,
                     notched=True, template=TEMPLATE, **hkw())
        apply_layout(fig, f'Boxen Plot: {y} by {x}', x, y)

    else:
        fig = go.Figure()

    n_cats = df[x].nunique()
    info   = (f'{x}: {n_cats} unique categories | '
              f'Most common: {df[x].value_counts().index[0]} '
              f'({df[x].value_counts().iloc[0]:,})')
    return fig, info


# ─────────────────────────────────────────────────────────────
# TAB 9: Statistics
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output('stats-heatmap', 'figure'),
    Output('stats-corr-alert', 'children'),
    Input('btn-stats-heat', 'n_clicks'),
    State('stats-heatmap-features', 'value'),
    State('stats-cmap', 'value'),
    prevent_initial_call=False,
)
def stats_heatmap(n, features, cmap):
    features = features or NUM_COLS[:6]
    corr = df[features].corr().round(2)
    fig  = px.imshow(corr, text_auto='.2f', color_continuous_scale=cmap,
                     zmin=-1, zmax=1, template=TEMPLATE,
                     aspect='auto')
    apply_layout(fig, 'Pearson Correlation Heatmap', '', '')

    # Find highest absolute correlation (off-diagonal)
    corr_abs = corr.abs().mask(np.eye(len(corr), dtype=bool), 0)
    idx  = corr_abs.stack().idxmax()
    val  = corr.loc[idx[0], idx[1]]
    msg  = (f'Strongest correlation: {idx[0]} ↔ {idx[1]} '
            f'(r = {val:.2f})')
    return fig, msg


@app.callback(
    Output('stats-pair', 'figure'),
    Input('btn-stats-pair', 'n_clicks'),
    State('stats-pair-features', 'value'),
    State('stats-pair-hue', 'value'),
    prevent_initial_call=False,
)
def stats_pair(n, features, hue):
    features = (features or NUM_COLS[:4])[:5]
    samp = df[features + [hue]].sample(min(2000, len(df)), random_state=42)
    fig  = px.scatter_matrix(samp, dimensions=features, color=hue,
                              color_discrete_sequence=PALETTE,
                              opacity=0.4, template=TEMPLATE)
    fig.update_traces(diagonal_visible=True, showupperhalf=False)
    apply_layout(fig, f'Scatter Matrix (Pair Plot) — color: {hue}', '', '')
    return fig


@app.callback(
    Output('stats-kde', 'figure'),
    Input('btn-stats-kde', 'n_clicks'),
    State('stats-kde-x', 'value'),
    State('stats-kde-y', 'value'),
    prevent_initial_call=False,
)
def stats_kde(n, x, y):
    samp = df[[x, y]].sample(min(3000, len(df)), random_state=42)
    fig  = go.Figure()
    fig.add_trace(go.Histogram2dContour(
        x=samp[x], y=samp[y],
        colorscale='Viridis',
        contours_coloring='fill',
        showscale=True,
        colorbar=dict(title='Density')
    ))
    fig.add_trace(go.Histogram2dContour(
        x=samp[x], y=samp[y],
        colorscale='Viridis',
        contours_coloring='lines',
        showscale=False,
        line_width=0.8,
        line_color='white'
    ))
    apply_layout(fig, f'Multivariate KDE: {x} vs {y}', x, y)
    return fig


@app.callback(
    Output('stats-table-div', 'children'),
    Input('main-tabs', 'active_tab'),
)
def stats_table(tab):
    desc = df[NUM_COLS].describe().round(2)
    extra = pd.DataFrame({
        'skewness': [skew(df[c].dropna()) for c in NUM_COLS],
        'kurtosis': [kurtosis(df[c].dropna()) for c in NUM_COLS],
    }, index=NUM_COLS).round(2).T
    desc = pd.concat([desc, extra])
    desc = desc.reset_index().rename(columns={'index': 'stat'})
    return dash_table.DataTable(
        data=desc.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in desc.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'padding': '4px', 'fontSize': '11px'},
        style_header={'backgroundColor': '#2c3e50', 'color': 'white',
                      'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f0f4f8'}],
        page_size=12,
    )
