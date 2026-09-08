"""
Regime Detection via HMM (Baum-Welch), Markov Switching (statsmodels)
ou Change-Point Detection (Binary Segmentation).

Uso:
    from engine.regime_detection import detect_regimes
    result = detect_regimes(df_ohlcv, method="hmm", n_states=0, features=["log_return", "volatility"])
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")


# Feature engineering
def _compute_features(df: pd.DataFrame, feature_list: list, vol_window: int = 20) -> pd.DataFrame:
    """Calcula features para o modelo a partir de OHLCV."""
    feats = pd.DataFrame(index=df.index)

    close = df["Close"] if "Close" in df.columns else df["close"]
    log_ret = np.log(close / close.shift(1))

    if "log_return" in feature_list:
        feats["log_return"] = log_ret

    if "volatility" in feature_list:
        feats["volatility"] = log_ret.rolling(vol_window).std()

    if "volume" in feature_list:
        vol_col = "Volume" if "Volume" in df.columns else "volume" if "volume" in df.columns else None
        if vol_col:
            feats["volume"] = np.log1p(df[vol_col].astype(float))

    feats.dropna(inplace=True)
    return feats


# Gaussian HMM via Baum-Welch (EM)
def _log_multivariate_normal_density(X, means, covars):
    """Log-pdf de N(mean, cov) para cada (obs, estado). Retorna (T, n_states)."""
    T, D = X.shape
    n_states = len(means)
    log_lik = np.zeros((T, n_states))

    for s in range(n_states):
        diff = X - means[s]  # (T, D)
        cov = covars[s]
        # Regulariza covariancia para estabilidade
        cov_reg = cov + np.eye(D) * 1e-6
        try:
            L = np.linalg.cholesky(cov_reg)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            solved = np.linalg.solve(L, diff.T)  # (D, T)
            mahal = np.sum(solved ** 2, axis=0)  # (T,)
            log_lik[:, s] = -0.5 * (D * np.log(2 * np.pi) + log_det + mahal)
        except np.linalg.LinAlgError:
            log_lik[:, s] = -1e10

    return log_lik


def _forward(log_lik, log_trans, log_start, n_states):
    """Forward pass (scaled). Retorna log_alpha (T, n_states) e log_scales (T,)."""
    T = len(log_lik)
    log_alpha = np.full((T, n_states), -np.inf)
    log_scale = np.zeros(T)

    # t = 0
    log_alpha[0] = log_start + log_lik[0]
    log_scale[0] = np.logaddexp.reduce(log_alpha[0])
    log_alpha[0] -= log_scale[0]

    for t in range(1, T):
        for j in range(n_states):
            log_alpha[t, j] = np.logaddexp.reduce(log_alpha[t - 1] + log_trans[:, j]) + log_lik[t, j]
        log_scale[t] = np.logaddexp.reduce(log_alpha[t])
        log_alpha[t] -= log_scale[t]

    return log_alpha, log_scale


def _backward(log_lik, log_trans, log_scale, n_states):
    """Backward pass (scaled). Retorna log_beta (T, n_states)."""
    T = len(log_lik)
    log_beta = np.full((T, n_states), -np.inf)
    log_beta[-1] = 0.0  # log(1)

    for t in range(T - 2, -1, -1):
        for j in range(n_states):
            log_beta[t, j] = np.logaddexp.reduce(
                log_trans[j, :] + log_lik[t + 1] + log_beta[t + 1]
            )
        log_beta[t] -= log_scale[t + 1]

    return log_beta


def _fit_hmm(X: np.ndarray, n_states: int, n_iter: int = 50, tol: float = 1e-4):
    """
    Gaussian HMM via Baum-Welch (EM) completo.
    Inicializa com K-Means, depois itera EM ate convergir.
    """
    T, D = X.shape

    # ── Inicializacao via K-Means ──
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_states, n_init=10, random_state=42)
    init_labels = kmeans.fit_predict(X)

    means = np.array([X[init_labels == s].mean(axis=0) if (init_labels == s).sum() > 0
                       else X.mean(axis=0) + np.random.randn(D) * 0.01
                       for s in range(n_states)])

    covars = np.array([np.cov(X[init_labels == s].T) + np.eye(D) * 1e-4
                        if (init_labels == s).sum() > D
                        else np.eye(D) * np.var(X, axis=0).mean()
                        for s in range(n_states)])
    # Garante 3D
    if covars.ndim == 1:
        covars = np.array([np.eye(D) * np.var(X, axis=0).mean() for _ in range(n_states)])
    elif covars.ndim == 2 and D == 1:
        covars = covars.reshape(n_states, 1, 1)

    # Transmat: forte persistencia (diag=0.95)
    transmat = np.full((n_states, n_states), 0.05 / (n_states - 1)) if n_states > 1 else np.ones((1, 1))
    np.fill_diagonal(transmat, 0.95)
    # Normaliza linhas
    transmat /= transmat.sum(axis=1, keepdims=True)

    startprob = np.ones(n_states) / n_states

    prev_ll = -np.inf

    for iteration in range(n_iter):
        # ── E-step ──
        log_lik = _log_multivariate_normal_density(X, means, covars)
        log_trans = np.log(transmat + 1e-300)
        log_start = np.log(startprob + 1e-300)

        log_alpha, log_scale = _forward(log_lik, log_trans, log_start, n_states)
        log_beta = _backward(log_lik, log_trans, log_scale, n_states)

        # Log-likelihood total
        ll = np.sum(log_scale)

        # Convergencia
        if iteration > 0 and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

        # Gamma (posteriors)
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Xi (transicoes): xi[t, i, j] = P(s_t=i, s_{t+1}=j | obs)
        xi = np.zeros((T - 1, n_states, n_states))
        for t in range(T - 1):
            for i in range(n_states):
                for j in range(n_states):
                    xi[t, i, j] = log_alpha[t, i] + log_trans[i, j] + log_lik[t + 1, j] + log_beta[t + 1, j]
            # Normaliza
            xi_max = xi[t].max()
            xi[t] = np.exp(xi[t] - xi_max)
            xi_sum = xi[t].sum()
            if xi_sum > 0:
                xi[t] /= xi_sum

        # ── M-step ──
        # Startprob
        startprob = gamma[0] + 1e-10
        startprob /= startprob.sum()

        # Transmat
        xi_sum_t = xi.sum(axis=0)  # (n_states, n_states)
        row_sums = xi_sum_t.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transmat = xi_sum_t / row_sums

        # Means
        gamma_sum = gamma.sum(axis=0)  # (n_states,)
        for s in range(n_states):
            if gamma_sum[s] > 1e-10:
                means[s] = (gamma[:, s:s+1] * X).sum(axis=0) / gamma_sum[s]

        # Covariances
        for s in range(n_states):
            if gamma_sum[s] > 1e-10:
                diff = X - means[s]
                weighted = diff * gamma[:, s:s+1]
                covars[s] = (weighted.T @ diff) / gamma_sum[s] + np.eye(D) * 1e-6

    # ── Viterbi para sequencia MAP ──
    log_lik = _log_multivariate_normal_density(X, means, covars)
    log_trans = np.log(transmat + 1e-300)
    log_start = np.log(startprob + 1e-300)

    delta = np.zeros((T, n_states))
    psi = np.zeros((T, n_states), dtype=int)

    delta[0] = log_start + log_lik[0]
    for t in range(1, T):
        for j in range(n_states):
            scores = delta[t - 1] + log_trans[:, j]
            psi[t, j] = np.argmax(scores)
            delta[t, j] = scores[psi[t, j]] + log_lik[t, j]

    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    # Posteriors finais (gamma do ultimo E-step)
    log_lik_final = _log_multivariate_normal_density(X, means, covars)
    log_alpha_f, log_scale_f = _forward(log_lik_final, log_trans, log_start, n_states)
    log_beta_f = _backward(log_lik_final, log_trans, log_scale_f, n_states)
    log_gamma_f = log_alpha_f + log_beta_f
    log_gamma_f -= np.logaddexp.reduce(log_gamma_f, axis=1, keepdims=True)
    posteriors = np.exp(log_gamma_f)

    # Posteriors FILTRADOS: P(s_t | obs_1..t) — o alpha escalado do forward já
    # é a distribuição filtrada. É o que dá para saber em tempo real na barra t
    # (dado o modelo ajustado), sem olhar o futuro.
    filtered = np.exp(log_alpha_f)

    # BIC
    n_params = n_states * D + n_states * D * (D + 1) / 2 + n_states * (n_states - 1) + (n_states - 1)
    bic = -2 * np.sum(log_scale) + n_params * np.log(T)

    return {
        "states": states,
        "posteriors": posteriors,
        "filtered": filtered,
        "transmat": transmat,
        "means": means,
        "bic": float(bic),
        "n_states": n_states,
        "ll": float(np.sum(log_scale)),
    }


def _select_n_states(X: np.ndarray, max_states: int = 5) -> tuple:
    """Seleciona numero otimo de estados via BIC. Retorna (best_n, bic_list)."""
    best_bic = np.inf
    best_n = 2
    bic_list = []
    for n in range(2, max_states + 1):
        try:
            result = _fit_hmm(X, n)
            bic_val = float(result["bic"])
            bic_list.append({"n": n, "bic": round(bic_val, 2)})
            if bic_val < best_bic:
                best_bic = bic_val
                best_n = n
        except Exception:
            continue
    return best_n, bic_list


def _label_regimes(states, n_states, close_aligned):
    """
    Classifica estados usando direcao E volatilidade do preco.

    Para cada estado, calcula:
    - trend: retorno medio geometrico nos segmentos contiguos do estado
    - vol: desvio padrao dos retornos nas barras do estado

    Se os retornos medios sao estatisticamente similares (todos positivos ou
    todos negativos), diferencia por volatilidade (low-vol vs high-vol).
    """
    returns = close_aligned.pct_change().fillna(0)

    # Calcula trend por segmentos contiguos (nao barras esparsas)
    state_trends = {}
    state_vols = {}
    for s in range(n_states):
        mask = (states == s)
        rets_s = returns[mask]
        state_vols[s] = float(rets_s.std()) if len(rets_s) > 1 else 0.0

        # Retorno geometrico por segmento contiguo
        seg_rets = []
        in_seg = False
        seg_start = None
        for k in range(len(mask)):
            if mask[k] and not in_seg:
                in_seg = True
                seg_start = k
            elif (not mask[k] or k == len(mask) - 1) and in_seg:
                seg_end = k + 1 if mask[k] else k
                if seg_end - seg_start >= 2:
                    seg_close = close_aligned.iloc[seg_start:seg_end]
                    seg_ret = float(seg_close.iloc[-1] / seg_close.iloc[0] - 1)
                    seg_rets.append(seg_ret)
                in_seg = False

        state_trends[s] = float(np.mean(seg_rets)) if seg_rets else 0.0

    # Decide se os estados se diferenciam por direcao ou por volatilidade
    trends = [state_trends[s] for s in range(n_states)]
    vols = [state_vols[s] for s in range(n_states)]

    # Criterio: se todos os trends tem o mesmo sinal e a diferenca relativa
    # e menor que 50% do maior, entao a separacao e por volatilidade
    all_same_sign = all(t >= 0 for t in trends) or all(t <= 0 for t in trends)
    trend_range = max(trends) - min(trends)
    max_abs_trend = max(abs(t) for t in trends) if any(t != 0 for t in trends) else 1
    use_vol_labels = all_same_sign and (trend_range / max_abs_trend < 0.5) if max_abs_trend > 0 else False

    if use_vol_labels:
        # Separa por volatilidade: low-vol, mid-vol, high-vol
        sorted_by_vol = sorted(range(n_states), key=lambda s: state_vols[s])
        if n_states == 2:
            label_map = {sorted_by_vol[0]: "low_vol", sorted_by_vol[1]: "high_vol"}
        elif n_states == 3:
            label_map = {sorted_by_vol[0]: "low_vol", sorted_by_vol[1]: "mid_vol", sorted_by_vol[2]: "high_vol"}
        else:
            label_map = {}
            for i, s in enumerate(sorted_by_vol):
                if i < len(sorted_by_vol) // 3:
                    label_map[s] = "low_vol"
                elif i < 2 * len(sorted_by_vol) // 3:
                    label_map[s] = "mid_vol"
                else:
                    label_map[s] = "high_vol"
    else:
        # Separa por direcao: bear / sideways / bull
        sorted_by_trend = sorted(range(n_states), key=lambda s: state_trends[s])
        if n_states == 2:
            label_map = {sorted_by_trend[0]: "bear", sorted_by_trend[1]: "bull"}
        elif n_states == 3:
            label_map = {sorted_by_trend[0]: "bear", sorted_by_trend[1]: "sideways", sorted_by_trend[2]: "bull"}
        else:
            label_map = {}
            for i, s in enumerate(sorted_by_trend):
                if i < len(sorted_by_trend) // 3:
                    label_map[s] = "bear"
                elif i < 2 * len(sorted_by_trend) // 3:
                    label_map[s] = "sideways"
                else:
                    label_map[s] = "bull"

    labels = [label_map[s] for s in states]
    return labels, label_map


# Markov Switching via statsmodels
def _fit_markov_switching(X_series: pd.Series, n_states: int):
    """Fit Markov Switching Regression via statsmodels."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    model = MarkovRegression(
        X_series,
        k_regimes=n_states,
        trend="c",
        switching_variance=True,
    )
    result = model.fit(maxiter=200, disp=False)
    return result


# Change-Point Detection (Binary Segmentation L2)
def _detect_changepoints(X: np.ndarray, close_aligned, min_segment: int = 20):
    """Change-point detection usando Binary Segmentation com custo L2."""
    T = len(X)
    if T < min_segment * 2:
        return np.zeros(T, dtype=int), ["sideways"] * T

    # Binary segmentation recursiva
    breakpoints = []
    _binary_segmentation(X, 0, T, min_segment, breakpoints, max_bkps=8)
    breakpoints = sorted(set(breakpoints))

    # Cria segmentos
    boundaries = [0] + breakpoints + [T]
    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        seg_close = close_aligned.iloc[start:end]
        seg_ret = seg_close.pct_change().mean() if len(seg_close) > 1 else 0
        segments.append({"start": start, "end": end, "mean_return": float(seg_ret)})

    # Rotula por retorno medio
    rets = [s["mean_return"] for s in segments]
    if not rets:
        return np.zeros(T, dtype=int), ["sideways"] * T

    q33 = np.percentile(rets, 33)
    q66 = np.percentile(rets, 66)

    labels = ["sideways"] * T
    for seg in segments:
        r = seg["mean_return"]
        # percentis relativos + sinal: 'bear' exige retorno negativo e 'bull'
        # positivo — sem isso, serie de segmento unico (q33=q66=r) caia sempre
        # em bear, mesmo num uptrend puro
        if r <= q33 and r < 0:
            l = "bear"
        elif r >= q66 and r > 0:
            l = "bull"
        else:
            l = "sideways"
        for i in range(seg["start"], min(seg["end"], T)):
            labels[i] = l

    # Remapeia para state IDs contiguos (0, 1, ..., n-1)
    unique_labels = sorted(set(labels))
    label_to_id = {lb: idx for idx, lb in enumerate(unique_labels)}
    states = np.array([label_to_id[lb] for lb in labels], dtype=int)

    return states, labels


def _binary_segmentation(X, start, end, min_size, breakpoints, max_bkps=8):
    """Binary segmentation recursiva com custo L2."""
    if end - start < 2 * min_size or len(breakpoints) >= max_bkps:
        return

    seg = X[start:end]
    total_cost = np.sum((seg - seg.mean(axis=0)) ** 2)

    best_gain = 0
    best_bp = None
    for bp in range(start + min_size, end - min_size):
        left = X[start:bp]
        right = X[bp:end]
        cost_left = np.sum((left - left.mean(axis=0)) ** 2)
        cost_right = np.sum((right - right.mean(axis=0)) ** 2)
        gain = total_cost - cost_left - cost_right
        if gain > best_gain:
            best_gain = gain
            best_bp = bp

    threshold = total_cost * 0.01
    if best_bp is not None and best_gain > threshold:
        breakpoints.append(best_bp)
        _binary_segmentation(X, start, best_bp, min_size, breakpoints, max_bkps)
        _binary_segmentation(X, best_bp, end, min_size, breakpoints, max_bkps)


# Metrics by regime
def _infer_bars_per_year(index: pd.DatetimeIndex) -> float:
    """Infere quantas barras existem por ano a partir dos dados reais."""
    if len(index) < 2:
        return 252.0
    total_days = max((index[-1] - index[0]).days, 1)
    n_bars = len(index)
    return n_bars / (total_days / 365.25)


def _metrics_by_regime(close_aligned, labels):
    """Calcula metricas condicionais por regime."""
    df = pd.DataFrame({"close": close_aligned.values, "label": labels}, index=close_aligned.index)
    df["return"] = df["close"].pct_change()

    bars_per_year = _infer_bars_per_year(close_aligned.index)

    results = {}
    for regime in sorted(set(labels)):
        mask = df["label"] == regime
        rets = df.loc[mask, "return"].dropna()
        if len(rets) < 2:
            results[regime] = {
                "count_bars": int(mask.sum()),
                "pct_time": round(mask.sum() / len(df) * 100, 1),
                "total_return": 0,
                "annualized_return": 0,
                "sharpe": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "volatility": 0,
            }
            continue

        total_ret = float((1 + rets).prod() - 1)

        # Annualized return: anualiza pela EXPOSICAO real (barras no regime),
        # consistente com vol/Sharpe. O span de calendario do regime inclui os
        # buracos em que outros regimes estavam ativos e distorcia forte
        # (regime fragmentado tinha o anualizado subestimado em ate ~3x).
        ann_ret = (1 + total_ret) ** (bars_per_year / len(rets)) - 1

        # Volatilidade e Sharpe: anualiza por barras/ano (consistente)
        vol = float(rets.std(ddof=1) * np.sqrt(bars_per_year))
        sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(bars_per_year)) if rets.std(ddof=1) > 0 else 0.0

        # Max Drawdown: por segmentos contiguos do regime
        max_dd = 0.0
        mask_arr = mask.values
        seg_start = None
        for k in range(len(mask_arr)):
            if mask_arr[k] and seg_start is None:
                seg_start = k
            elif (not mask_arr[k] or k == len(mask_arr) - 1) and seg_start is not None:
                seg_end = k + 1 if mask_arr[k] else k
                seg_close = df["close"].iloc[seg_start:seg_end].values
                if len(seg_close) > 1:
                    seg_peak = np.maximum.accumulate(seg_close)
                    seg_dd = (seg_close - seg_peak) / np.where(seg_peak != 0, seg_peak, 1.0)
                    seg_min_dd = float(seg_dd.min()) * 100
                    if seg_min_dd < max_dd:
                        max_dd = seg_min_dd
                seg_start = None

        win_rate = float((rets > 0).sum() / len(rets) * 100)

        results[regime] = {
            "count_bars": int(mask.sum()),
            "pct_time": round(mask.sum() / len(df) * 100, 1),
            "total_return": round(total_ret * 100, 2),
            "annualized_return": round(ann_ret * 100, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd, 2),
            "win_rate": round(win_rate, 1),
            "volatility": round(vol * 100, 2),
        }

    return results


# Transition matrix
def _transition_matrix(states, n_states):
    """Calcula matriz de transicao empirica (state x state)."""
    mat = np.zeros((n_states, n_states))
    for i in range(1, len(states)):
        mat[states[i - 1], states[i]] += 1

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat = mat / row_sums
    return np.round(mat * 100, 1).tolist()


# Main entry point
def detect_regimes(
    df: pd.DataFrame,
    method: str = "hmm",
    n_states: int = 0,
    features: list = None,
    vol_window: int = 20,
    causal: bool = False,
) -> dict:
    """
    Detecta regimes de mercado.

    Parameters
    ----------
    df : DataFrame com colunas OHLCV e DatetimeIndex
    method : "hmm", "markov_switching" ou "changepoint"
    n_states : numero de estados (0 = auto via BIC, so para HMM/MS)
    features : lista de features ["log_return", "volatility", "volume"]
    vol_window : janela para calculo de volatilidade realizada
    causal : True = probabilidades FILTRADAS (P(s_t | obs ate t) — o que dá
        para saber em tempo real na barra t, sem olhar o futuro). False =
        suavizadas com a série inteira (melhor descrição histórica, mas o
        rótulo de uma barra passada usa informação futura). Os parâmetros do
        modelo continuam ajustados na série inteira nos dois casos.
        Ignorado no changepoint (retrospectivo por construção).

    Returns
    -------
    dict com: dates, prices, regimes, regime_probs, metrics_by_regime,
              transition_matrix, state_names, method, n_states, bic_scores,
              causal
    """
    if features is None:
        features = ["log_return", "volatility"]

    feat_df = _compute_features(df, features, vol_window)
    X = feat_df.values

    close = df["Close"] if "Close" in df.columns else df["close"]
    close_aligned = close.loc[feat_df.index]

    bic_scores = None

    if method == "markov_switching":
        log_ret_series = feat_df["log_return"] if "log_return" in feat_df.columns else feat_df.iloc[:, 0]

        if n_states <= 0:
            bic_list = []
            best_bic = np.inf
            best_n = 2
            for n in range(2, 5):
                try:
                    res = _fit_markov_switching(log_ret_series, n)
                    bic_val = float(res.bic)
                    bic_list.append({"n": n, "bic": round(bic_val, 2)})
                    if bic_val < best_bic:
                        best_bic = bic_val
                        best_n = n
                except Exception:
                    pass
            n_states = best_n
            bic_scores = bic_list

        ms_result = _fit_markov_switching(log_ret_series, n_states)

        probs = (ms_result.filtered_marginal_probabilities if causal
                 else ms_result.smoothed_marginal_probabilities)
        states = np.array(probs.values.argmax(axis=1), dtype=int)
        posteriors = probs.values

        labels, label_map = _label_regimes(states, n_states, close_aligned)

        state_names_ordered = []
        for s in range(n_states):
            state_names_ordered.append(label_map.get(s, f"state_{s}"))

        transition = _transition_matrix(states, n_states)

        regime_probs = {
            state_names_ordered[s]: [round(float(p), 4) for p in posteriors[:, s]]
            for s in range(n_states)
        }

    elif method == "hmm":
        if n_states <= 0:
            n_states, bic_scores = _select_n_states(X, max_states=5)

        hmm_result = _fit_hmm(X, n_states)
        if causal:
            # filtrado: estado da barra t decidido só com obs até t
            posteriors = hmm_result["filtered"]
            states = np.asarray(posteriors.argmax(axis=1), dtype=int)
        else:
            states = hmm_result["states"]          # Viterbi (série inteira)
            posteriors = hmm_result["posteriors"]  # suavizado

        labels, label_map = _label_regimes(states, n_states, close_aligned)

        state_names_ordered = []
        for s in range(n_states):
            state_names_ordered.append(label_map.get(s, f"state_{s}"))

        transition = _transition_matrix(states, n_states)

        regime_probs = {
            state_names_ordered[s]: [round(float(p), 4) for p in posteriors[:, s]]
            for s in range(n_states)
        }

    else:
        # changepoint
        states, labels = _detect_changepoints(X, close_aligned)
        unique_labels = sorted(set(labels))
        n_states = len(unique_labels)
        state_names_ordered = unique_labels
        transition = _transition_matrix(states, n_states)
        regime_probs = None

    # Build output
    dates = [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in feat_df.index]
    prices = [round(float(v), 6) for v in close_aligned.values]

    metrics = _metrics_by_regime(close_aligned, labels)

    return {
        "dates": dates,
        "prices": prices,
        "regimes": labels,
        "regime_probs": regime_probs,
        "metrics_by_regime": metrics,
        "transition_matrix": transition,
        "state_names": state_names_ordered,
        "method": method,
        "n_states": n_states,
        "bic_scores": bic_scores,
        "causal": bool(causal) if method != "changepoint" else None,
    }
