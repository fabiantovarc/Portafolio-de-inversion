import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


# ----------
# Metricas
# ----------

@dataclass
class PortfolioMetrics:
    ann_return: float   # Retorno esperado anual del portafolio
    ann_vol: float      # Volatilidad anual del portafolio
    sharpe: float       # Sharpe anual
    cv: float           # Coeficiente de variación anual


def annualize_stats(
    rets: pd.DataFrame,
    weights: np.ndarray,
    rf: float = 0.0,
    periods_per_year: int = 252,
    kind: str = "log",
) -> PortfolioMetrics:
    """
    Calcula retorno y volatilidad anualizados, ratio de Sharpe y CV.

    Parámetros
    ----------
    rets : DataFrame
        Retornos por periodo (por defecto diarios). Pueden ser simples o log;
        por ahora `kind` se usa solo como etiqueta (la fórmula es la misma).
    weights : array-like
        Vector de pesos del portafolio (se asume que suma 1 y coincide en orden
        con las columnas de `rets`).
    rf : float
        Tasa libre de riesgo ANUAL (por ejemplo 0.05 = 5%).
    periods_per_year : int
        Número de periodos por año (252 si son retornos diarios).
    kind : {"log","simple"}
        Se mantiene por compatibilidad con el resto del código; por ahora
        no altera la fórmula.
    """
    rets = rets.dropna(how="any")
    w = np.array(weights, dtype=float).reshape(-1, 1)

    if rets.shape[1] != w.shape[0]:
        raise ValueError("Dimensión de pesos y columnas de retornos no coincide.")

    # Estadísticos diarios
    mu_daily = rets.mean().values.reshape(-1, 1)
    cov_daily = rets.cov().values

    # Anualización (como en TESIFINAL)
    mu_vec = mu_daily * periods_per_year
    cov_mat = cov_daily * periods_per_year

    # Retorno y volatilidad del portafolio
    port_mu = float((w.T @ mu_vec)[0, 0])
    port_sigma = float(np.sqrt(w.T @ cov_mat @ w)[0, 0])

    # Sharpe anual
    sharpe = (port_mu - rf) / port_sigma if port_sigma > 0 else np.nan

    # Coeficiente de variación
    cv = port_sigma / port_mu if abs(port_mu) > 1e-12 else np.nan

    return PortfolioMetrics(
        ann_return=port_mu,
        ann_vol=port_sigma,
        sharpe=sharpe,
        cv=cv
    )

