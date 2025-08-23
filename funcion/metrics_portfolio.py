#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:50:47 2025

@author: camilocastillo
"""

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
    ann_return: float
    ann_vol: float
    sharpe: float
    cv: float


def annualize_stats(rets: pd.DataFrame, weights: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> PortfolioMetrics:
    """
    Calcula retorno y volatilidad anualizados, ratio de Sharpe y CV.
    - rets: DataFrame de retornos (diarios por defecto).
    - weights: vector de pesos (suma 1).
    - rf: tasa libre de riesgo anual (p.ej., 0.05 = 5%).
    """
    w = np.array(weights).reshape(-1, 1)
    mu_vec = rets.mean().values.reshape(-1, 1) * periods_per_year                # retorno esperado anual por activo
    cov_mat = rets.cov().values * periods_per_year                               # covarianza anual
    port_mu = float((w.T @ mu_vec)[0, 0])                                        # retorno anual esperado
    port_sigma = float(np.sqrt(w.T @ cov_mat @ w)[0, 0])                          # vol anual
    # Sharpe anual usando retorno en exceso
    sharpe = (port_mu - rf) / port_sigma if port_sigma > 0 else np.nan
    # CV anual = vol / retorno (precaución si retorno ~ 0)
    cv = port_sigma / port_mu if abs(port_mu) > 1e-12 else np.nan

    return PortfolioMetrics(ann_return=port_mu, ann_vol=port_sigma, sharpe=sharpe, cv=cv)
