#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:53:34 2025

@author: camilocastillo
"""

import numpy as np
import pandas as pd
import yahoo_data
from typing import List, Tuple, Dict, Optional
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import plot
import metrics_portfolio

class Portfolio:
    "Partiendo de un df con los precios y un portafolio construye el valor de portafolio y las metricas"
    def __init__(self, tickers:List[str], weights: np.ndarray, initial: str = '2024-10-01', end: Optional[str]=None, initial_capital: float = 100_000.0):
        self.tickers = tickers
        self.initial = initial
        self.end = end
        self._info = yahoo_data.yahoo_data(tickers = self.tickers, start = self.initial, end = self.end)
        self.precios = self._info.data
        self.log_rets = self._info.compute_returns()
        self.rets = self._info.compute_returns('simple')
        self.weights = weights
        self.V0 = initial_capital
        self.Value_portfolio = self._portfolio_value()
        self.metrics_port = None
        
    def _portfolio_value(self) -> pd.DataFrame:
        """
        Construye un DataFrame con la serie temporal del valor del portafolio.
        Usa pesos fijos y buy&hold sin rebalanceo continuo (rebalanceo solo al inicio).
        """
        w = np.array(self.weights)
        if not np.isclose(w.sum(), 1.0):
            raise ValueError("Los pesos deben sumar 1.")
            # Asignar capital inicial por activo según pesos
        else:
            alloc = self.V0 * w
            # Número de acciones por activo al inicio
            first_prices = self.precios.iloc[0].values
            shares = alloc / first_prices
            # Valor en el tiempo
            port_val = self.precios.values @ shares
            port_val = pd.DataFrame(port_val, index=self.precios.index, columns=["PortfolioValue"])
        print("Valor del portafolio creado")
        return port_val
        
    def plot_value_portfolio(self):
        "Construye la grafica del valor del portafolio"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = self.Value_portfolio.index,
                                 y = self.Value_portfolio['PortfolioValue'],
                                  mode = 'lines',
                                  name = 'Total Position',
                                  line = dict(color = 'black')))
        fig.update_layout(xaxis_title = 'Time', yaxis_title = 'Total Position',
                          title = 'Total Position')
        plot(fig, auto_open = True)
        
    def metrics(self, rf: float = 0.0, periods_per_year: int = 252, ret:bool = True):
        aux = metrics_portfolio.annualize_stats(self.log_rets,
                                                self.weights,
                                                rf = rf,
                                                periods_per_year= periods_per_year)
        self.metrics_port = aux
        print("***Metrics***")
        if ret == True:
            return aux
    
    def __str__(self):
        return (
            f'Annual return: {self.metrics_port.ann_return}\n'
            f'Annual Volatility: {self.metrics_port.ann_vol}\n'
            f'Sharpe (annual): {self.metrics_port.sharpe}\n'
            f'Variation Coef. : {self.metrics_port.cv}')
        

#aux = Portfolio(['AAPL','KO','CX'], np.array([0.5,0.2,0.3]))

#aux.plot_value_portfolio()

#aux.metrics(ret=False)
#print(aux)




