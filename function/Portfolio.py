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
    def __init__(self,
                 tickers: List[str],
                 weights: np.ndarray,
                 initial: str = '2024-10-01',
                 end: Optional[str] = None,
                 initial_capital: float = 100_000.0):
        self.tickers = tickers
        self.initial = initial
        self.end = end

        # Descarga de datos
        self._info = yahoo_data.yahoo_data(
            tickers=self.tickers,
            start=self.initial,
            end=self.end
        )
        self.precios = self._info.data

        # Retornos de los ACTIVOS
        self.log_rets = self._info.compute_returns(kind="log")
        self.rets = self._info.compute_returns(kind="simple")

        # Validación de pesos
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.shape[0] != len(self.tickers):
            raise ValueError("La longitud de weights debe coincidir con el número de tickers.")
        if not np.isclose(w.sum(), 1.0):
            raise ValueError("Los pesos deben sumar 1.")
        self.weights = w

        # Capital inicial y valor del portafolio buy&hold
        self.V0 = initial_capital
        self.Value_portfolio = self._portfolio_value()
        self.metrics_port = None

        
    def _portfolio_value(self) -> pd.DataFrame:
        """
        Construye un DataFrame con la serie temporal del valor del portafolio.
        Usa pesos fijos y buy&hold sin rebalanceo continuo (rebalanceo solo al inicio).
        """
        w = self.weights
        # Asignar capital inicial por activo según pesos
        alloc = self.V0 * w
        # Número de acciones por activo al inicio
        first_prices = self.precios.iloc[0].values
        shares = alloc / first_prices
        # Valor en el tiempo (buy&hold)
        port_val = self.precios.values @ shares
        port_val = pd.DataFrame(port_val,
                                index=self.precios.index,
                                columns=["PortfolioValue"])
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
        
    def metrics(self,
                rf: float = 0.0,
                periods_per_year: int = 252,
                ret: bool = True):
        """
        Calcula métricas SOBRE EL PORTAFOLIO buy&hold que está en self.Value_portfolio.
        - rf: tasa libre de riesgo anual (en proporción, ej. 0.05 = 5%).
        """
        # Log-retornos del PORTAFOLIO (no de los activos)
        port_log_rets = np.log(
            self.Value_portfolio / self.Value_portfolio.shift(1)
        ).dropna()

        # Llamamos a annualize_stats como si fuera un solo activo con peso 1.0
        aux = metrics_portfolio.annualize_stats(
            port_log_rets,                 # DataFrame de 1 columna
            np.array([1.0]),               # peso 100% en ese "activo portafolio"
            rf=rf,
            periods_per_year=periods_per_year
        )
        self.metrics_port = aux
        print("***Metrics***")
        if ret:
            return aux

    
    def __str__(self):
        return (
            f'Annual return: {self.metrics_port.ann_return}\n'
            f'Annual Volatility: {self.metrics_port.ann_vol}\n'
            f'Sharpe (annual): {self.metrics_port.sharpe}\n'
            f'Variation Coef. : {self.metrics_port.cv}'
        )


        

#aux = Portfolio(['AAPL','KO','CX'], np.array([0.5,0.2,0.3]))

#aux.plot_value_portfolio()

#aux.metrics(ret=False)
#print(aux)




