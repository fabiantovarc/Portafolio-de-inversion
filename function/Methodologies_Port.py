import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple, Dict, Optional
import yahoo_data
import metrics_portfolio
import pypfopt
from pypfopt import expected_returns
from pypfopt.efficient_frontier.efficient_cdar import EfficientFrontier
from pypfopt import plotting
import cvxpy as cp


try:
    from scipy.optimize import minimize
except ImportError as e:
    raise ImportError(
        "Falta SciPy para resolver Markowitz con SLSQP. "
        "Instala con: pip install scipy"
    ) from e

class Do_portfolio:
    def __init__(self, tickers:List[str], start: str = '2024-01-01', end: Optional[str] = None):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.weights = [0]*len(self.tickers)
        self.info = yahoo_data.yahoo_data(tickers=self.tickers,
                                          start=self.start,
                                          end = self.end)
        self.bar_mu = None
        self.bar_var = None
        

    def random_weights(self,n_port:int, short: bool = True, seed: Optional[int] = None) -> np.ndarray:
        "Genera pesos aleatorios que suman 1"
        rng = np.random.default_rng(seed)
        if not short:
            W = rng.dirichlet(np.ones(len(self.tickers)), size = n_port)
        else:
            X = rng.normal(size = (n_port,len(self.tickers)))
            W = X/np.sum(np.abs(X), axis = 1, keepdims= True)
        return W
    
    def Monte_Carlo(self, n_port: int, short: bool = True, seed: Optional[int] = None, rf: float = 0.01, periods_per_year: int = 252):
        "Construye un portafolio por Monte Carlos con el Max. Sharpe"
        W = self.random_weights(n_port,short = short,seed = seed)
        results = np.zeros((n_port,4))
        for i in range(n_port):
            m = metrics_portfolio.annualize_stats(self.info.compute_returns(), W[i], rf= rf, periods_per_year=periods_per_year)
            results[i,0] = m.ann_return
            results[i,1] = m.ann_vol
            results[i,2] = m.sharpe
            results[i,3] = m.cv
        
        df = pd.DataFrame(results, columns = ['ann_returns', 'ann_vol', 'Sharpe', 'CV'])
        
        for j, t in enumerate(self.tickers):
            df[f'w_{t}'] = W[:,j]
        return df
    
    def pick_candidates(self, n_port: int, short: bool = True, rf: float= 0.01, seed: Optional[int]= None, periods_per_year: int = 252) -> Dict[str, pd.Series]:
        "Retorna el portafolios candidatos: max_sharpe, min vol"
        aux = self.Monte_Carlo(n_port,short = short, seed = seed, rf = rf, periods_per_year= periods_per_year)
        idx_sharpe = aux['Sharpe'].idxmax()
        idx_minvol = aux['ann_vol'].idxmin()
        return {
            'max Sharpe': aux.loc[idx_sharpe],
            'min Vol': aux.loc[idx_minvol]
            }
    
    def _CAPM(self, rf:float = 0.01,market: str = '^GSPC', periods_per_year: int = 252):
        "Halla el retorno esperado por CAPM"
        x = expected_returns.capm_return(
            self.info.compute_returns(),
            market_prices = yahoo_data.yahoo_data(ticker = [market],
                                                  start = self.start,
                                                  end = self.end).compute_returns(),
            returns_data=True,
            risk_free_rate = rf,
            compounding=False,
            frequency= periods_per_year,
            log_returns= True
            )
        print('Retornos generados por CAPM')
        return x
    
    def _hist_mean(self):
        x = self.info.compute_returns().mean()
        return x
        
    def mean_estimation(self, tipo: str = 'hist'):
        if tipo == 'hist':
            self.bar_mu = self._hist_mean()
        elif tipo == 'CAPM':
            self.bar_mu = self._CAPM()
        else:
            ValueError('No es una metodologpia válida de estimación')
        return self.bar_mu
    
    def _hist_var(self, periods_per_year: int = 252):
        sig2 = self.info.compute_returns().cov()*periods_per_year
        return sig2
    
    def var_estimation(self, tipo: str = 'hist'):
        if tipo == 'hist':
            self.bar_var = self._hist_var()
        elif tipo == 'ARMA':
            self.bar_var = self._ARMA()
        else:
            ValueError('No es una metodología válidad de estimación')
        return self.bar_var
    
    ## def _ARMA(self, p, q): construir
    
    def Markowitz(self, short: bool = True, portfolio: str = 'min_vol', rf: float = 0.001):
        if short:
            frontera_efici = EfficientFrontier(self.bar_mu.values, self.bar_var.values, weight_bounds=(-1,1))
            if portfolio == 'min_vol':
                self.weights = list(frontera_efici.min_volatility().values())
            elif portfolio == 'max_sharpe':
                self.weights = list(frontera_efici.max_sharpe(risk_free_rate=rf).values())
        else:
            frontera_efici = EfficientFrontier(self.bar_mu, self.bar_var, weight_bounds=(-1,1))
        return self.weights

    @staticmethod
    def min_var_portfolio(mu: pd.Series,
                          Sigma: pd.DataFrame,
                          target_return: float,
                          long_only: bool = True,
                          solver: Optional[str] = None) -> Dict[str, object]:
        """
        2.2 Mínima varianza con retorno objetivo (Markowitz), long-only por defecto.

        Parámetros
        ----------
        mu : pd.Series (anual)
            Vector de retornos esperados anualizados por activo.
        Sigma : pd.DataFrame (anual)
            Matriz de covarianzas anualizada.
        target_return : float (anual)
            Retorno objetivo µ* del portafolio (en las mismas unidades de mu).
        long_only : bool
            Si True, impone w >= 0.
        solver : str | None
            Forzar solver de cvxpy (p.ej. "OSQP" o "SCS"). Si None, se elige uno disponible.

        Retorna
        -------
        dict con:
            - 'weights' : pd.Series (pesos óptimos, suma=1)
            - 'expected_return' : float (µ del portafolio)
            - 'variance' : float (w'Σw)
            - 'volatility' : float (sqrt(w'Σw))
            - 'status' : str (estado del solver)
        """
        # Alinear y pasar a numpy de forma segura
        tickers = list(mu.index)
        Sigma = Sigma.loc[tickers, tickers]  # asegura el mismo orden
        mu_vec = mu.values.astype(float)
        Sigma_mat = Sigma.values.astype(float)

        n = len(mu_vec)
        w = cp.Variable(n)

        constraints = [cp.sum(w) == 1, mu_vec @ w >= float(target_return)]
        if long_only:
            constraints.append(w >= 0)

        objective = cp.Minimize(cp.quad_form(w, Sigma_mat))
        problem = cp.Problem(objective, constraints)

        # Elección de solver (fallback)
        if solver is None:
            installed = set(cp.installed_solvers())
            solver = 'OSQP' if 'OSQP' in installed else 'SCS'

        problem.solve(solver=solver, verbose=False)

        w_val = np.array(w.value).reshape(-1)
        weights = pd.Series(w_val, index=tickers)

        port_mu = float(weights @ mu_vec)
        port_var = float(weights.values @ Sigma_mat @ weights.values)
        port_vol = float(np.sqrt(port_var)) if port_var >= 0 else np.nan

        return {
            "weights": weights,
            "expected_return": port_mu,
            "variance": port_var,
            "volatility": port_vol,
            "status": problem.status,
        }

    def min_var_target(self,
                       target_return: float,
                       mu: Optional[pd.Series] = None,
                       Sigma: Optional[pd.DataFrame] = None,
                       long_only: bool = True,
                       kind: str = "log",
                       periods_per_year: int = 252,
                       solver: Optional[str] = None) -> Dict[str, object]:
        """
        Wrapper de instancia: si no se pasan mu/Sigma, los calcula con los datos
        que la clase ya maneja internamente (reutiliza tu pipeline actual).

        target_return : float (anual)
            Retorno objetivo µ*.
        kind : {'log','simple'}
            Convención de retornos diarios para estimar mu/Sigma si no se pasan.
        """
        # Si no nos pasan mu/Sigma, estimamos desde self.info (tu clase ya lo hace)
        if (mu is None) or (Sigma is None):
            # self.info debe exponer precios/retornos; reusamos tu convención
            rets = self.info.compute_returns(kind=kind)      # diarios
            mu = rets.mean() * periods_per_year              # anual
            Sigma = rets.cov() * periods_per_year            # anual

        return self.min_var_portfolio(mu=mu,
                                      Sigma=Sigma,
                                      target_return=target_return,
                                      long_only=long_only,
                                      solver=solver)

    @staticmethod
    def min_variance_weights(Sigma: pd.DataFrame,
                             long_only: bool = True,
                             solver: str | None = None) -> pd.Series:
        """
        PMV (mínima varianza sin objetivo de retorno): min w'Σw s.a. 1'w=1 (y w>=0 si long_only).
        Devuelve una Serie con los pesos (index=tickers).
        """
        tickers = list(Sigma.columns)
        Sigma = Sigma.loc[tickers, tickers].values.astype(float)
        n = len(tickers)
        w = cp.Variable(n)

        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)

        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)

        if solver is None:
            installed = set(cp.installed_solvers())
            solver = 'OSQP' if 'OSQP' in installed else 'SCS'
        prob.solve(solver=solver, verbose=False)

        w_val = np.array(w.value).reshape(-1)
        return pd.Series(w_val, index=tickers)

    @staticmethod
    def efficient_frontier_by_return(mu: pd.Series,
                                     Sigma: pd.DataFrame,
                                     n_points: int = 25,
                                     mu_targets: np.ndarray | None = None,
                                     long_only: bool = True,
                                     solver: str | None = None) -> dict:
        """
        2.3 Frontera eficiente: barre retornos objetivo y resuelve 2.2 para cada punto.

        Parámetros
        ----------
        mu, Sigma : anualizados (como en 2.1)
        n_points  : número de puntos en la curva (si no se provee mu_targets)
        mu_targets: array explícito de retornos-objetivo (anual). Si None, se genera.
        long_only : restringe w>=0 si True
        solver    : solver cvxpy ('OSQP' o 'SCS'); auto si None

        Retorna
        -------
        dict con:
          - 'curve'     : DataFrame con columnas ['target_return','volatility','variance']
          - 'weights'   : DataFrame (n_points x n_assets) con pesos por punto
          - 'pmv'       : dict con info del PMV (pesos, mu, vol)
          - 'statuses'  : lista de estados del solver por punto
        """
        # Orden y tipos
        tickers = list(mu.index)
        mu = mu.loc[tickers].astype(float)
        Sigma = Sigma.loc[tickers, tickers].astype(float)

        # 1) PMV (sin restricción de retorno)
        w_pmv = Do_portfolio.min_variance_weights(Sigma, long_only=long_only, solver=solver)
        mu_pmv = float(mu @ w_pmv)
        var_pmv = float(w_pmv.values @ Sigma.values @ w_pmv.values)
        vol_pmv = float(np.sqrt(max(var_pmv, 0.0)))

        # 2) Construir grid de retornos objetivo
        mu_lo_auto = mu_pmv + 1e-8               # desde un pelo por encima del PMV
        mu_hi_auto = float(mu.max()) - 1e-8      # hasta casi el máximo individual
        if mu_targets is None:
            if mu_lo_auto > mu_hi_auto:          # salvaguarda si todo es igual
                mu_lo_auto, mu_hi_auto = float(mu.min()), float(mu.max())
            mu_targets = np.linspace(mu_lo_auto, mu_hi_auto, n_points)

        # 3) Barrido usando el solver de 2.2
        vols, vars_, rets, statuses = [], [], [], []
        W = []  # lista de Series con pesos

        for mu_star in mu_targets:
            res = Do_portfolio.min_var_portfolio(mu=mu, Sigma=Sigma,
                                                 target_return=float(mu_star),
                                                 long_only=long_only, solver=solver)
            statuses.append(res["status"])
            rets.append(res["expected_return"])
            vols.append(res["volatility"])
            vars_.append(res["variance"])
            W.append(res["weights"].reindex(tickers))

        # 4) Empaquetar resultados
        curve = pd.DataFrame({
            "target_return": np.array(mu_targets, dtype=float),
            "expected_return": np.array(rets, dtype=float),
            "volatility": np.array(vols, dtype=float),
            "variance": np.array(vars_, dtype=float),
        })

        weights_df = pd.DataFrame(W, index=range(len(W)))  # filas = puntos
        pmv_info = {"weights": w_pmv, "expected_return": mu_pmv,
                    "volatility": vol_pmv, "variance": var_pmv}

        return {
            "curve": curve,
            "weights": weights_df,   # columnas = tickers
            "pmv": pmv_info,
            "statuses": statuses,
        }

    def efficient_frontier(self,
                           n_points: int = 25,
                           mu_targets: np.ndarray | None = None,
                           mu: pd.Series | None = None,
                           Sigma: pd.DataFrame | None = None,
                           long_only: bool = True,
                           kind: str = "log",
                           periods_per_year: int = 252,
                           solver: str | None = None) -> dict:
        """
        Versión de instancia: si no pasas mu/Sigma, se estiman desde los datos que
        maneja la clase (reutiliza tu pipeline).
        """
        if (mu is None) or (Sigma is None):
            rets = self.info.compute_returns(kind=kind)   # diarios
            mu = rets.mean() * periods_per_year
            Sigma = rets.cov() * periods_per_year

        return Do_portfolio.efficient_frontier_by_return(mu=mu,
                                                         Sigma=Sigma,
                                                         n_points=n_points,
                                                         mu_targets=mu_targets,
                                                         long_only=long_only,
                                                         solver=solver)



### Example ###
tickers = ['AAPL','CX','KO']
porta = Do_portfolio(tickers)
w_MC = porta.pick_candidates(n_port = 1000)
#for i,j in zip(w['max Sharpe'].iloc[-1:-4:-1],tickers):
#    print(f'{j}: {i}')


### Markowitz 
porta.mean_estimation()
porta.var_estimation()

w_Marko = porta.Markowitz()
    
    
    
    