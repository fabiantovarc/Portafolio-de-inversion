import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple, Dict, Optional, Union,Any
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
        

    def random_weights(self, n_port: int, short: bool = True, seed: Optional[int] = None) -> np.ndarray:
        """
        Genera pesos aleatorios:
        - short=False: sin cortos, suman 1 y son >=0
        - short=True: con cortos, suman 1 pero pueden ser negativos
        """
        rng = np.random.default_rng(seed)
        n_assets = len(self.tickers)

        if not short:
            W = rng.dirichlet(np.ones(n_assets), size=n_port)
        else:
            X = rng.normal(size=(n_port, n_assets))
            # Normalizar para que la suma de pesos sea 1 (no la suma de absolutos)
            W = X / X.sum(axis=1, keepdims=True)
        return W


    
    def Monte_Carlo(
        self,
        n_port: int,
        short: bool = True,
        seed: Optional[int] = None,
        rf: float = 0.01,
        periods_per_year: int = 252
    ) -> pd.DataFrame:
        "Construye un conjunto de portafolios por Monte Carlo y calcula métricas."
        # Retornos (por defecto, log) -> los calculamos UNA sola vez
        rets = self.info.compute_returns(kind="log")

        # Pesos aleatorios
        W = self.random_weights(n_port, short=short, seed=seed)

        results = np.zeros((n_port, 4))
        for i in range(n_port):
            m = metrics_portfolio.annualize_stats(
                rets,
                W[i],
                rf=rf,
                periods_per_year=periods_per_year,
                kind="log"        # importante: consistente con compute_returns
            )
            results[i, 0] = m.ann_return
            results[i, 1] = m.ann_vol
            results[i, 2] = m.sharpe
            results[i, 3] = m.cv

        df = pd.DataFrame(results, columns=['ann_returns', 'ann_vol', 'Sharpe', 'CV'])

        for j, t in enumerate(self.tickers):
            df[f'w_{t}'] = W[:, j]

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
    
    def _CAPM(
        self,
        rf: float = 0.01,
        market: str = '^GSPC',
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Halla el retorno esperado por CAPM usando PyPortfolioOpt.

        Usa precios (Adj Close) de los activos y del índice de mercado.
        """
        # Precios de los activos (de la instancia principal)
        prices_assets = self.info.data

        # Precios del mercado
        market_info = yahoo_data.yahoo_data(
            tickers=[market],
            start=self.start,
            end=self.end
        )
        market_prices = market_info.data

        x = expected_returns.capm_return(
            prices_assets,
            market_prices=market_prices,
            returns_data=False,       # le pasamos precios, no retornos
            risk_free_rate=rf,        # tasa libre de riesgo anual
            compounding=False,        # media aritmética (recomendable para CAPM)
            frequency=periods_per_year,
            log_returns=False         # que compute retornos simples internamente
        )
        print('Retornos generados por CAPM')
        return x


    
    def _hist_mean(self, periods_per_year: int = 252) -> pd.Series:
        """
        Estima retornos esperados anuales a partir de retornos simples diarios (media histórica).
        """
        rets_simple = self.info.compute_returns(kind="simple")
        mu_daily = rets_simple.mean()
        mu_ann = mu_daily * periods_per_year
        return mu_ann

        
    def mean_estimation(self, tipo: str = 'hist', periods_per_year: int = 252):
        if tipo == 'hist':
            self.bar_mu = self._hist_mean(periods_per_year=periods_per_year)
        elif tipo == 'CAPM':
            self.bar_mu = self._CAPM()
        else:
            raise ValueError('No es una metodología válida de estimación')
        return self.bar_mu

    
    def _hist_var(self, periods_per_year: int = 252) -> pd.DataFrame:
        """
        Matriz de covarianza anual basada en retornos simples diarios.
        """
        rets_simple = self.info.compute_returns(kind="simple")
        sig2_daily = rets_simple.cov()
        sig2_ann = sig2_daily * periods_per_year
        return sig2_ann

    
    def var_estimation(self, tipo: str = 'hist', periods_per_year: int = 252):
        if tipo == 'hist':
            self.bar_var = self._hist_var(periods_per_year=periods_per_year)
        elif tipo == 'ARMA':
            self.bar_var = self._ARMA()  # asumiendo que luego lo implementas
        else:
            raise ValueError('No es una metodología válida de estimación')
        return self.bar_var

    


    def Markowitz(
        self,
        short: bool = False,
        portfolio: str = "min_vol",
        rf: float = 0.0,
    ) -> pd.Series:
        """
        Resuelve un problema Markowitz clásico usando PyPortfolioOpt:

        - Si portfolio == 'min_vol':
            min w'Σw   s.a. 1'w = 1  (y bounds según 'short')
        - Si portfolio == 'max_sharpe':
            max (μ_p - rf)/σ_p  (tangency portfolio)

        Requiere que:
        - self.bar_mu : pd.Series  (retornos esperados ANUALES)
        - self.bar_var: pd.DataFrame (covarianza ANUAL)
        hayan sido definidos con mean_estimation(...) y var_estimation(...).

        Este es el problema de un periodo de Markowitz (1952): media µ y varianza Σ
        de los retornos, con restricción de presupuesto 1'w=1 y, opcionalmente,
        restricción de no negatividad (long-only).
        """
        if self.bar_mu is None or self.bar_var is None:
            raise ValueError(
                "Primero llama a mean_estimation(...) y var_estimation(...) "
                "para definir self.bar_mu y self.bar_var."
            )

        # Bounds según si se permiten cortos o no
        if short:
            bounds = (-1.0, 1.0)   # largos y cortos acotados
        else:
            bounds = (0.0, 1.0)    # long-only clásico

        # EfficientFrontier acepta Series/DataFrame directamente
        ef = EfficientFrontier(self.bar_mu, self.bar_var, weight_bounds=bounds)

        if portfolio == "min_vol":
            raw_weights: Dict[str, float] = ef.min_volatility()
        elif portfolio == "max_sharpe":
            raw_weights = ef.max_sharpe(risk_free_rate=rf)
        else:
            raise ValueError("portfolio debe ser 'min_vol' o 'max_sharpe'.")

        # Convertir dict -> Serie indexada por tickers en el orden de self.bar_mu
        w_series = pd.Series(raw_weights, dtype=float).reindex(self.bar_mu.index)

        # Guardamos en el objeto
        self.weights = w_series

        return w_series


    @staticmethod
    def min_var_portfolio(
        mu: pd.Series,
        Sigma: pd.DataFrame,
        target_return: float,
        long_only: bool = True,
        solver: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        2.2 Mínima varianza con retorno objetivo (Markowitz), long-only por defecto.

        Problema:
            min_w  w'Σw
            s.a.   1'w = 1
                μ'w ≥ μ*    (Markowitz 1952: mínimo V para un E dado o mayor)
                w ≥ 0       si long_only=True.

        mu : pd.Series (anual)
            Retornos esperados anualizados por activo.
        Sigma : pd.DataFrame (anual)
            Matriz de covarianzas anualizada (dimensión n x n).
        target_return : float (anual)
            Retorno objetivo μ* (mismas unidades que mu).
        """
        # Alinear orden de activos
        tickers = list(mu.index)
        mu = mu.loc[tickers].astype(float)
        Sigma = Sigma.loc[tickers, tickers].astype(float)

        mu_vec = mu.values
        Sigma_mat = Sigma.values

        # Forzar simetría numérica
        Sigma_mat = 0.5 * (Sigma_mat + Sigma_mat.T)

        n = len(mu_vec)
        w = cp.Variable(n)

        constraints = [cp.sum(w) == 1, mu_vec @ w >= float(target_return)]
        if long_only:
            constraints.append(w >= 0)

        objective = cp.Minimize(cp.quad_form(w, Sigma_mat))
        problem = cp.Problem(objective, constraints)

        # Elección de solver por defecto
        if solver is None:
            installed = set(cp.installed_solvers())
            solver = "OSQP" if "OSQP" in installed else "SCS"

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


    def min_var_target(
        self,
        target_return: float,
        mu: Optional[pd.Series] = None,
        Sigma: Optional[pd.DataFrame] = None,
        long_only: bool = True,
        kind: str = "simple",
        periods_per_year: int = 252,
        solver: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Wrapper de instancia para el problema de Markowitz de mínima varianza con
        retorno objetivo (un periodo).

        Si no se pasan mu/Sigma, los estima a partir de retornos diarios de self.info:

            - mu: media diaria * periods_per_year
            - Sigma: covarianza diaria * periods_per_year

        target_return : float (anual)
            Retorno objetivo μ*.
        kind : {'simple','log'}
            Tipo de retorno diario para estimar mu/Sigma si no se pasan.
            Para un Markowitz discreto "clásico" (Markowitz 1952), lo estándar
            es usar retornos simples.
        """
        if (mu is None) or (Sigma is None):
            rets = self.info.compute_returns(kind=kind).dropna(how="any")  # diarios
            mu = rets.mean() * periods_per_year        # anualización simple
            Sigma = rets.cov() * periods_per_year      # anualización de covarianza

        return self.min_var_portfolio(
            mu=mu,
            Sigma=Sigma,
            target_return=target_return,
            long_only=long_only,
            solver=solver,
        )


    @staticmethod
    def min_variance_weights(
        Sigma: pd.DataFrame,
        long_only: bool = True,
        solver: str | None = None
    ) -> pd.Series:
        """
        PMV (mínima varianza sin objetivo de retorno):
            min w'Σw   s.a. 1'w = 1
            (y w >= 0 si long_only=True).

        Devuelve una Serie con los pesos (index=tickers).
        """
        tickers = list(Sigma.columns)
        Sigma = Sigma.loc[tickers, tickers].astype(float)
        Sigma_m = Sigma.values

        # Simetrizar por seguridad numérica
        Sigma_m = 0.5 * (Sigma_m + Sigma_m.T)

        n = len(tickers)
        w = cp.Variable(n)

        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)

        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma_m)), constraints)

        if solver is None:
            installed = set(cp.installed_solvers())
            solver = "OSQP" if "OSQP" in installed else "SCS"

        prob.solve(solver=solver, verbose=False)

        w_val = np.array(w.value).reshape(-1)
        return pd.Series(w_val, index=tickers)


    @staticmethod
    def efficient_frontier_by_return(
        mu: pd.Series,
        Sigma: pd.DataFrame,
        n_points: int = 25,
        mu_targets: np.ndarray | None = None,
        long_only: bool = True,
        solver: str | None = None
    ) -> dict:
        """
        2.3 Frontera eficiente (Markowitz de un periodo):
        Barre retornos objetivo y resuelve el problema de mínima varianza (2.2) para cada punto.

        Problema base (por punto):
            min_w  w'Σw
            s.a.   1'w = 1
                μ'w ≥ μ*     (mínimo riesgo para un retorno dado o mayor)
                w ≥ 0        si long_only=True.

        Parámetros
        ----------
        mu, Sigma : anualizados
            mu: pd.Series con E[R_i] anual.
            Sigma: covarianza anual de retornos.
        n_points  : número de puntos si no se pasa mu_targets.
        mu_targets: array explícito de retornos objetivo μ* (anual). Si None, se genera.
        long_only : restringe w>=0 si True.
        solver    : solver cvxpy ('OSQP' o 'SCS'); auto si None.

        Retorna
        -------
        dict con:
        - 'curve'   : DataFrame con columnas
                        ['target_return', 'expected_return', 'volatility', 'variance']
        - 'weights' : DataFrame (n_points x n_assets) con pesos por punto
        - 'pmv'     : dict con info del PMV (pesos, mu, vol, var)
        - 'statuses': lista de estados del solver por punto
        """
        # Orden y tipos
        tickers = list(mu.index)
        mu = mu.loc[tickers].astype(float)
        Sigma = Sigma.loc[tickers, tickers].astype(float)

        # Simetrizar Σ
        Sigma_values = Sigma.values
        Sigma_values = 0.5 * (Sigma_values + Sigma_values.T)
        Sigma = pd.DataFrame(Sigma_values, index=tickers, columns=tickers)

        #  PMV (sin restricción de retorno)
        w_pmv = Do_portfolio.min_variance_weights(Sigma, long_only=long_only, solver=solver)
        mu_pmv = float(mu @ w_pmv)
        var_pmv = float(w_pmv.values @ Sigma.values @ w_pmv.values)
        vol_pmv = float(np.sqrt(max(var_pmv, 0.0)))

        #  Construir grid de retornos objetivo
        mu_lo_auto = mu_pmv + 1e-8              # un pelo por encima del PMV
        mu_hi_auto = float(mu.max()) - 1e-8     # casi el máximo activo
        if mu_targets is None:
            if mu_lo_auto > mu_hi_auto:         # salvaguarda si todos iguales
                mu_lo_auto, mu_hi_auto = float(mu.min()), float(mu.max())
            mu_targets = np.linspace(mu_lo_auto, mu_hi_auto, n_points)

        #  Barrido usando el solver de 2.2 (min_var_portfolio)
        vols, vars_, rets, statuses = [], [], [], []
        W = []  # lista de Series con pesos

        for mu_star in mu_targets:
            res = Do_portfolio.min_var_portfolio(
                mu=mu,
                Sigma=Sigma,
                target_return=float(mu_star),
                long_only=long_only,
                solver=solver
            )
            statuses.append(res["status"])
            rets.append(res["expected_return"])
            vols.append(res["volatility"])
            vars_.append(res["variance"])
            W.append(res["weights"].reindex(tickers))

        #  Empaquetar resultados
        curve = pd.DataFrame({
            "target_return": np.array(mu_targets, dtype=float),
            "expected_return": np.array(rets, dtype=float),
            "volatility": np.array(vols, dtype=float),
            "variance": np.array(vars_, dtype=float),
        })

        weights_df = pd.DataFrame(W, index=range(len(W)))  # filas = puntos, columnas=tickers
        pmv_info = {
            "weights": w_pmv,
            "expected_return": mu_pmv,
            "volatility": vol_pmv,
            "variance": var_pmv,
        }

        return {
            "curve": curve,
            "weights": weights_df,
            "pmv": pmv_info,
            "statuses": statuses,
        }
    
    @staticmethod
    def efficient_frontier(
        self,
        n_points: int = 25,
        mu_targets: np.ndarray | None = None,
        mu: pd.Series | None = None,
        Sigma: pd.DataFrame | None = None,
        long_only: bool = True,
        kind: str = "simple",
        periods_per_year: int = 252,
        solver: str | None = None,
    ) -> dict:
        """
        Construye la frontera eficiente media-varianza a la Markowitz (un periodo).

        Si mu y Sigma son None:
        - Estima retornos diarios r_i con self.info.compute_returns(kind=kind)
        - Anualiza:
            mu_i   = E[r_i] * periods_per_year
            Sigma  = Cov(r_i, r_j) * periods_per_year

        Si pasas mu y Sigma, se usan tal cual (deben estar ya anualizados).

        Markowitz (1952):
            Para cada punto de la frontera, resuelves:
                min_w  w'Σw
                s.a.   1'w = 1
                    μ'w ≥ μ*.

            La colección de soluciones (σ_p, μ_p) es la frontera eficiente.
        """
        if (mu is None) and (Sigma is None):
            rets = self.info.compute_returns(kind=kind).dropna(how="any")  # diarios
            mu = rets.mean() * periods_per_year
            Sigma = rets.cov() * periods_per_year
        elif (mu is None) or (Sigma is None):
            raise ValueError(
                "O pasas ambos (mu y Sigma) o ninguno; "
                "si ninguno, se estiman a partir de los datos de self.info."
            )

        return Do_portfolio.efficient_frontier_by_return(
            mu=mu,
            Sigma=Sigma,
            n_points=n_points,
            mu_targets=mu_targets,
            long_only=long_only,
            solver=solver,
        )



    @staticmethod
    def min_variance_weights_bounded(
        Sigma: pd.DataFrame,
        lb: float = 0.0,
        ub: float | None = 1.0,
        solver: str | None = None,
    ) -> pd.Series:
        """
        Portafolio de mínima varianza con cotas de peso (Markowitz de un periodo):

            min   w' Σ w
            s.a.  1'w = 1
                lb <= w_i <= ub

        Con lb=0, ub=1 obtienes el GMV estándar sin cortos de Markowitz.
        """
        tickers = list(Sigma.columns)
        Sigma_sub = Sigma.loc[tickers, tickers].astype(float)
        Sigma_m = Sigma_sub.to_numpy()

        # Simetrizar por seguridad numérica (muestras pueden dar Σ ligeramente no simétrica)
        Sigma_m = 0.5 * (Sigma_m + Sigma_m.T)

        n = len(tickers)

        # Chequeos básicos de factibilidad de las cajas
        if ub is not None and ub < lb:
            raise ValueError(f"Infeasible: ub={ub} < lb={lb}")

        # Si todos los pesos >= lb, la suma mínima es n*lb => debe ser <= 1
        if lb * n - 1.0 > 1e-12:
            raise ValueError(f"Infeasible: lb * n = {lb * n:.3f} > 1")

        # Si todos los pesos <= ub, la suma máxima es n*ub => debe ser >= 1
        if (ub is not None) and (ub * n + 1e-12 < 1.0):
            raise ValueError(f"Infeasible: ub * n = {ub * n:.3f} < 1")

        # Variables y restricciones
        w = cp.Variable(n)
        cons = [cp.sum(w) == 1, w >= lb]
        if ub is not None:
            cons.append(w <= ub)

        # Problema cuadrático convexo: varianza = w' Σ w
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma_m)), cons)

        if solver is None:
            installed = set(cp.installed_solvers())
            solver = "OSQP" if "OSQP" in installed else "SCS"

        prob.solve(solver=solver, verbose=False)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"PMV bounded no se resolvió: status = {prob.status}")

        w_val = np.asarray(w.value, dtype=float).reshape(-1)
        return pd.Series(w_val, index=tickers)



    @staticmethod
    def pmv_info(
        mu: pd.Series,
        Sigma: pd.DataFrame,
        long_only: bool = True,
        solver: str | None = None,
    ) -> dict:
        """
        Resumen del portafolio de mínima varianza (GMV) a la Markowitz.

        GMV clásico:
            min   w' Σ w
            s.a.  1'w = 1
                w >= 0   (si long_only)

        Ver TESIFINAL, ec. (30), donde se plantea exactamente este problema.
        """
        # Pesos GMV (usa tu rutina base, sin cotas explícitas salvo long_only)
        w = Do_portfolio.min_variance_weights(
            Sigma, long_only=long_only, solver=solver
        )

        # Alineación y casting a float para evitar sorpresas
        w = w.astype(float)
        mu = mu.loc[w.index].astype(float)
        Sigma_sub = Sigma.loc[w.index, w.index].astype(float)

        mu_p = float(mu @ w)                                # w' mu
        var_p = float(w.to_numpy() @ Sigma_sub.to_numpy() @ w.to_numpy())  # w' Σ w
        vol_p = float(np.sqrt(max(var_p, 0.0)))             # σ_p = sqrt(w' Σ w)

        return {
            "weights": w,
            "expected_return": mu_p,
            "variance": var_p,
            "volatility": vol_p,
        }
    
    @staticmethod
    def pmv_info_bounded(
        mu: pd.Series,
        Sigma: pd.DataFrame,
        lb: float = 0.01,          # por defecto, pesos mínimos del 1%
        ub: float | None = 1.0,
        solver: str | None = None,
    ) -> dict:
        """
        Resumen del portafolio de mínima varianza con cotas (lb, ub).

        Es el mismo GMV de Markowitz, pero forzando pesos entre lb y ub:

            min   w' Σ w
            s.a.  1'w = 1
                lb <= w_i <= ub
        """
        # Pesos GMV con cajas
        w = Do_portfolio.min_variance_weights_bounded(
            Sigma=Sigma, lb=lb, ub=ub, solver=solver
        ).astype(float)

        mu_aligned = mu.loc[w.index].astype(float)
        Sigma_sub = Sigma.loc[w.index, w.index].astype(float)

        w_vec = w.to_numpy()
        Sigma_m = Sigma_sub.to_numpy()
        Sigma_m = 0.5 * (Sigma_m + Sigma_m.T)

        mu_p = float(mu_aligned @ w_vec)
        var_p = float(w_vec @ Sigma_m @ w_vec)
        vol_p = float(np.sqrt(max(var_p, 0.0)))

        return {
            "weights": w,
            "expected_return": mu_p,
            "variance": var_p,
            "volatility": vol_p,
        }


    
    @staticmethod
    def tangency_portfolio(
        mu: pd.Series,
        Sigma: pd.DataFrame,
        rf_ann: float,
        long_only: bool = True,
        solver: str | None = None,
        frontier_points: int = 200,
    ) -> dict:
        """
        Portafolio tangente (máxima razón de Sharpe) a la Markowitz.

        Problema clásico:
            max   (w' μ - rf_ann) / sqrt(w' Σ w)
            s.a.  1'w = 1,  w >= 0 (si long_only)

        TESIFINAL plantea la versión sin rf (ec. (33)); la extensión con rf es directa:
        trabajas con excesos de retorno (μ - rf * 1).

        Implementación:
        1) Formulación SOCP equivalente (homogeneidad en la escala de w):
                max (μ - rf 1)' v
                s.a. v' Σ v <= 1,   v >= 0 (si long_only)
        y luego normalizamos w = v / (1'v).
        2) Si falla o no hay solver cónico, fallback:
        buscamos el punto de la frontera eficiente con Sharpe máximo.
        """
        # Alineamos y casteamos
        tickers = list(mu.index)
        mu = mu.loc[tickers].astype(float)
        Sigma = Sigma.loc[tickers, tickers].astype(float)

        # Simetrizar Σ (evita problemas numéricos en quad_form)
        Sigma_m = Sigma.to_numpy()
        Sigma_m = 0.5 * (Sigma_m + Sigma_m.T)

        installed = set(cp.installed_solvers())
        preferred_conic = ["SCS", "ECOS", "CLARABEL"]

        # Elegir solver cónico, si existe
        if solver is not None and solver in installed:
            chosen = solver
        else:
            chosen = next((s for s in preferred_conic if s in installed), None)

        # ---------- 1) Intento SOCP (formulación exacta de Máx-Sharpe) ----------
        if chosen is not None:
            try:
                mu_ex = mu - rf_ann  # vector de excesos de retorno (μ_i - rf)

                n = len(tickers)
                v = cp.Variable(n)

                cons = [cp.quad_form(v, Sigma_m) <= 1]
                if long_only:
                    cons.append(v >= 0)

                prob = cp.Problem(cp.Maximize(mu_ex.to_numpy() @ v), cons)
                prob.solve(solver=chosen, verbose=False)

                v_val = np.asarray(v.value, dtype=float).reshape(-1)

                # Chequeo numérico básico
                if (
                    v_val is None
                    or (not np.isfinite(v_val).all())
                    or np.allclose(v_val.sum(), 0.0)
                ):
                    raise RuntimeError("Solución degenerada del solver cónico.")

                # Normalizamos para obtener pesos que suman 1
                w = v_val / v_val.sum()
                w = pd.Series(w, index=tickers)

                mu_p = float(mu @ w)
                var_p = float(w.to_numpy() @ Sigma_m @ w.to_numpy())
                vol_p = float(np.sqrt(max(var_p, 0.0)))
                sharpe_p = (mu_p - rf_ann) / vol_p if vol_p > 0 else np.nan

                return {
                    "status": f"optimal_socp ({chosen})",
                    "weights": w,
                    "expected_return": mu_p,
                    "volatility": vol_p,
                    "sharpe": sharpe_p,
                }
            except Exception:
                # Si algo va mal, caemos al fallback
                pass

        # ---------- 2) Fallback: buscar Máx-Sharpe sobre la frontera ----------
        ef = Do_portfolio.efficient_frontier_by_return(
            mu=mu,
            Sigma=Sigma,
            n_points=frontier_points,
            mu_targets=None,
            long_only=long_only,
            solver="OSQP" if "OSQP" in installed else None,
        )

        curve = ef["curve"]      # DataFrame con expected_return y volatility
        weights = ef["weights"]  # DataFrame con los w_k de cada punto

        sharpe = (curve["expected_return"] - rf_ann) / curve["volatility"]
        idx = sharpe.idxmax()  # usamos el índice tal cual

        w = weights.loc[idx].reindex(tickers).astype(float)
        mu_p = float(curve.loc[idx, "expected_return"])
        vol_p = float(curve.loc[idx, "volatility"])
        sharpe_p = float(sharpe.loc[idx])

        return {
            "status": "fallback_frontier_argmax",
            "weights": w,
            "expected_return": mu_p,
            "volatility": vol_p,
            "sharpe": sharpe_p,
        }



    def tangency(
        self,
        rf_ann: float,
        mu: Optional[pd.Series] = None,
        Sigma: Optional[pd.DataFrame] = None,
        long_only: bool = True,
        kind: str = "simple",
        periods_per_year: int = 252,
        solver: Optional[str] = None,
    ) -> dict:
        """
        Portafolio tangente (máx. Sharpe) a partir de μ y Σ ANUALES.

        Si mu y Sigma no se pasan:
        - estima retornos diarios con self.info.compute_returns(kind)
        - anualiza como en Markowitz/MPT:
                μ_ann  = μ_daily * periods_per_year
                Σ_ann  = Σ_daily * periods_per_year

        Si se pasan mu y Sigma:
        - Se asume que ya están anualizados y en unidades coherentes con rf_ann.

        Parámetros
        ----------
        rf_ann : float
            Tasa libre de riesgo ANUAL en las mismas unidades que μ_ann:
            - si μ son retornos simples anuales => rf_ann es retorno simple anual;
            - si μ son log-retornos anuales   => rf_ann es log rf anual.
        kind : {'simple','log'}
            Tipo de retorno diario usado si se estiman μ y Σ internamente.
            Para un Markowitz discreto clásico, 'simple' es lo estándar.
        """
        # Estimación de μ y Σ si no vienen dados
        if (mu is None) and (Sigma is None):
            rets = self.info.compute_returns(kind=kind).dropna(how="any")  # retornos diarios
            mu = rets.mean() * periods_per_year
            Sigma = rets.cov() * periods_per_year
        elif (mu is None) or (Sigma is None):
            raise ValueError(
                "O pasas ambos (mu y Sigma) o ninguno; "
                "si ninguno, se estiman a partir de los datos de self.info."
            )

        # Aseguramos consistencia de tipos y orden
        mu = mu.astype(float)
        Sigma = Sigma.loc[mu.index, mu.index].astype(float)

        return Do_portfolio.tangency_portfolio(
            mu=mu,
            Sigma=Sigma,
            rf_ann=rf_ann,
            long_only=long_only,
            solver=solver,
        )




    from typing import Any, List, Optional, Union

    def compare_candidates_5_2(
        self,
        mu: Optional[pd.Series] = None,
        Sigma: Optional[pd.DataFrame] = None,
        rf_ann: float = 0.0,
        long_only: bool = True,
        *,
        include_pmv_bounded: bool = False,
        lb: float = 0.0,
        ub: Optional[float] = 1.0,
        include_markowitz_target: bool = True,
        mu_target: Optional[float] = None,
        include_mc: bool = True,
        n_port: int = 10_000,
        seed: int = 42,
        kind: str = "simple",
        periods_per_year: int = 252,
        return_details: bool = False,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict]]:
        """
        Tabla comparativa (tipo sección 5.2 TESI) con métricas y pesos de:
        - PMV (o PMV con cotas si include_pmv_bounded=True)
        - Tangente (máx. Sharpe con rf_ann)
        - Markowitz(μ*) (min var para retorno objetivo)
        - Monte Carlo: Max Sharpe y Min Vol

        μ y Σ deben estar ANUALIZADOS y en las mismas unidades de rf_ann.
        Si mu o Sigma no se pasan, se estiman desde self.info usando `kind` y `periods_per_year`.
        """

        # 1) Estimar μ/Sigma si no vienen
        if (mu is None) and (Sigma is None):
            rets = self.info.compute_returns(kind=kind).dropna(how="any")  # retornos diarios
            mu = rets.mean() * periods_per_year
            Sigma = rets.cov() * periods_per_year
        elif (mu is None) or (Sigma is None):
            raise ValueError(
                "O pasas ambos (mu y Sigma) o ninguno; "
                "si ninguno, se estiman a partir de los datos de self.info."
            )

        # Aseguramos consistencia
        mu = mu.astype(float)
        Sigma = Sigma.loc[mu.index, mu.index].astype(float)

        tickers: List[str] = list(mu.index)
        weight_cols = [f"w_{t}" for t in tickers]

        def _row_from_obj(name: str, obj: Any) -> dict:
            """
            Construye una fila de la tabla a partir de:
            - dict devuelto por pmv_info / pmv_info_bounded / tangency_portfolio / min_var_portfolio
            - Serie (fila) del Monte Carlo (salida de pick_candidates).
            """
            row: dict[str, Any] = {"Portafolio": name}

            # Caso 1: dict con expected_return / volatility (nuestros solvers)
            if isinstance(obj, dict) and ("expected_return" in obj) and ("volatility" in obj):
                row["μ_ann"] = float(obj["expected_return"])
                row["σ_ann"] = float(obj["volatility"])

                sharpe = obj.get("sharpe", None)
                if sharpe is not None and np.isfinite(sharpe):
                    row["Sharpe"] = float(sharpe)
                else:
                    row["Sharpe"] = (
                        (row["μ_ann"] - rf_ann) / row["σ_ann"]
                        if row["σ_ann"] > 0
                        else np.nan
                    )

                # Pesos → usamos _coerce_weights (acepta Series / arrays / listas / dict)
                w_obj = obj.get("weights", None)
                if w_obj is not None:
                    w_ser = Do_portfolio._coerce_weights(w_obj, tickers)
                    for t in tickers:
                        row[f"w_{t}"] = float(w_ser[t])
                return row

            # Caso 2: Series / dict-like de Monte Carlo (aux.loc[...] de Monte_Carlo)
            if hasattr(obj, "get"):
                row["μ_ann"] = float(obj.get("ann_returns"))
                row["σ_ann"] = float(obj.get("ann_vol"))
                row["Sharpe"] = float(obj.get("Sharpe"))
                for t in tickers:
                    col = f"w_{t}"
                    if col in obj:
                        row[col] = float(obj.get(col))
                return row

            raise ValueError("Objeto de candidato no reconocido para la tabla 5.2")

        filas: List[dict] = []
        detalles: dict[str, Any] = {}

        # 2) PMV (con o sin cotas)
        if include_pmv_bounded:
            pmv_obj = Do_portfolio.pmv_info_bounded(mu, Sigma, lb=lb, ub=ub)
            filas.append(_row_from_obj(f"PMV (lb={lb:.2%})", pmv_obj))
            detalles["PMV_bounded"] = pmv_obj
        else:
            pmv_obj = Do_portfolio.pmv_info(mu, Sigma, long_only=long_only)
            filas.append(_row_from_obj("PMV", pmv_obj))
            detalles["PMV"] = pmv_obj

        # 3) Tangente (máx. Sharpe)
        tan_obj = Do_portfolio.tangency_portfolio(
            mu=mu,
            Sigma=Sigma,
            rf_ann=rf_ann,
            long_only=long_only,
        )
        filas.append(_row_from_obj("Tangente", tan_obj))
        detalles["Tangente"] = tan_obj

        # 4) Markowitz con retorno objetivo μ*
        if include_markowitz_target:
            mu_star = float(mu.mean()) if mu_target is None else float(mu_target)
            mv_obj = Do_portfolio.min_var_portfolio(
                mu=mu,
                Sigma=Sigma,
                target_return=mu_star,
                long_only=long_only,
            )
            filas.append(_row_from_obj(f"Markowitz(μ*={mu_star:.3f})", mv_obj))
            detalles["Markowitz_mu*"] = mv_obj

        # 5) Monte Carlo → max Sharpe y min Vol
        if include_mc:
            mc = self.pick_candidates(
                n_port=n_port,
                short=not long_only,         # coherente con el régimen de cortos
                rf=rf_ann,
                seed=seed,
                periods_per_year=periods_per_year,
            )
            filas.append(_row_from_obj("MC Max Sharpe", mc["max Sharpe"]))
            filas.append(_row_from_obj("MC Min Vol",    mc["min Vol"]))
            detalles["MC_max_Sharpe"] = mc["max Sharpe"]
            detalles["MC_min_Vol"]    = mc["min Vol"]

        # 6) Construcción del DataFrame final
        tabla = pd.DataFrame(filas)

        metric_cols = ["μ_ann", "σ_ann", "Sharpe"]
        for c in metric_cols:
            if c in tabla.columns:
                tabla[c] = pd.to_numeric(tabla[c], errors="coerce")
        for c in weight_cols:
            if c in tabla.columns:
                tabla[c] = pd.to_numeric(tabla[c], errors="coerce")

        cols = ["Portafolio"] + metric_cols + weight_cols
        tabla = tabla[[c for c in cols if c in tabla.columns]]

        return (tabla, detalles) if return_details else tabla



    @staticmethod
    def _coerce_weights(obj: Any, tickers: List[str]) -> pd.Series:
        """
        Convierte `obj` en un vector de pesos (pd.Series con índice = tickers).

        Acepta:
        - pd.Series con índice = tickers o con claves 'w_TICKER'
        - dict con:
            * 'weights' (pd.Series, np.ndarray o list), o
            * claves tipo 'w_TICKER', o
            * claves iguales a los TICKERS
        - np.ndarray / list / tuple con la misma longitud que tickers

        Normaliza a suma = 1 y rellena faltantes con 0.
        """
        # Serie directa
        if isinstance(obj, pd.Series):
            # Si la Serie trae 'w_TICKER', construir desde ahí
            have_prefixed = any((f"w_{t}" in obj.index) for t in tickers)
            if have_prefixed:
                w = pd.Series(
                    {t: obj.get(f"w_{t}", np.nan) for t in tickers},
                    index=tickers,
                    dtype="float64",
                )
            else:
                # Si trae directamente los TICKERS como índice
                w = obj.reindex(tickers).astype(float)

        # Diccionario
        elif isinstance(obj, dict):
            if "weights" in obj:
                w_raw = obj["weights"]
                if isinstance(w_raw, pd.Series):
                    w = w_raw.reindex(tickers).astype(float)
                else:
                    w_arr = np.asarray(w_raw, dtype=float).reshape(-1)
                    if len(w_arr) != len(tickers):
                        raise ValueError("len(weights) != número de tickers")
                    w = pd.Series(w_arr, index=tickers, dtype="float64")
            else:
                # a) Formato Monte Carlo: 'w_TICKER'
                if any((f"w_{t}" in obj) for t in tickers):
                    w = pd.Series(
                        {t: obj.get(f"w_{t}", np.nan) for t in tickers},
                        index=tickers,
                        dtype="float64",
                    )
                # b) Claves = TICKERS directamente
                elif any((t in obj) for t in tickers):
                    w = pd.Series(
                        {t: obj.get(t, np.nan) for t in tickers},
                        index=tickers,
                        dtype="float64",
                    )
                else:
                    raise ValueError("Dict sin 'weights' ni claves w_TICKER/TICKER")

        # Array/list plano
        elif isinstance(obj, (np.ndarray, list, tuple)):
            arr = np.asarray(obj, dtype=float).reshape(-1)
            if len(arr) != len(tickers):
                raise ValueError("Vector de pesos con longitud distinta a tickers")
            w = pd.Series(arr, index=tickers, dtype="float64")

        else:
            raise ValueError("Formato de pesos no reconocido")

        # Limpieza y normalización
        w = w.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            raise ValueError("Suma de pesos no positiva")
        return w / s





    @staticmethod
    def backtest_buy_and_hold(
        prices: pd.DataFrame,
        weights_map: Dict[str, Any],
        V0: float = 100_000.0
    ) -> pd.DataFrame:
        """
        6.1 — Backtest buy & hold (sin rebalanceo).

        prices      : DataFrame de precios (Adj Close) indexado por fecha, columnas=tickers.
        weights_map : dict {nombre_portafolio: pesos}
                    (acepta formatos soportados por _coerce_weights).
        V0          : valor inicial.

        Retorna:
        DataFrame con columnas = nombres de portafolio,
        filas = fechas, valores en moneda (base V0).
        """
        if V0 <= 0:
            raise ValueError("V0 debe ser positivo.")

        prices = prices.sort_index().copy()
        # Limpieza mínima: sin NaN en ninguna fecha
        prices = prices.dropna(how="any")

        # Primer vector de precios (t0); no puede haber ceros
        p0 = prices.iloc[0].replace(0.0, np.nan)
        if p0.isna().any():
            raise ValueError("Precios iniciales contienen ceros/NaN; revise el rango de backtest.")

        tickers = list(prices.columns)
        values: Dict[str, pd.Series] = {}

        for name, w_obj in weights_map.items():
            # Serie index=tickers, suma=1
            w = Do_portfolio._coerce_weights(w_obj, tickers)
            # Número de acciones que compro en t0
            shares = (V0 * w) / p0
            # Valor diario = sum_i shares_i * P_{t,i}
            vals = (prices * shares).sum(axis=1)
            values[name] = vals

        return pd.DataFrame(values, index=prices.index)

    @staticmethod
    def run_backtest(
        self,
        weights_map: Dict[str, Any],
        start_bt: str,
        end_bt: str,
        V0: float = 100_000.0,
        price_col: str = "Adj Close"
    ) -> pd.DataFrame:
        """
        Descarga precios del periodo de backtest y ejecuta el buy & hold
        para varios portafolios.

        Reutiliza tu clase yahoo_data.yahoo_data (no imprime ni grafica).

        weights_map : dict {nombre_portafolio: pesos}
        start_bt, end_bt : fechas (YYYY-MM-DD) del backtest
        V0        : capital inicial
        price_col : columna de precios (por defecto 'Adj Close')
        """
        from yahoo_data import yahoo_data as YahooData

        if V0 <= 0:
            raise ValueError("V0 debe ser positivo.")

        # Instancia para bajar precios del backtest
        try:
            Ybt = YahooData(
                tickers=self.tickers,
                start=start_bt,
                end=end_bt,
                price_col=price_col
            )
        except TypeError:
            # Por compatibilidad si la clase no acepta price_col
            Ybt = YahooData(self.tickers, start=start_bt, end=end_bt)

        prices_bt = (
            Ybt.data.copy()
            .sort_index()
            .dropna(how="any")
        )

        return Do_portfolio.backtest_buy_and_hold(prices_bt, weights_map, V0=V0)

    
    @staticmethod
    def summarize_backtest(
        values: pd.DataFrame,
        rf_daily: Optional[pd.Series] = None,
        periods_per_year: int = 252
    ) -> pd.DataFrame:
        """
        Resumen de métricas de backtest (buy & hold):
        - CAGR (retorno anual compuesto realizado)
        - Volatilidad anual realizada
        - Sharpe (exceso vs rf_daily)
        - Máx. Drawdown
        - Calmar (CAGR / |MáxDD|)

        Parameters
        ----------
        values : DataFrame de valores (columnas = portafolios).
        rf_daily : Serie de tasa libre diaria (en proporción) alineada al índice de 'values'.
                Debe estar en el MISMO 'kind' que los retornos implícitos (simple).
                Si None, asume rf=0.
        periods_per_year : 252 por defecto.

        Returns
        -------
        DataFrame con columnas: ['CAGR','AnnVol','Sharpe','MaxDD','Calmar']
        """
        if not isinstance(values, pd.DataFrame) or values.shape[0] < 2:
            raise ValueError(
                "values debe ser un DataFrame con >1 fila (serie de valores del backtest)."
            )

        values = values.sort_index()
        # Retornos simples diarios
        rets = values.pct_change().dropna()

        # rf diario alineado (si no viene, 0)
        if rf_daily is None:
            rf_al = pd.Series(0.0, index=rets.index)
        else:
            rf_al = rf_daily.reindex(rets.index).fillna(method="ffill").fillna(0.0)
            # si viene como DataFrame de 1 columna, conviértelo a Serie
            if isinstance(rf_al, pd.DataFrame):
                rf_al = rf_al.squeeze()

        # Métricas realizadas
        n = rets.shape[0]

        # CAGR: (V_T / V_0)^(periods_per_year / n) - 1
        cagr = (values.iloc[-1] / values.iloc[0]) ** (periods_per_year / n) - 1.0

        # Volatilidad anual de los retornos brutos
        ann_vol = rets.std(ddof=1) * np.sqrt(periods_per_year)

        # Sharpe con retorno en exceso diario:
        #   μ_excess,ann = E[r - rf] * periods_per_year
        #   σ_excess,ann = std(r - rf) * sqrt(periods_per_year)
        excess = rets.sub(rf_al, axis=0)
        excess_std_ann = excess.std(ddof=1) * np.sqrt(periods_per_year)
        sharpe = (excess.mean() * periods_per_year) / excess_std_ann
        sharpe.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drawdown y métricas asociadas
        cum = (1.0 + rets).cumprod()
        rolling_peak = cum.cummax()
        dd = cum / rolling_peak - 1.0
        maxdd = dd.min()  # valor negativo
        calmar = cagr / maxdd.abs()
        calmar.replace([np.inf, -np.inf], np.nan, inplace=True)

        out = pd.DataFrame({
            "CAGR":   cagr,
            "AnnVol": ann_vol,
            "Sharpe": sharpe,
            "MaxDD":  maxdd,
            "Calmar": calmar,
        })
        return out



    @staticmethod
    def drawdown(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calcula el drawdown de cada portafolio a partir de 'values' (valores monetarios).

        Retorna
        -------
        dd : DataFrame
            Drawdowns por fecha y portafolio (0 en máximos históricos, negativo en caídas).
        resumen : DataFrame
            Índice = nombre de portafolio, columnas:
            - MaxDD        : drawdown máximo (número negativo)
            - PeakDate     : fecha del pico previo
            - TroughDate   : fecha del valle (mínimo)
            - RecoveryDate : primera fecha posterior donde dd >= 0 (NaT si nunca recupera)
        """
        if not isinstance(values, pd.DataFrame) or values.shape[0] < 2:
            raise ValueError("values debe ser DataFrame con >1 fila.")

        # Aseguramos orden temporal y tipo numérico
        values = values.sort_index().astype(float)

        # Normaliza por valor inicial para trabajar en múltiplos (no cambia DD)
        cum = values.div(values.iloc[0])
        peak = cum.cummax()
        dd = cum / peak - 1.0  # 0 en máximos, negativo en caídas

        # Resumen por columna
        rows = []
        for col in dd.columns:
            s_dd = dd[col]
            maxdd = float(s_dd.min())        # drawdown máximo (negativo)
            t_trough = s_dd.idxmin()         # fecha del valle

            # Pico más alto alcanzado antes (o en) el valle
            s_cum = cum[col]
            t_peak = s_cum.loc[:t_trough].idxmax()

            # Primera recuperación (dd >= 0) después del valle
            s_after = s_dd.loc[t_trough:]
            rec_index = s_after[s_after >= 0].index
            t_rec = rec_index.min() if len(rec_index) > 0 else pd.NaT

            rows.append({
                "Portafolio": col,
                "MaxDD": maxdd,
                "PeakDate": t_peak,
                "TroughDate": t_trough,
                "RecoveryDate": t_rec,
            })

        resumen = pd.DataFrame(rows).set_index("Portafolio")
        return dd, resumen




    @staticmethod
    def rolling_stats(
        values: pd.DataFrame,
        rf_daily: pd.Series | None = None,
        window: int = 63,
        periods_per_year: int = 252,
        auto_shrink: bool = True,
        min_window: int = 10
    ) -> dict[str, pd.DataFrame]:
        """
        Métricas móviles (ventana 'window') sobre valores de portafolio.

        - Usa retornos SIMPLES diarios (values.pct_change()).
        - Devuelve:
            * mu   : medias anuales móviles
            * vol  : volatilidades anuales móviles
            * sharpe : Sharpe móvil usando retornos en exceso vs rf_daily

        Si auto_shrink=True y la muestra es corta, reduce la ventana a
        min(window, len(values)-1), pero si esa ventana es < min_window lanza error.
        """
        if not isinstance(values, pd.DataFrame):
            raise ValueError("values debe ser DataFrame.")

        values = values.sort_index()
        rets = values.pct_change().dropna()

        # Ajuste de ventana si la muestra es corta
        if auto_shrink:
            # ventana máxima posible = len(rets)
            win_raw = min(window, len(rets))
            if win_raw < min_window:
                raise ValueError(
                    f"Muestra insuficiente: len(rets)={len(rets)} < min_window={min_window}"
                )
            window = int(win_raw)

        # Si no hay auto_shrink, aún así chequeamos viabilidad
        if len(rets) < max(2, window):
            raise ValueError(f"Muestra insuficiente para rolling: len(rets)={len(rets)}, window={window}")

        # Alinear rf diaria si viene
        if rf_daily is None:
            rf_al = pd.Series(0.0, index=rets.index)
        else:
            rf_al = rf_daily.reindex(rets.index).fillna(method="ffill").fillna(0.0)
            if isinstance(rf_al, pd.DataFrame):
                rf_al = rf_al.squeeze()

        # Medias y volatilidades móviles de retornos brutos
        roll_mean = rets.rolling(window).mean()
        roll_vol  = rets.rolling(window).std(ddof=1)
        mu_ann    = roll_mean * periods_per_year
        vol_ann   = roll_vol  * np.sqrt(periods_per_year)

        # Sharpe móvil usando retornos en exceso
        excess = rets.sub(rf_al, axis=0)
        roll_excess_mean = excess.rolling(window).mean()
        roll_excess_std  = excess.rolling(window).std(ddof=1)
        excess_mu_ann    = roll_excess_mean * periods_per_year
        excess_sigma_ann = roll_excess_std * np.sqrt(periods_per_year)

        sharpe = excess_mu_ann / excess_sigma_ann
        sharpe.replace([np.inf, -np.inf], np.nan, inplace=True)

        return {"mu": mu_ann, "vol": vol_ann, "sharpe": sharpe}




    @staticmethod
    def dd_table(values: pd.DataFrame) -> pd.DataFrame:
        """
        Atajo: devuelve una tabla 'bonita' de drawdowns máximos con fechas y duración.

        Retorna un DataFrame con:
        - MaxDD
        - PeakDate
        - TroughDate
        - RecoveryDate
        - Days_Peak2Trough
        - Days_Trough2Rec
        """
        # Necesitamos índice de fechas para poder medir duraciones en días
        if not isinstance(values.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise TypeError(
                "El índice de 'values' debe ser DatetimeIndex o PeriodIndex "
                "para poder calcular duraciones en días."
            )

        dd, res = Do_portfolio.drawdown(values)

        # Duración pico->valle y valle->recuperación (si existiera)
        durations = []
        for p, row in res.iterrows():
            peak, trough, rec = row["PeakDate"], row["TroughDate"], row["RecoveryDate"]
            dur_down = (trough - peak).days
            dur_recover = (rec - trough).days if pd.notna(rec) else np.nan
            durations.append((dur_down, dur_recover))

        res["Days_Peak2Trough"] = [d[0] for d in durations]
        res["Days_Trough2Rec"]  = [d[1] for d in durations]

        return res

    @staticmethod
    def merton_raw(
        r: pd.Series | float,
        gamma: float,
        Sigma: pd.DataFrame,
        mu: pd.Series,
    ) -> pd.Series:
        """
        Regla de Merton tal como la tienes en el script (Rogers, cap. 1):

            pi = (Σ^{-1} (μ - r)) / γ

        Donde en tu código original r venía como Series y usabas r.loc[0].
        Aquí acepto float o Serie (uso el primer elemento si es Serie).

        Parámetros
        ----------
        r     : float o pd.Series con la tasa libre ANUAL.
        gamma : aversión al riesgo (CRRA, >0).
        Sigma : matriz de covarianza ANUAL (DataFrame, índices = activos).
        mu    : vector de retornos esperados ANUALES (Series, índice = activos).

        Devuelve
        --------
        pd.Series con los pesos Merton en los activos riesgosos (pueden no sumar 1).
        """
        if gamma <= 0:
            raise ValueError("gamma debe ser > 0 (aversión al riesgo CRRA).")

        # r exacto como en tu código: si es Serie uso el primer elemento
        if isinstance(r, pd.Series):
            rf = float(r.iloc[0])
        else:
            rf = float(r)

        # Alineo mu y Sigma
        tickers = list(mu.index)
        mu = mu.loc[tickers].astype(float)
        Sigma = Sigma.loc[tickers, tickers].astype(float)

        # Tu línea original: pi = (np.linalg.inv(Sigma).dot(mu - r.loc[0]))/gamma
        Sigma_inv = np.linalg.inv(Sigma.values)
        pi_vec = Sigma_inv.dot(mu.values - rf) / gamma

        return pd.Series(pi_vec, index=tickers, name="pi_Merton")

    def merton_sensitivity_rf(
        self,
        gamma: float,
        r_min: float = 0.0,
        r_max: float = 0.15,
        n_points: int = 25,
        use_bar: bool = True,
        kind: str = "simple",
        periods_per_year: int = 252,
    ) -> pd.DataFrame:
        """
        Sensibilidad de los pesos de Merton a la tasa libre de riesgo.

        Construye un grid de tasas r en [r_min, r_max] y aplica la misma lógica
        que en tu script (merton(aux, gamma, Sigma_annual, mu_annual)).

        Devuelve un DataFrame:
          índice  -> r (tasa libre)
          columnas -> activos, con π_i(r).
        """
        # Preparamos μ y Σ igual que en merton(...)
        if use_bar:
            if self.bar_mu is None or self.bar_var is None:
                raise ValueError(
                    "bar_mu/bar_var no definidos. Llama antes a "
                    "mean_estimation(...) y var_estimation(...), o usa use_bar=False."
                )
            mu_ann = self.bar_mu
            Sigma_ann = self.bar_var
        else:
            rets = self.info.compute_returns(kind=kind).dropna(how="any")
            mu_ann = rets.mean() * periods_per_year
            Sigma_ann = rets.cov() * periods_per_year

        vec_r = np.linspace(r_min, r_max, n_points)
        rows = []

        for rr in vec_r:
            # Igual que tu aux = pd.Series(rr)
            r_ser = pd.Series(rr)
            pi = Do_portfolio.merton_raw(r=r_ser, gamma=gamma,
                                         Sigma=Sigma_ann, mu=mu_ann)
            pi.name = rr
            rows.append(pi)

        pi_df = pd.DataFrame(rows)
        pi_df.index.name = "r_free"
        return pi_df


    @staticmethod
    def merton_interest_rate_sensitivity(
        r: float | pd.Series,
        gamma: float,
        Sigma: pd.DataFrame,
        mu: pd.Series,
        dr: float = 1e-4,
    ) -> dict[str, pd.Series]:
        """
        Riesgo de tasa de interés en el problema de Merton (tasa constante).

        Implementa la fórmula cerrada de Merton (Rogers, cap. 1 / TESIFINAL):
            π(r) = Σ^{-1} (μ - r·1) / γ

        y su derivada respecto a la tasa libre r:
            dπ/dr = - Σ^{-1} 1 / γ

        donde:
        - μ  : retornos esperados ANUALES (Series, índice = activos)
        - Σ  : matriz de covarianza ANUAL (DataFrame, índices = activos)
        - γ  : aversión al riesgo (CRRA, >0)
        - r  : tasa libre ANUAL (float o Series; si es Series se usa el primer elemento)

        El vector dπ/dr se interpreta como un "delta" de tasa de interés: cuánto
        cambia el peso óptimo de cada activo ante un pequeño cambio en r.

        Además de la derivada analítica, calculamos una derivada numérica de
        chequeo usando un incremento pequeño dr.
        """
        if gamma <= 0:
            raise ValueError("gamma debe ser > 0 (aversión al riesgo CRRA).")

        # r igual que en merton_raw: si es Serie, usar el primer elemento
        if isinstance(r, pd.Series):
            rf = float(r.iloc[0])
        else:
            rf = float(r)

        # Alineamos mu y Sigma
        tickers = list(mu.index)
        mu = mu.loc[tickers].astype(float)
        Sigma = Sigma.loc[tickers, tickers].astype(float)

        # Inversa de Σ
        Sigma_inv = np.linalg.inv(Sigma.values)
        ones = np.ones(len(tickers))

        # Pesos Merton base: π(r)
        pi_vec = Sigma_inv.dot(mu.values - rf * ones) / gamma
        pi = pd.Series(pi_vec, index=tickers, name="pi_Merton")

        # Derivada analítica: dπ/dr = - Σ^{-1} 1 / γ
        dpi_dr_vec = - Sigma_inv.dot(ones) / gamma
        dpi_dr_analytic = pd.Series(dpi_dr_vec, index=tickers, name="dpi_dr_analytic")

        # Derivada numérica (para sanity check): π(r + dr) - π(r) / dr
        rf_up = rf + dr
        pi_up_vec = Sigma_inv.dot(mu.values - rf_up * ones) / gamma
        pi_up = pd.Series(pi_up_vec, index=tickers, name="pi_Merton_r_plus_dr")

        dpi_dr_numeric = (pi_up - pi) / dr
        dpi_dr_numeric.name = "dpi_dr_numeric"

        return {
            "pi": pi,
            "dpi_dr_analytic": dpi_dr_analytic,
            "dpi_dr_numeric": dpi_dr_numeric,
        }

    def interest_rate_risk_merton(
        self,
        gamma: float,
        r0: float,
        dr: float = 0.01,
        use_bar: bool = True,
        kind: str = "simple",
        periods_per_year: int = 252,
    ) -> dict[str, pd.Series]:
        """
        Envoltorio de instancia para medir RIESGO DE TASA DE INTERÉS
        del portafolio de Merton, usando μ y Σ ANUALES de self.

        Básicamente aplica merton_interest_rate_sensitivity con:

            π(r0)      = Σ^{-1} (μ - r0·1) / γ
            dπ/dr(r0)  = - Σ^{-1} 1 / γ

        y, opcionalmente, re-estima μ y Σ a partir de los datos de self.info.

        Parámetros
        ----------
        gamma : float
            Aversión al riesgo CRRA (>0).
        r0 : float
            Tasa libre de riesgo ANUAL alrededor de la cual medimos la sensibilidad.
        dr : float
            Incremento pequeño para la derivada numérica (por defecto 1%).
        use_bar : bool
            Si True, usa self.bar_mu / self.bar_var (precalculados con
            mean_estimation / var_estimation). Si False, re-estima μ y Σ
            desde retornos diarios de self.info.
        kind : {'simple', 'log'}
            Tipo de retorno diario si use_bar=False (para estimar μ/Sigma).
        periods_per_year : int
            Períodos por año (252 por defecto).

        Devuelve
        --------
        dict con:
        - 'pi'               : pesos de Merton en r0
        - 'dpi_dr_analytic'  : derivada analítica dπ/dr en r0
        - 'dpi_dr_numeric'   : derivada numérica aproximada en r0
        """
        # Preparar μ y Σ
        if use_bar:
            if self.bar_mu is None or self.bar_var is None:
                raise ValueError(
                    "bar_mu/bar_var no definidos. Llama antes a "
                    "mean_estimation(...) y var_estimation(...), o usa use_bar=False."
                )
            mu_ann = self.bar_mu
            Sigma_ann = self.bar_var
        else:
            rets = self.info.compute_returns(kind=kind).dropna(how="any")
            mu_ann = rets.mean() * periods_per_year
            Sigma_ann = rets.cov() * periods_per_year

        # Llamar al estático
        out = Do_portfolio.merton_interest_rate_sensitivity(
            r=r0,
            gamma=gamma,
            Sigma=Sigma_ann,
            mu=mu_ann,
            dr=dr,
        )

        return out


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
    
    
    
    