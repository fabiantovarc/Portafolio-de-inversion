#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 10:38:45 2025

@author: camilocastillo
"""
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
    
    
    
    