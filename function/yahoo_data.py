#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:28:03 2025

@author: camilocastillo
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Optional


class yahoo_data:
    "Esta clase descarga la información de cierre para una lista de tickers"
    def __init__(self,tickers:List[str], start: str='2021-01-01', end: Optional[str]=None) -> pd.DataFrame:
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data = yf.download(self.tickers,start = start,end = end, progress=False,auto_adjust=False).Close
        if isinstance(self.data, pd.Series):
            self.data = self.data.to_frame()
        self.data.dropna(how= 'all', inplace = True)
    
    def compute_returns(self, kind : str = 'log') -> pd.DataFrame:
        "Calcula los retornos sean simples o logretornos"
        if kind == 'log':
            self.rets = np.log(self.data / self.data.shift(1))
        elif kind == 'simple':
            self.rets = self.data.pct_change()
        else:
            raise ValueError('kind  debe ser log o simple')
        self.rets.dropna(inplace = True)
        print(f'*** retornos tipo {kind} creados ***')
        return self.rets
                             