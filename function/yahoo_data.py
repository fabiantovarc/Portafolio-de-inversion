import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Optional, Union

class yahoo_data:
    """
    Clase para descargar precios (por defecto 'Adj Close'), limpiar y calcular retornos.
    Ahora incluye métodos de clase/estáticos para:
      - download_data      (1.1)
      - preprocess_data    (1.1)
      - convert_rf         (1.2)
      - compute_returns_from_prices (aux para 1.3 sin tocar el estado)
      - excess_returns     (1.3)
    """
    def __init__(self,
                 tickers: List[str],
                 start: str = '2021-01-01',
                 end: Optional[str] = None,
                 price_col: str = "Adj Close",  # mejora: usar ajustados por defecto
                 auto_adjust: bool = False):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.price_col = price_col
        self.auto_adjust = auto_adjust

        raw = yf.download(self.tickers, start=self.start, end=self.end,
                          progress=False, auto_adjust=self.auto_adjust)

        # Selecciona la columna de precios deseada (por defecto 'Adj Close')
        if isinstance(raw, pd.DataFrame) and self.price_col in raw.columns:
            data = raw[self.price_col]
        else:
            # Compatibilidad si el MultiIndex viene invertido o casos raros
            try:
                data = raw.xs(self.price_col, axis=1, level=0, drop_level=False)
                # Si queda con MultiIndex, colapsar al nivel de tickers
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.droplevel(0, axis=1)
            except Exception:
                # Fallback a Close si no se encuentra price_col
                data = raw.Close if "Close" in raw.columns else raw

        if isinstance(data, pd.Series):
            data = data.to_frame()
        self.data = data.dropna(how='all')

    

    def compute_returns(self, kind: str = 'log', verbose: bool = False) -> pd.DataFrame:
        """
        Calcula los retornos sean simples o logretornos.
        Si verbose=True, imprime un mensaje al crearlos.
        """
        if kind == 'log':
            self.rets = np.log(self.data / self.data.shift(1))
        elif kind == 'simple':
            self.rets = self.data.pct_change()
        else:
            raise ValueError('kind  debe ser log o simple')
        self.rets.dropna(inplace=True)
        if verbose:
            print(f'*** retornos tipo {kind} creados ***')
        return self.rets



    @classmethod
    def download_data(cls,
                      tickers: Union[List[str], str],
                      start: str,
                      end: Optional[str],
                      price_col: str = "Adj Close",
                      auto_adjust: bool = False) -> pd.DataFrame:
        """
        Descarga precios (por defecto 'Adj Close') y retorna DataFrame limpio.
        Emula la función download_data de tu compa, pero encapsulada en la clase.
        """
        data = yf.download(tickers, start=start, end=end,
                           progress=False, auto_adjust=auto_adjust)
        if isinstance(data, pd.DataFrame) and price_col in data.columns:
            data = data[price_col]
        else:
            # Manejo de MultiIndex (caso común de yfinance con múltiples tickers)
            try:
                data = data.xs(price_col, axis=1, level=0, drop_level=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.droplevel(0, axis=1)
            except Exception:
                # Fallback: Close
                data = data.Close if "Close" in data.columns else data

        if isinstance(data, pd.Series):  # si es un solo ticker
            data = data.to_frame()
        return data.dropna(how="all")

    @classmethod
    def preprocess_data(cls,
                        tickers: List[str],
                        rf_ticker: Union[List[str], str],
                        start: str,
                        end: Optional[str],
                        price_col: str = "Adj Close",
                        auto_adjust: bool = False) -> pd.DataFrame:
        """
        Descarga activos + ^IRX, alinea calendarios (inner join) y elimina faltantes.
        Devuelve un único DataFrame: columnas = tickers + rf_ticker(s).
        """
        prices_assets = cls.download_data(tickers, start, end,
                                          price_col=price_col, auto_adjust=auto_adjust)
        prices_rf = cls.download_data(rf_ticker, start, end,
                                      price_col=price_col, auto_adjust=auto_adjust)

        # Alinear por intersección de fechas y limpiar filas con NAs
        data_all = prices_assets.join(prices_rf, how="inner").dropna(how="any")
        # Validaciones ligeras
        if not data_all.index.is_monotonic_increasing:
            data_all = data_all.sort_index()
        data_all = data_all[~data_all.index.duplicated(keep="first")]
        return data_all

    # conversión de ^IRX

    @staticmethod
    def convert_rf(data_rf: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """
        Convierte ^IRX (tasa anualizada en %) a:
          - tasa anual (proporción)
          - tasa diaria efectiva
        Añade sufijos '_ann' y '_daily' para claridad.
        """
        rf_ann = data_rf / 100.0
        rf_daily = (1.0 + rf_ann) ** (1.0 / 252.0) - 1.0

        rf_ann = rf_ann.copy()
        rf_daily = rf_daily.copy()

        rf_ann.columns = [f"{c}_ann" for c in rf_ann.columns]
        rf_daily.columns = [f"{c}_daily" for c in rf_daily.columns]
        return rf_ann, rf_daily

    # retornos y excesos

    @staticmethod
    def compute_returns_from_prices(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
        """
        Calcula retornos desde un DataFrame de precios (sin depender de self.data).
        Evita colisionar con tu método de instancia compute_returns().
        """
        if kind == "log":
            rets = np.log(prices / prices.shift(1))
        elif kind == "simple":
            rets = prices.pct_change()
        else:
            raise ValueError("kind debe ser 'log' o 'simple'")
        return rets.dropna()

    @staticmethod
    def excess_returns(assets_rets: pd.DataFrame, rf_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula retornos en exceso: activos - rf diaria (alineada y con ffill).
        """
        rf_aligned = rf_daily.reindex(assets_rets.index).fillna(method="ffill")
        # Si rf_daily viene con una sola columna, aplastar para broadcast correcto:
        if isinstance(rf_aligned, pd.DataFrame) and rf_aligned.shape[1] == 1:
            rf_series = rf_aligned.iloc[:, 0]
        else:
            # Si vinieran varias columnas de rf, usar la primera por convención
            rf_series = rf_aligned.iloc[:, 0]
        ex_rets = assets_rets.sub(rf_series, axis=0)
        return ex_rets
