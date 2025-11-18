import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

from arch import arch_model
from sklearn.metrics import mean_squared_error


class TimeSeriesARMA:
    """
    Clase para análisis exploratorio de series de tiempo al estilo de tu notebook:
    - Gráficas de la serie
    - Test ADF (Dickey-Fuller)
    - Rolling mean / std
    - ACF y PACF
    - Descomposición (tendencia, estacionalidad, residuo)
    - Ajuste de SARIMAX y diagnóstico de residuales

    La idea está alineada con la teoría de ARMA/ARIMA (Box-Jenkins) y con tus
    notas de TimeSeries/ARMA: primero revisamos estacionariedad, luego ACF/PACF,
    luego modelo SARIMAX, luego revisamos residuales.
    """

    def __init__(self, series: pd.Series, name: str | None = None):
        if not isinstance(series, pd.Series):
            raise TypeError("series debe ser un pandas.Series")
        self.series = series.dropna()
        self.name = name or (series.name if series.name is not None else "serie")
        self.sarimax_res = None  # aquí guardaremos el ajuste SARIMAX

    # ------------------------------
    # Visualización básica
    # ------------------------------
    def plot_series(self):
        """Grafica la serie con grid, similar a tus ejemplos con seaborn."""
        plt.figure(figsize=(10, 4))
        sns.lineplot(x=self.series.index, y=self.series.values)
        plt.grid(ls="--")
        plt.ylabel(self.name)
        plt.title(f"Serie: {self.name}")
        plt.tight_layout()
        plt.show()

    # ------------------------------
    # Test de Dickey-Fuller Aumentado
    # ------------------------------
    def adf_test(self, autolag: str = "AIC") -> pd.DataFrame:
        """
        Prueba de Dickey-Fuller Aumentada (ADF).
        H0: la serie NO es estacionaria.

        Devuelve un DataFrame con el mismo formato que usaste en tus ejemplos.
        """
        adft = adfuller(self.series, autolag=autolag)
        output_df = pd.DataFrame(
            {
                "Values": [
                    adft[0],
                    adft[1],
                    adft[2],
                    adft[3],
                    adft[4]["1%"],
                    adft[4]["5%"],
                    adft[4]["10%"],
                ],
                "Metrics": [
                    "Test Statistics",
                    "p-value",
                    "No. of lags used",
                    "No. obs used",
                    "Critical Value (1%)",
                    "Critical Value (5%)",
                    "Critical Value (10%)",
                ],
            }
        )
        print("Resultados prueba Dickey-Fuller Aumentada:\n")
        print(output_df)
        return output_df

    # ------------------------------
    # Rolling mean / std (suavizado)
    # ------------------------------
    def plot_rolling(self, q: int = 7):
        """
        Grafica la serie original, la media móvil y la desviación estándar móvil,
        como hiciste con q=7, 30, etc.
        """
        rolling_mean = self.series.rolling(q).mean()
        rolling_std = self.series.rolling(q).std()

        plt.figure(figsize=(10, 4))
        plt.plot(self.series, color="blue", label="Original")
        plt.plot(rolling_mean, color="red", label="Rolling mean")
        plt.plot(rolling_std, color="black", label="Rolling Std")
        plt.grid(ls="--")
        plt.legend()
        plt.title(f"{self.name} - Rolling (q={q})")
        plt.tight_layout()
        plt.show()

    # ------------------------------
    # ACF y PACF
    # ------------------------------
    def plot_acf_pacf(self, lags: int = 20):
        """
        Grafica ACF y PACF, como en tus ejemplos.
        Nota: la teoría estándar (Box-Jenkins) recomienda estacionariedad,
        pero se puede mirar igual como hiciste tú.
        """
        plt.figure(figsize=(10, 4))
        plot_acf(self.series, lags=lags)
        plt.title(f"ACF - {self.name}")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plot_pacf(self.series, lags=lags)
        plt.title(f"PACF - {self.name}")
        plt.tight_layout()
        plt.show()

    # ------------------------------
    # Descomposición (tendencia/estación/ruido)
    # ------------------------------
    def decompose(self, model: str = "additive", period: int = 12):
        """
        Descompone la serie en tendencia, estacionalidad y residuo,
        igual que hiciste con AirPassengers.

        model : 'additive' o 'multiplicative'
        period: periodo estacional (7, 12, etc.)
        """
        decomp = seasonal_decompose(self.series, model=model, period=period)

        # Gráfica estándar de statsmodels
        decomp.plot()
        plt.suptitle(f"Descomposición ({model}, period={period}) - {self.name}")
        plt.tight_layout()
        plt.show()

        # Residuales
        plt.figure(figsize=(10, 4))
        decomp.resid.plot()
        plt.title(f"Residuales de descomposición - {self.name}")
        plt.grid(ls="--")
        plt.tight_layout()
        plt.show()

        # ACF de residuales
        resid_clean = decomp.resid.dropna()
        plt.figure(figsize=(10, 4))
        plot_acf(resid_clean)
        plt.title(f"ACF de residuales - {self.name}")
        plt.tight_layout()
        plt.show()

        return decomp

    # ------------------------------
    # Ajuste SARIMAX + diagnóstico de residuales
    # ------------------------------
    def fit_sarimax(
        self,
        order: tuple[int, int, int] = (0, 0, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str = "n",
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ):
        """
        Ajusta un modelo SARIMAX a la serie, tal como hiciste con:
        SARIMAX(..., order=(p,0,q), seasonal_order=(0,0,0,0), trend='n', ...)

        Devuelve el resultado ajustado y lo guarda en self.sarimax_res.
        """
        model = SARIMAX(
            self.series,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )
        res = model.fit(disp=False)
        self.sarimax_res = res
        print(res.summary())
        return res

    def sarimax_residual_diagnostics(self, lags_lb: int = 12):
        """
        Hace diagnóstico de residuales del modelo SARIMAX:
        - Ljung-Box
        - ACF de residuales

        Similar a lo que hiciste con acorr_ljungbox + plot_acf(res.resid).
        """
        if self.sarimax_res is None:
            raise ValueError("Primero ajusta un modelo con fit_sarimax().")

        resid = self.sarimax_res.resid.dropna()

        lb = acorr_ljungbox(resid, lags=lags_lb, return_df=True)
        lb.columns = ["LB_stat", "LB_pvalue"]
        print("Ljung-Box sobre residuales:\n")
        print(lb)

        plt.figure(figsize=(10, 4))
        plot_acf(resid)
        plt.title(f"ACF de residuales SARIMAX - {self.name}")
        plt.tight_layout()
        plt.show()

        return lb

    # ------------------------------
    # Simulador ARMA (tu función) como método estático
    # ------------------------------
    @staticmethod
    def simulate_arma(
        n: int,
        phi=None,
        theta=None,
        mu: float = 0.0,
        sigma: float = 1.0,
        burn: int = 300,
        seed: int | None = None,
        as_series: bool = True,
    ):
        """
        Simula un ARMA(p,q) como en tu código:

        Y_t = c + sum_{i=1}^p phi_i * Y_{t-i} + e_t + sum_{j=1}^q theta_j * e_{t-j},
        con e_t ~ N(0, sigma^2), c = mu*(1 - sum(phi)).

        Parámetros iguales a tu versión original.
        """
        np.random.seed(seed)
        phi = np.asarray(phi) if phi is not None else np.array([])
        theta = np.asarray(theta) if theta is not None else np.array([])
        p = len(phi)
        q = len(theta)

        # Intercepto para tener media = mu
        c = mu * (1.0 - np.sum(phi))
        T = n + burn
        y = np.zeros(T)
        e = np.random.normal(loc=0.0, scale=sigma, size=T)

        # Simulación por recursión (igual que tu implementación)
        for t in range(max(p, q) + 5, T):
            ar_part = 0.0
            if p > 0:
                ar_part = np.dot(phi, y[t - np.arange(1, p + 1)])

            ma_part = 0.0
            if q > 0:
                ma_part = np.dot(theta, e[t - np.arange(1, q + 1)])

            y[t] = c + ar_part + e[t] + ma_part

        # Descartar burn-in
        y = y[max(p, q) + burn : max(p, q) + burn + n]
        if as_series:
            return pd.Series(y, name="arma_sim")
        return y
