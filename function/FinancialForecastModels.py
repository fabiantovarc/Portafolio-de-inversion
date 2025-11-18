import TimeSeriesARMA
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


class FinancialForecastModels(TimeSeriesARMA):
    """
    Extiende TimeSeriesARMA para el caso financiero:
    - Ajustar ARIMA a log-retornos
    - Ajustar SARIMAX con estacionalidad
    - Ajustar GARCH(1,1) a la volatilidad
    - Comparar pronósticos (ARIMA, SARIMAX, GARCH) vía MSE y gráficos

    Está en línea con el enfoque de series financieras tipo:
    - log-retornos ~ ARIMA/SARIMAX
    - volatilidad ~ GARCH(1,1)
    """

    def __init__(self, series: pd.Series, name: str | None = None):
        super().__init__(series, name=name)
        self.arima_res = None
        self.sarimax_res = None  # ya existe en la clase padre, pero lo dejamos claro
        self.garch_res = None

    # ------------------------------
    # ARIMA
    # ------------------------------
    def fit_arima(self, order: tuple[int, int, int] = (1, 1, 1)):
        """
        Ajusta un modelo ARIMA(p,d,q) a la serie de log-retornos,
        igual que tu ejemplo con ARIMA(1,1,1).
        """
        model_arima = ARIMA(self.series, order=order)
        self.arima_res = model_arima.fit()
        print(self.arima_res.summary())
        return self.arima_res

    def forecast_arima(self, steps: int = 10) -> pd.Series:
        if self.arima_res is None:
            raise ValueError("Primero ajusta ARIMA con fit_arima().")
        fc = self.arima_res.forecast(steps=steps)
        return fc

    # ------------------------------
    # SARIMAX financiero (por ejemplo, estacionalidad mensual)
    # ------------------------------
    def fit_sarimax_model(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12),
    ):
        """
        Ajusta un SARIMAX con estacionalidad, como en tu ejemplo:
        SARIMAX(log_returns, order=(1,1,1), seasonal_order=(1,1,1,12)).
        """
        model_sarimax = SARIMAX(
            self.series,
            order=order,
            seasonal_order=seasonal_order,
        )
        self.sarimax_res = model_sarimax.fit(disp=False)
        print(self.sarimax_res.summary())
        return self.sarimax_res

    def forecast_sarimax(self, steps: int = 10) -> pd.Series:
        if self.sarimax_res is None:
            raise ValueError("Primero ajusta SARIMAX con fit_sarimax_model().")
        fc = self.sarimax_res.forecast(steps=steps)
        return fc

    # ------------------------------
    # GARCH(1,1) sobre la varianza condicional
    # ------------------------------
    def fit_garch(self, p: int = 1, q: int = 1):
        """
        Ajusta un modelo GARCH(p,q) a los log-retornos, como en tu ejemplo:

        model_garch = arch_model(log_returns, vol="Garch", p=1, q=1)
        """
        model_garch = arch_model(self.series, vol="Garch", p=p, q=q)
        self.garch_res = model_garch.fit(disp="off")
        print(self.garch_res.summary())
        return self.garch_res

    def forecast_garch_mean(self, horizon: int = 10) -> pd.Series:
        """
        Toma el pronóstico de varianza condicional y devuelve la media
        (como en tu código, donde usas forecast_garch.variance[-1:]),
        reescalada a la misma escala de log-retornos.
        """
        if self.garch_res is None:
            raise ValueError("Primero ajusta GARCH con fit_garch().")

        forecast_garch = self.garch_res.forecast(horizon=horizon)
        # Tomamos la varianza pronosticada en el último tiempo
        var_fc = forecast_garch.variance.iloc[-1]
        # En tu ejemplo dividías por 100; lo dejo opcional según cómo quieras escalar.
        mean_fc = var_fc / 100.0
        mean_fc.name = "garch_mean_forecast"
        return mean_fc

    # ------------------------------
    # Comparación de pronósticos (MSE + gráfico conjunto)
    # ------------------------------
    def compare_forecasts(self, horizon: int = 10):
        """
        Compara ARIMA, SARIMAX y GARCH usando MSE sobre los últimos `horizon`
        datos reales, similar a tu script final.

        Asume que ya has ajustado:
        - ARIMA (fit_arima)
        - SARIMAX (fit_sarimax_model)
        - GARCH (fit_garch)
        """
        if self.arima_res is None or self.sarimax_res is None or self.garch_res is None:
            raise ValueError(
                "Necesitas ajustar ARIMA, SARIMAX y GARCH antes de comparar pronósticos."
            )

        true_values = self.series[-horizon:]

        # Pronósticos
        fc_arima = self.arima_res.forecast(steps=horizon)
        fc_sarimax = self.sarimax_res.forecast(steps=horizon)
        forecast_garch = self.garch_res.forecast(horizon=horizon)
        mean_forecast = forecast_garch.variance.iloc[-1] / 100.0

        # Aseguramos dimensiones y usamos valores
        y_true = true_values.values
        y_arima = fc_arima.values
        y_sarimax = fc_sarimax.values
        y_garch = mean_forecast.values

        mse_arima = mean_squared_error(y_true, y_arima)
        mse_sarimax = mean_squared_error(y_true, y_sarimax)
        mse_garch = mean_squared_error(y_true, y_garch)

        print("MSE ARIMA:", mse_arima)
        print("MSE SARIMAX:", mse_sarimax)
        print("MSE GARCH:", mse_garch)

        # Gráfico comparativo
        plt.figure(figsize=(12, 6))
        plt.plot(true_values.index, true_values.values, "o-", label="Últimos reales")
        plt.plot(true_values.index, y_arima, "r--", label="ARIMA Pronóstico")
        plt.plot(true_values.index, y_sarimax, "g--", label="SARIMAX Pronóstico")
        plt.plot(true_values.index, y_garch, "b--", label="GARCH Pronóstico (media)")
        plt.title(f"Comparación de Pronósticos: ARIMA, SARIMAX y GARCH - {self.name}")
        plt.legend()
        plt.grid(ls="--")
        plt.tight_layout()
        plt.show()

        return {
            "mse_arima": mse_arima,
            "mse_sarimax": mse_sarimax,
            "mse_garch": mse_garch,
        }
