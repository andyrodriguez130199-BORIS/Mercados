"""
Markowitz Portfolio Optimization
=================================
Descarga datos históricos de 75 activos en 5 mercados globales, ejecuta una
simulación Monte Carlo de 100,000 portafolios, extrae el portafolio de Máximo
Sharpe (Max Sharpe) y el de Mínima Varianza Global (GMV), filtra y normaliza
los pesos, exporta resultados a Excel (multi-hoja) y genera múltiples gráficas
de visualización de la Frontera Eficiente.

Mercados incluidos
------------------
  - Japón    (Nikkei 225)  — 15 tickers
  - Alemania (DAX 40)      — 15 tickers
  - Reino Unido (FTSE 100) — 15 tickers
  - Hong Kong (Hang Seng)  — 15 tickers
  - Francia  (CAC 40)      — 15 tickers
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
MARKETS = {
    "Japón (Nikkei 225)": [
        "8035.T", "6758.T", "9984.T", "6857.T", "6702.T",
        "6503.T", "4689.T", "7203.T", "6301.T", "7011.T",
        "9101.T", "6501.T", "7267.T", "6902.T", "1801.T",
    ],
    "Alemania (DAX 40)": [
        "SAP.DE", "IFX.DE", "AFX.DE", "WCH.DE", "NEM.DE",
        "VAR1.DE", "SOW.DE", "SIE.DE", "DHL.DE", "VOW3.DE",
        "MBG.DE", "MTX.DE", "RHM.DE", "CON.DE", "HEI.DE",
    ],
    "Reino Unido (FTSE 100)": [
        "SGE.L", "OCDO.L", "RMV.L", "AUTO.L", "DARK.L",
        "SPT.L", "BAE.L", "RR.L", "IAG.L", "FGP.L",
        "MGG.L", "MEL.L", "WEIR.L", "RKT.L", "SMDS.L",
    ],
    "Hong Kong (Hang Seng)": [
        "0700.HK", "9988.HK", "3690.HK", "9888.HK", "1810.HK",
        "0285.HK", "1347.HK", "1211.HK", "0669.HK", "1928.HK",
        "0001.HK", "0016.HK", "0066.HK", "0386.HK", "1088.HK",
    ],
    "Francia (CAC 40)": [
        "CAP.PA", "STMPA.PA", "DSY.PA", "ATO.PA", "WLN.PA",
        "SESL.PA", "ORA.PA", "AIR.PA", "SAF.PA", "DG.PA",
        "SCHP.PA", "ALO.PA", "ENGI.PA", "SU.PA", "TTE.PA",
    ],
}

ALL_TICKERS = [t for tickers in MARKETS.values() for t in tickers]  # 75 tickers

# Periodos de descarga (de mayor a menor); se usa el primero que devuelva datos
PERIODS_YEARS = [10, 5, 3, 2]
RISK_FREE_RATE = 0.04       # Tasa libre de riesgo anual (4 %)
N_SIMULATIONS = 100_000     # Número de portafolios Monte Carlo
WEIGHT_THRESHOLD = 0.001    # Umbral de viabilidad (0.1 %)
TRADING_DAYS = 252          # Días hábiles por año
OUTPUT_EXCEL = "Resultados_Markowitz.xlsx"
OUTPUT_FRONTIER = "Frontera_Eficiente.png"
OUTPUT_WEIGHTS_MS = "Pesos_MaxSharpe.png"
OUTPUT_WEIGHTS_GMV = "Pesos_GMV.png"
OUTPUT_COMPARISON = "Comparacion_Portafolios.png"

np.random.seed(42)  # Reproducibilidad

# ─────────────────────────────────────────────────────────────────────────────
# 1. DESCARGA DE DATOS HISTÓRICOS
# ─────────────────────────────────────────────────────────────────────────────

def download_prices(tickers: list[str], periods_years: list[int]) -> pd.DataFrame:
    """
    Descarga precios de cierre ajustados para la lista de tickers.
    Intenta periodos en orden descendente y conserva el más largo
    disponible para cada ticker.  Retorna un DataFrame con la fecha
    como índice y los tickers como columnas.
    """
    today = date.today()
    best_data: dict[str, pd.Series] = {}

    print("\n📥  Descargando datos históricos …")
    for ticker in tickers:
        series = None
        for years in periods_years:
            start = (today - relativedelta(years=years)).isoformat()
            try:
                raw = yf.download(
                    ticker,
                    start=start,
                    end=today.isoformat(),
                    auto_adjust=True,
                    progress=False,
                )
                if raw.empty:
                    continue
                # yfinance puede devolver MultiIndex si hay un solo ticker
                if isinstance(raw.columns, pd.MultiIndex):
                    close = raw["Close"][ticker]
                else:
                    close = raw["Close"]
                if len(close.dropna()) >= 60:   # al menos 60 observaciones
                    series = close.dropna()
                    break
            except Exception:
                continue
        if series is not None and len(series) >= 60:
            best_data[ticker] = series
        else:
            print(f"   ⚠️  Sin datos suficientes para {ticker}; se omite.")

    if not best_data:
        raise RuntimeError("No se pudo descargar datos para ningún ticker.")

    prices = pd.DataFrame(best_data)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Conservar solo filas sin NaN (sincronización)
    prices = prices.dropna()
    print(f"   ✅  Datos sincronizados: {len(prices)} observaciones × {prices.shape[1]} activos.")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# 2. CÁLCULO DE RETORNOS Y MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos logarítmicos diarios."""
    return np.log(prices / prices.shift(1)).dropna()


def annualized_metrics(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna:
        mean_returns : vector de retornos esperados anualizados (n_assets,)
        cov_matrix   : matriz de covarianza anualizada (n_assets, n_assets)
    """
    mean_ret = returns.mean() * TRADING_DAYS
    cov_mat = returns.cov() * TRADING_DAYS
    return mean_ret.values, cov_mat.values


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIMULACIÓN MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_assets: int,
    n_sims: int,
    rf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera n_sims portafolios aleatorios y calcula sus métricas.

    Retorna:
        weights      : (n_sims, n_assets)
        port_returns : (n_sims,)
        port_vols    : (n_sims,)
        sharpe_ratios: (n_sims,)
    """
    print(f"\n⚙️   Ejecutando simulación Monte Carlo ({n_sims:,} portafolios) …")

    weights_all = np.zeros((n_sims, n_assets))
    port_returns = np.zeros(n_sims)
    port_vols = np.zeros(n_sims)
    sharpe_ratios = np.zeros(n_sims)

    for i in range(n_sims):
        w = np.random.dirichlet(np.ones(n_assets))  # pesos que suman 1
        weights_all[i] = w
        ret = np.dot(w, mean_returns)
        vol = np.sqrt(w @ cov_matrix @ w)
        port_returns[i] = ret
        port_vols[i] = vol
        sharpe_ratios[i] = (ret - rf) / vol if vol > 0 else 0.0

    print("   ✅  Simulación completada.")
    return weights_all, port_returns, port_vols, sharpe_ratios


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXTRACCIÓN DE PORTAFOLIOS ÓPTIMOS
# ─────────────────────────────────────────────────────────────────────────────

def extract_optimal_portfolios(
    weights_all: np.ndarray,
    port_returns: np.ndarray,
    port_vols: np.ndarray,
    sharpe_ratios: np.ndarray,
) -> dict:
    """
    Ubica los índices del Max Sharpe y del GMV usando argmax / argmin
    y extrae las métricas y pesos de cada uno.
    """
    # ── índices
    idx_ms = int(np.argmax(sharpe_ratios))   # Máximo Sharpe Ratio
    idx_gmv = int(np.argmin(port_vols))      # Global Minimum Variance

    print(f"\n📌  Índice Max Sharpe : {idx_ms}")
    print(f"📌  Índice GMV        : {idx_gmv}")

    return {
        "max_sharpe": {
            "index": idx_ms,
            "return": port_returns[idx_ms],
            "volatility": port_vols[idx_ms],
            "sharpe": sharpe_ratios[idx_ms],
            "weights": weights_all[idx_ms],
        },
        "gmv": {
            "index": idx_gmv,
            "return": port_returns[idx_gmv],
            "volatility": port_vols[idx_gmv],
            "sharpe": sharpe_ratios[idx_gmv],
            "weights": weights_all[idx_gmv],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. FILTRO DE VIABILIDAD Y RE-NORMALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def filter_and_normalize(weights: np.ndarray, threshold: float = WEIGHT_THRESHOLD) -> np.ndarray:
    """
    Aplica el umbral de viabilidad:
      1. Pone a 0 los pesos inferiores al umbral.
      2. Re-normaliza para que la suma sea exactamente 1.0.
    """
    filtered = np.where(weights < threshold, 0.0, weights)
    total = filtered.sum()
    if total == 0:
        raise ValueError("Todos los pesos quedaron en 0 tras el filtro.")
    return filtered / total


# ─────────────────────────────────────────────────────────────────────────────
# 6. ESTRUCTURACIÓN EN DATAFRAMES
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframes(
    tickers: list[str],
    optimal: dict,
    weights_ms_norm: np.ndarray,
    weights_gmv_norm: np.ndarray,
    corr_matrix: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Construye todos los DataFrames listos para exportar a Excel.
    """
    # ── Tabla de Métricas
    metrics_df = pd.DataFrame(
        {
            "Portafolio": ["Max Sharpe", "GMV"],
            "Rendimiento Esperado (%)": [
                round(optimal["max_sharpe"]["return"] * 100, 4),
                round(optimal["gmv"]["return"] * 100, 4),
            ],
            "Volatilidad (%)": [
                round(optimal["max_sharpe"]["volatility"] * 100, 4),
                round(optimal["gmv"]["volatility"] * 100, 4),
            ],
            "Ratio de Sharpe": [
                round(optimal["max_sharpe"]["sharpe"], 6),
                round(optimal["gmv"]["sharpe"], 6),
            ],
        }
    ).set_index("Portafolio")

    # ── Tabla Max Sharpe
    df_ms = pd.DataFrame({"Ticker": tickers, "Peso": weights_ms_norm})
    df_ms["Peso (%)"] = (df_ms["Peso"] * 100).round(4)
    df_ms = df_ms.sort_values("Peso", ascending=False)
    df_ms = df_ms[df_ms["Peso"] > 0].reset_index(drop=True)
    df_ms.drop(columns=["Peso"], inplace=True)

    # ── Tabla GMV
    df_gmv = pd.DataFrame({"Ticker": tickers, "Peso": weights_gmv_norm})
    df_gmv["Peso (%)"] = (df_gmv["Peso"] * 100).round(4)
    df_gmv = df_gmv.sort_values("Peso", ascending=False)
    df_gmv = df_gmv[df_gmv["Peso"] > 0].reset_index(drop=True)
    df_gmv.drop(columns=["Peso"], inplace=True)

    # ── Correlación
    corr_df = corr_matrix.round(4)

    return {
        "Resumen_Metricas": metrics_df,
        "Pesos_Max_Sharpe": df_ms,
        "Pesos_GMV": df_gmv,
        "Matriz_Correlacion": corr_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPORTACIÓN A EXCEL (MULTI-HOJA)
# ─────────────────────────────────────────────────────────────────────────────

def export_to_excel(dataframes: dict[str, pd.DataFrame], filename: str) -> None:
    """Escribe cada DataFrame en una hoja separada del archivo Excel."""
    print(f"\n💾  Exportando resultados a '{filename}' …")
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=True)
    print("   ✅  Archivo Excel guardado correctamente.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────

def plot_efficient_frontier(
    port_vols: np.ndarray,
    port_returns: np.ndarray,
    sharpe_ratios: np.ndarray,
    optimal: dict,
    filename: str = OUTPUT_FRONTIER,
) -> None:
    """
    Gráfica de dispersión de la Frontera Eficiente (nube de portafolios)
    con marcadores destacados para Max Sharpe y GMV.
    """
    fig, ax = plt.subplots(figsize=(13, 8))

    scatter = ax.scatter(
        port_vols * 100,
        port_returns * 100,
        c=sharpe_ratios,
        cmap="viridis",
        alpha=0.4,
        s=3,
        linewidths=0,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Ratio de Sharpe", fontsize=11)

    # ── Marcador Max Sharpe
    ax.scatter(
        optimal["max_sharpe"]["volatility"] * 100,
        optimal["max_sharpe"]["return"] * 100,
        marker="*",
        color="#FFD700",
        edgecolors="black",
        s=600,
        zorder=5,
        label=(
            f"Max Sharpe  |  SR={optimal['max_sharpe']['sharpe']:.4f}  "
            f"R={optimal['max_sharpe']['return']*100:.2f}%  "
            f"σ={optimal['max_sharpe']['volatility']*100:.2f}%"
        ),
    )

    # ── Marcador GMV
    ax.scatter(
        optimal["gmv"]["volatility"] * 100,
        optimal["gmv"]["return"] * 100,
        marker="D",
        color="#FF4444",
        edgecolors="black",
        s=250,
        zorder=5,
        label=(
            f"GMV  |  SR={optimal['gmv']['sharpe']:.4f}  "
            f"R={optimal['gmv']['return']*100:.2f}%  "
            f"σ={optimal['gmv']['volatility']*100:.2f}%"
        ),
    )

    ax.set_xlabel("Volatilidad Anual (%)", fontsize=13)
    ax.set_ylabel("Rendimiento Esperado Anual (%)", fontsize=13)
    ax.set_title(
        "Frontera Eficiente de Markowitz\n(100,000 portafolios — 5 mercados globales)",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"   📊  Gráfica '{filename}' guardada.")


def plot_weights_bar(
    df_weights: pd.DataFrame,
    title: str,
    color: str,
    filename: str,
    top_n: int = 20,
) -> None:
    """
    Gráfica de barras horizontales con los pesos más significativos
    de un portafolio.  Muestra hasta top_n activos.
    """
    df_plot = df_weights.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, max(5, len(df_plot) * 0.4 + 1)))

    bars = ax.barh(df_plot["Ticker"], df_plot["Peso (%)"], color=color, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Peso en el Portafolio (%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"   📊  Gráfica '{filename}' guardada.")


def plot_weights_pie(
    df_weights: pd.DataFrame,
    title: str,
    filename: str,
    top_n: int = 15,
) -> None:
    """
    Gráfica de pastel con los pesos del portafolio.
    Los activos menores al (100/top_n)% se agrupan en 'Otros'.
    """
    df_plot = df_weights.head(top_n).copy()
    other_pct = df_weights["Peso (%)"].iloc[top_n:].sum() if len(df_weights) > top_n else 0.0

    labels = list(df_plot["Ticker"])
    sizes = list(df_plot["Peso (%)"])
    if other_pct > 0:
        labels.append("Otros")
        sizes.append(other_pct)

    colors = cm.tab20.colors[: len(labels)]
    explode = [0.04] * len(labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        explode=explode,
        startangle=140,
        pctdistance=0.82,
    )
    for t in texts:
        t.set_fontsize(9)
    for at in autotexts:
        at.set_fontsize(8)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"   📊  Gráfica '{filename}' guardada.")


def plot_comparison(
    df_ms: pd.DataFrame,
    df_gmv: pd.DataFrame,
    optimal: dict,
    filename: str = OUTPUT_COMPARISON,
) -> None:
    """
    Panel de comparación lado a lado:
      - Métricas (retorno, volatilidad, Sharpe) para ambos portafolios.
      - Top 10 activos de cada portafolio como barras horizontales.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    ax_metrics = fig.add_subplot(gs[0, :])
    ax_ms = fig.add_subplot(gs[1, 0])
    ax_gmv = fig.add_subplot(gs[1, 1])

    # ── Panel de métricas (tabla)
    metrics_data = [
        ["Max Sharpe",
         f"{optimal['max_sharpe']['return']*100:.2f}%",
         f"{optimal['max_sharpe']['volatility']*100:.2f}%",
         f"{optimal['max_sharpe']['sharpe']:.4f}"],
        ["GMV",
         f"{optimal['gmv']['return']*100:.2f}%",
         f"{optimal['gmv']['volatility']*100:.2f}%",
         f"{optimal['gmv']['sharpe']:.4f}"],
    ]
    col_labels = ["Portafolio", "Rendimiento", "Volatilidad", "Sharpe"]
    ax_metrics.axis("off")
    tbl = ax_metrics.table(
        cellText=metrics_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2.2)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == 1:
            cell.set_facecolor("#f0c040")
        else:
            cell.set_facecolor("#fde8e8")
    ax_metrics.set_title("Métricas de los Portafolios Óptimos", fontsize=13, fontweight="bold", pad=10)

    # ── Barras Top 10 Max Sharpe
    top10_ms = df_ms.head(10)
    ax_ms.barh(top10_ms["Ticker"], top10_ms["Peso (%)"], color="#f0c040", edgecolor="white")
    ax_ms.invert_yaxis()
    ax_ms.set_xlabel("Peso (%)")
    ax_ms.set_title("Top 10 — Max Sharpe", fontweight="bold")
    ax_ms.grid(axis="x", alpha=0.3)

    # ── Barras Top 10 GMV
    top10_gmv = df_gmv.head(10)
    ax_gmv.barh(top10_gmv["Ticker"], top10_gmv["Peso (%)"], color="#e74c3c", edgecolor="white")
    ax_gmv.invert_yaxis()
    ax_gmv.set_xlabel("Peso (%)")
    ax_gmv.set_title("Top 10 — GMV", fontweight="bold")
    ax_gmv.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Análisis Comparativo de Portafolios Óptimos",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊  Gráfica '{filename}' guardada.")


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    filename: str = "Mapa_Correlacion.png",
) -> None:
    """
    Mapa de calor de la matriz de correlación global.
    """
    n = len(corr_matrix)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.3), max(10, n * 0.3)))
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04).set_label("Correlación", fontsize=10)

    if n <= 75:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(corr_matrix.index, fontsize=7)

    ax.set_title("Matriz de Correlación — 75 Activos Globales", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close()
    print(f"   📊  Gráfica '{filename}' guardada.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  OPTIMIZACIÓN DE PORTAFOLIO DE MARKOWITZ — 5 MERCADOS GLOBALES")
    print("=" * 65)

    # ── 1. Descarga de precios
    prices = download_prices(ALL_TICKERS, PERIODS_YEARS)
    tickers_ok = list(prices.columns)
    n_assets = len(tickers_ok)
    print(f"\n   Activos con datos válidos: {n_assets}")

    if n_assets < 5:
        raise RuntimeError("Insuficientes activos para la optimización (mínimo 5).")

    # ── 2. Retornos y métricas anualizadas
    returns = compute_returns(prices)
    mean_returns, cov_matrix = annualized_metrics(returns)
    corr_matrix = returns.corr()

    # ── 3. Simulación Monte Carlo
    weights_all, port_returns, port_vols, sharpe_ratios = monte_carlo(
        mean_returns, cov_matrix, n_assets, N_SIMULATIONS, RISK_FREE_RATE
    )

    # ── 4. Extracción de portafolios óptimos (argmax / argmin)
    optimal = extract_optimal_portfolios(weights_all, port_returns, port_vols, sharpe_ratios)

    print(f"\n   Max Sharpe → R={optimal['max_sharpe']['return']*100:.2f}%  "
          f"σ={optimal['max_sharpe']['volatility']*100:.2f}%  "
          f"SR={optimal['max_sharpe']['sharpe']:.4f}")
    print(f"   GMV        → R={optimal['gmv']['return']*100:.2f}%  "
          f"σ={optimal['gmv']['volatility']*100:.2f}%  "
          f"SR={optimal['gmv']['sharpe']:.4f}")

    # ── 5. Filtro de viabilidad y re-normalización
    weights_ms_norm = filter_and_normalize(optimal["max_sharpe"]["weights"], WEIGHT_THRESHOLD)
    weights_gmv_norm = filter_and_normalize(optimal["gmv"]["weights"], WEIGHT_THRESHOLD)

    print(f"\n   Activos activos Max Sharpe: {(weights_ms_norm > 0).sum()}")
    print(f"   Activos activos GMV       : {(weights_gmv_norm > 0).sum()}")

    # ── 6. Construcción de DataFrames
    dataframes = build_dataframes(
        tickers_ok, optimal, weights_ms_norm, weights_gmv_norm, corr_matrix
    )

    # ── 7. Exportación a Excel
    export_to_excel(dataframes, OUTPUT_EXCEL)

    # ── 8. Visualizaciones
    print("\n🎨  Generando gráficas …")

    plot_efficient_frontier(port_vols, port_returns, sharpe_ratios, optimal)

    plot_weights_bar(
        dataframes["Pesos_Max_Sharpe"],
        "Distribución de Pesos — Portafolio Max Sharpe (Top 20)",
        "#f0c040",
        OUTPUT_WEIGHTS_MS,
    )

    plot_weights_bar(
        dataframes["Pesos_GMV"],
        "Distribución de Pesos — Portafolio GMV (Top 20)",
        "#e74c3c",
        OUTPUT_WEIGHTS_GMV,
    )

    plot_weights_pie(
        dataframes["Pesos_Max_Sharpe"],
        "Composición del Portafolio Max Sharpe",
        "Pesos_MaxSharpe_Pie.png",
    )

    plot_weights_pie(
        dataframes["Pesos_GMV"],
        "Composición del Portafolio GMV",
        "Pesos_GMV_Pie.png",
    )

    plot_comparison(
        dataframes["Pesos_Max_Sharpe"],
        dataframes["Pesos_GMV"],
        optimal,
    )

    plot_correlation_heatmap(corr_matrix)

    print("\n" + "=" * 65)
    print("  ✅  Proceso completado exitosamente.")
    print(f"  📁  Archivo Excel : {OUTPUT_EXCEL}")
    print("  📊  Gráficas      : *.png")
    print("=" * 65)


if __name__ == "__main__":
    main()
