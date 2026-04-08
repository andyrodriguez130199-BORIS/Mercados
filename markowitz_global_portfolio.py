from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font
from scipy.optimize import minimize


RISK_FREE_RATE = 0.042
TRADING_DAYS = 252
NUM_PORTFOLIOS = 100_000
MIN_ASSETS = 8
OUTPUT_FILE = "Resultados_Markowitz.xlsx"


MARKETS: dict[str, list[str]] = {
    "US_Tech": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"],
    "US_Industrial": ["CAT", "DE", "HON", "GE", "UNP", "UPS", "MMM", "F"],
    "Europe_BlueChips": ["ASML", "NVO", "SAP", "SHEL", "UL", "AZN"],
    "Japan": ["TM", "SONY", "MUFG", "NTDOY", "HMC", "SMFG"],
    "LatinAmerica": ["MELI", "PBR", "VALE", "ITUB", "BBAR", "BAP"],
}


GLOBAL_FALLBACK_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "V",
    "MA",
    "UNH",
    "XOM",
    "JNJ",
    "PG",
    "KO",
    "PFE",
    "HD",
    "DIS",
    "CSCO",
]


@dataclass
class PortfolioResult:
    returns: np.ndarray
    volatility: np.ndarray
    sharpe: np.ndarray
    weights: np.ndarray
    max_idx: int
    gmv_idx: int
    assets: list[str]
    weights_df_max: pd.DataFrame
    weights_df_gmv: pd.DataFrame
    expected_returns: pd.Series
    covariance: pd.DataFrame
    used_tickers: dict[str, list[str]]


def _download_close(
    ticker: str, start: datetime, end: datetime, windows: Iterable[int]
) -> pd.Series | None:
    for years in windows:
        s = max(start, end - timedelta(days=365 * years))
        try:
            data = yf.download(
                ticker,
                start=s.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                interval="1d",
            )
        except Exception:
            continue

        if data is None or data.empty:
            continue

        if "Close" in data:
            series = data["Close"].dropna()
        else:
            series = data.squeeze().dropna()

        if len(series) >= 200:
            series.name = ticker
            return series
    return None


def _validate_ticker(ticker: str) -> bool:
    try:
        probe = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
    except Exception:
        return False
    return probe is not None and not probe.empty


def resolve_tickers(markets: dict[str, list[str]]) -> dict[str, list[str]]:
    resolved: dict[str, list[str]] = {}
    used = set()

    for market, tickers in markets.items():
        valid: list[str] = []
        for ticker in tickers:
            if ticker in used:
                continue
            if _validate_ticker(ticker):
                valid.append(ticker)
                used.add(ticker)
            else:
                replacement = next(
                    (
                        t
                        for t in GLOBAL_FALLBACK_TICKERS
                        if t not in used and _validate_ticker(t)
                    ),
                    None,
                )
                if replacement:
                    print(f"   ↪️  Reemplazo: {ticker} → {replacement}")
                    valid.append(replacement)
                    used.add(replacement)
                else:
                    print(f"   ⚠️  Sin reemplazo para {ticker}; se omite.")
        if not valid:
            fallback = [
                t
                for t in GLOBAL_FALLBACK_TICKERS
                if t not in used and _validate_ticker(t)
            ][:4]
            valid.extend(fallback)
            used.update(fallback)
            if fallback:
                print(f"   ↪️  Mercado '{market}' sin datos; usando fallback: {fallback}")
        resolved[market] = valid

    return resolved


def build_price_matrix(start: datetime, end: datetime, tickers: list[str]) -> pd.DataFrame:
    series_list: list[pd.Series] = []
    for ticker in tickers:
        series = _download_close(ticker, start, end, windows=(10, 5, 3, 2, 1))
        if series is None:
            print(f"   ⚠️  Sin datos suficientes para {ticker}; se omite.")
            continue
        series_list.append(series)

    if not series_list:
        raise ValueError("No se obtuvieron series de precios válidas.")

    prices = pd.concat(series_list, axis=1).dropna(how="all")
    prices = prices.ffill().dropna()
    if prices.shape[1] < MIN_ASSETS:
        raise ValueError(f"Activos válidos insuficientes ({prices.shape[1]}).")
    return prices


def simulate_portfolios(returns: pd.DataFrame, num_portfolios: int) -> PortfolioResult:
    assets = list(returns.columns)
    n_assets = len(assets)
    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS

    random_w = np.random.random((num_portfolios, n_assets))
    random_w /= random_w.sum(axis=1, keepdims=True)

    port_returns = random_w @ mu.values
    cov_np = cov.values
    port_vol = np.sqrt(np.einsum("ij,jk,ik->i", random_w, cov_np, random_w))
    sharpe = (port_returns - RISK_FREE_RATE) / np.where(port_vol == 0, np.nan, port_vol)

    max_idx = int(np.nanargmax(sharpe))
    gmv_idx = int(np.nanargmin(port_vol))

    max_weights = pd.Series(random_w[max_idx], index=assets).sort_values(ascending=False)
    gmv_weights = pd.Series(random_w[gmv_idx], index=assets).sort_values(ascending=False)

    return PortfolioResult(
        returns=port_returns,
        volatility=port_vol,
        sharpe=sharpe,
        weights=random_w,
        max_idx=max_idx,
        gmv_idx=gmv_idx,
        assets=assets,
        weights_df_max=(max_weights[max_weights > 0.001] * 100).round(2).rename("Peso (%)").to_frame(),
        weights_df_gmv=(gmv_weights[gmv_weights > 0.001] * 100).round(2).rename("Peso (%)").to_frame(),
        expected_returns=mu,
        covariance=cov,
        used_tickers={},
    )


def optimize_frontier(returns: pd.DataFrame, target_points: int = 50) -> pd.DataFrame:
    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    n_assets = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    x0 = np.ones(n_assets) / n_assets

    def var_fn(w: np.ndarray) -> float:
        return float(w.T @ cov.values @ w)

    targets = np.linspace(mu.min(), mu.max(), target_points)
    frontier = []
    for t in targets:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, tr=t: np.dot(w, mu.values) - tr},
        )
        result = minimize(var_fn, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            vol = np.sqrt(var_fn(result.x))
            frontier.append((t, vol))
    return pd.DataFrame(frontier, columns=["Return", "Volatility"])


def _fig_to_image(fig) -> XLImage:
    stream = BytesIO()
    fig.savefig(stream, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    stream.seek(0)
    return XLImage(stream)


def create_figures(port: PortfolioResult, corr: pd.DataFrame, frontier: pd.DataFrame) -> dict[str, XLImage]:
    images: dict[str, XLImage] = {}

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sc = ax1.scatter(port.volatility, port.returns, c=port.sharpe, cmap="viridis", alpha=0.35, s=7)
    ax1.scatter(port.volatility[port.max_idx], port.returns[port.max_idx], color="red", s=120, label="Max Sharpe")
    ax1.scatter(port.volatility[port.gmv_idx], port.returns[port.gmv_idx], color="blue", s=120, label="GMV")
    if not frontier.empty:
        ax1.plot(frontier["Volatility"], frontier["Return"], color="black", linewidth=2, label="Frontera Eficiente")
    ax1.set_title("Frontera Eficiente de Markowitz")
    ax1.set_xlabel("Volatilidad Anualizada")
    ax1.set_ylabel("Rendimiento Esperado")
    ax1.legend()
    fig1.colorbar(sc, label="Sharpe Ratio")
    images["Frontera"] = _fig_to_image(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    port.weights_df_max["Peso (%)"].plot(kind="bar", ax=ax2, color="#2E86AB")
    ax2.set_title("Pesos - Portafolio Max Sharpe")
    ax2.set_ylabel("Peso (%)")
    images["Pesos_MaxSharpe"] = _fig_to_image(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    port.weights_df_gmv["Peso (%)"].plot(kind="bar", ax=ax3, color="#A23B72")
    ax3.set_title("Pesos - Portafolio GMV")
    ax3.set_ylabel("Peso (%)")
    images["Pesos_GMV"] = _fig_to_image(fig3)

    fig4, ax4 = plt.subplots(figsize=(9, 7))
    im = ax4.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax4.set_xticks(np.arange(len(corr.columns)))
    ax4.set_yticks(np.arange(len(corr.index)))
    ax4.set_xticklabels(corr.columns, rotation=90)
    ax4.set_yticklabels(corr.index)
    ax4.set_title("Mapa de Correlación")
    fig4.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    images["Correlacion"] = _fig_to_image(fig4)

    return images


def export_single_excel(
    output_path: str,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    corr: pd.DataFrame,
    port: PortfolioResult,
    frontier: pd.DataFrame,
    used_tickers: dict[str, list[str]],
) -> None:
    summary = pd.DataFrame(
        {
            "Métrica": [
                "Activos analizados",
                "Observaciones",
                "Portafolios simulados",
                "Riesgo libre",
                "Max Sharpe - Rendimiento",
                "Max Sharpe - Volatilidad",
                "Max Sharpe - Sharpe",
                "GMV - Rendimiento",
                "GMV - Volatilidad",
                "GMV - Sharpe",
            ],
            "Valor": [
                prices.shape[1],
                prices.shape[0],
                NUM_PORTFOLIOS,
                RISK_FREE_RATE,
                float(port.returns[port.max_idx]),
                float(port.volatility[port.max_idx]),
                float(port.sharpe[port.max_idx]),
                float(port.returns[port.gmv_idx]),
                float(port.volatility[port.gmv_idx]),
                float(port.sharpe[port.gmv_idx]),
            ],
        }
    )
    market_rows = [
        {"Mercado": market, "Tickers utilizados": ", ".join(tickers)}
        for market, tickers in used_tickers.items()
    ]
    market_table = pd.DataFrame(market_rows)

    images = create_figures(port, corr, frontier)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Resumen", index=False, startrow=0)
        market_table.to_excel(writer, sheet_name="Resumen", index=False, startrow=len(summary) + 3)
        prices.tail(300).to_excel(writer, sheet_name="Precios")
        returns.tail(300).to_excel(writer, sheet_name="Rendimientos")
        corr.to_excel(writer, sheet_name="Correlacion")
        port.weights_df_max.to_excel(writer, sheet_name="Pesos_MaxSharpe")
        port.weights_df_gmv.to_excel(writer, sheet_name="Pesos_GMV")
        pd.DataFrame(
            {
                "Return": port.returns,
                "Volatility": port.volatility,
                "Sharpe": port.sharpe,
            }
        ).to_excel(writer, sheet_name="Simulacion", index=False)
        frontier.to_excel(writer, sheet_name="Frontera", index=False)

        wb = writer.book
        chart_sheet = wb.create_sheet("Graficas")
        chart_sheet["A1"] = "Gráficas del análisis"
        chart_sheet["A1"].font = Font(bold=True, size=14)
        chart_sheet.add_image(images["Frontera"], "A3")
        chart_sheet.add_image(images["Pesos_MaxSharpe"], "A35")
        chart_sheet.add_image(images["Pesos_GMV"], "A67")
        chart_sheet.add_image(images["Correlacion"], "A99")


def main() -> None:
    print("OPTIMIZACIÓN DE PORTAFOLIO DE MARKOWITZ — 5 MERCADOS GLOBALES")
    print("=" * 65)
    print("\n📥  Descargando datos históricos …")

    end = datetime.utcnow()
    start = end - timedelta(days=365 * 10)

    used_tickers = resolve_tickers(MARKETS)
    final_tickers = sorted({ticker for group in used_tickers.values() for ticker in group})
    print(f"   Tickers candidatos: {len(final_tickers)}")

    prices = build_price_matrix(start, end, final_tickers)
    returns = prices.pct_change().dropna()
    corr = returns.corr()

    print(f"   ✅  Datos sincronizados: {prices.shape[0]} observaciones × {prices.shape[1]} activos.")
    print(f"\n   Activos con datos válidos: {prices.shape[1]}")

    print(f"\n⚙️   Ejecutando simulación Monte Carlo ({NUM_PORTFOLIOS:,} portafolios) …")
    port = simulate_portfolios(returns, NUM_PORTFOLIOS)
    port.used_tickers = used_tickers
    frontier = optimize_frontier(returns, target_points=60)
    print("   ✅  Simulación completada.")

    print(f"\n📌  Índice Max Sharpe : {port.max_idx}")
    print(f"📌  Índice GMV        : {port.gmv_idx}\n")
    print(
        f"   Max Sharpe → R={port.returns[port.max_idx]:.2%}  "
        f"σ={port.volatility[port.max_idx]:.2%}  SR={port.sharpe[port.max_idx]:.4f}"
    )
    print(
        f"   GMV        → R={port.returns[port.gmv_idx]:.2%}  "
        f"σ={port.volatility[port.gmv_idx]:.2%}  SR={port.sharpe[port.gmv_idx]:.4f}"
    )

    print(f"\n   Activos incluidos Max Sharpe: {len(port.weights_df_max)}")
    print(f"   Activos incluidos GMV       : {len(port.weights_df_gmv)}")

    print(f"\n💾  Exportando resultados a '{OUTPUT_FILE}' …")
    export_single_excel(
        OUTPUT_FILE,
        prices=prices,
        returns=returns,
        corr=corr,
        port=port,
        frontier=frontier,
        used_tickers=used_tickers,
    )
    print("   ✅  Archivo Excel guardado correctamente.")
    print("\n" + "=" * 65)
    print("  ✅  Proceso completado exitosamente.")
    print(f"  📁  Archivo Excel : {OUTPUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
