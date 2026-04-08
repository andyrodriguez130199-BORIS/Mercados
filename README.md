# Mercados

Script principal: `/home/runner/work/Mercados/Mercados/markowitz_global_portfolio.py`

## Requisitos

- Python 3.10+
- Paquetes:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `yfinance`
  - `openpyxl`

Instalación rápida:

```bash
pip install numpy pandas matplotlib scipy yfinance openpyxl
```

## Ejecución

```bash
python /home/runner/work/Mercados/Mercados/markowitz_global_portfolio.py
```

## Qué hace

- Descarga precios históricos de 5 mercados globales.
- Si un ticker no tiene datos en Yahoo Finance, lo reemplaza por uno común y válido.
- Si un mercado queda vacío, usa fallback automático.
- Ejecuta simulación Monte Carlo para Max Sharpe y GMV.
- Exporta todo en **un solo archivo Excel**: `Resultados_Markowitz.xlsx`, incluyendo tablas y gráficas embebidas.
