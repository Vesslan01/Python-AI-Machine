import yfinance as yf
import pandas as pd
import backtrader as bt
import ta
import numpy as np

# Hämta de senaste 1 månadens data från Yahoo Finance
interval = "1h"  # Ändrat 4h till 1h för att Yahoo Finance ska fungera
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
start_date = (pd.Timestamp.today() - pd.DateOffset(months=1)).strftime('%Y-%m-%d')

data = yf.download("GC=F", start=start_date, end=end_date, interval=interval)

# Debugging: Kontrollera om datan hämtats korrekt
print("Antal rader i datasetet:", len(data))
print("Finns NaN?", data.isnull().values.any())
print("Finns Inf?", np.isinf(data).values.any())
print(data.head())  # Se de första raderna

data.dropna(inplace=True)

# Säkerställ att det finns tillräckligt med data
if data.empty or len(data) < 50:
    raise ValueError("För lite data hämtades. Kontrollera intervallet och datakällan.")

# Fix: Hantera kolumnnamn korrekt
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(0)  # Tar bort extra index om det finns
data.columns = [col.lower().strip() for col in data.columns]  # Konvertera till små bokstäver och ta bort mellanslag

# Fix: Se till att det inte finns NaN eller Inf i datasetet
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

if data.empty:
    raise ValueError("All data innehöll NaN eller Inf. Kontrollera datakällan.")

print("Första raderna av renad data:")
print(data.head())


# Skapa en Backtrader-strategi
class MACD_RSI_Strategy(bt.Strategy):
    params = (
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('take_profit_pips', 1000)
    )

    def __init__(self):
        self.macd = bt.indicators.MACDHisto(
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.macd.macd[0] > self.macd.signal[0] and self.rsi[0] < self.params.rsi_oversold:
                self.order = self.buy()

        elif self.macd.macd[0] < self.macd.signal[0] or self.rsi[0] > self.params.rsi_overbought:
            self.order = self.sell()


# Sätta upp backtest-miljön
cerebro = bt.Cerebro()
data_feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(data_feed)
cerebro.addstrategy(MACD_RSI_Strategy)

# Förhindra att NaN eller Inf orsakar problem vid plotting
if not data.isnull().values.any() and not np.isinf(data).values.any():
    cerebro.run()
    if len(data) >= 50:
        cerebro.plot()
    else:
        print("För lite data för att plotta grafen.")
else:
    print("Datan innehåller NaN eller Inf, kan ej genomföra backtesting eller plotta.")
