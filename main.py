import yfinance as yf
import pandas as pd
import backtrader as bt
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries  # För alternativ datakälla (Alpha Vantage)

# Konfiguration
SYMBOL = "GC=F"  # Symbol för Guld Futures
INTERVAL = "1d"  # Intervall för data (1 dag)
START_DATE = (pd.Timestamp.today() - pd.DateOffset(months=12)).strftime('%Y-%m-%d')  # Senaste 12 månaderna
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')

# Alternativ datakälla (Alpha Vantage)
ALPHA_VANTAGE_API_KEY = 'DIN_ALPHA_VANTAGE_API_NYCKEL'  # Ersätt med din API-nyckel


def fetch_yahoo_data():
    """Hämta data från Yahoo Finance."""
    try:
        print("Försöker hämta data från Yahoo Finance...")
        data = yf.download(SYMBOL, start=START_DATE, end=END_DATE, interval=INTERVAL, auto_adjust=False)
        if data.empty:
            raise ValueError("Ingen data hämtades från Yahoo Finance.")
        return data
    except Exception as e:
        print(f"Ett fel uppstod vid hämtning från Yahoo Finance: {e}")
        return None


def fetch_alpha_vantage_data():
    """Hämta data från Alpha Vantage."""
    try:
        print("Försöker hämta data från Alpha Vantage...")
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=SYMBOL, outputsize='full')
        data.index = pd.to_datetime(data.index)  # Konvertera index till datetime
        data = data.sort_index()  # Sortera efter datum
        data = data.loc[START_DATE:END_DATE]  # Filtrera för önskad tidsperiod
        if data.empty:
            raise ValueError("Ingen data hämtades från Alpha Vantage.")
        return data
    except Exception as e:
        print(f"Ett fel uppstod vid hämtning från Alpha Vantage: {e}")
        return None


def prepare_data(data):
    """Förbered data för backtesting."""
    # Hantera MultiIndex för kolumner
    if isinstance(data.columns, pd.MultiIndex):
        # Ta bort extra nivåer om det finns
        data.columns = data.columns.droplevel(1)  # Ta bort den andra nivån (t.ex. 'GC=F')

    # Konvertera kolumnnamn till gemener och ta bort mellanslag
    data.columns = data.columns.str.lower().str.strip()

    # Hantera volymkolumnen dynamiskt
    if 'volume' in data.columns:
        data['volume'].replace(0, np.nan, inplace=True)
    else:
        print("Volymkolumn saknas, skapar en dummy-kolumn.")
        data['volume'] = np.nan

    # Ta bort rader med saknade värden
    data.dropna(inplace=True)

    # Säkerställ att det finns tillräckligt med data
    if data.empty or len(data) < 30:
        raise ValueError("För lite data hämtades. Kontrollera intervallet och datakällan.")

    # Säkerställ att datetime-index är korrekt
    data.index = pd.to_datetime(data.index)

    print("Första raderna av renad data:")
    print(data.head())

    return data


def run_backtest(data):
    """Kör backtest med Backtrader."""

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

            print(
                f"Dagens stängningspris: {self.data.close[0]}, MACD: {self.macd.macd[0]}, RSI: {self.rsi[0]}")  # Felsökningsutskrift

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

    # Kör backtest
    cerebro.run()

    # Plotta resultat
    cerebro.plot()


def main():
    # Försök hämta data från Yahoo Finance
    data = fetch_yahoo_data()

    # Om Yahoo Finance misslyckas, försök med Alpha Vantage
    if data is None:
        print("Försöker med alternativ datakälla (Alpha Vantage)...")
        data = fetch_alpha_vantage_data()

    # Om ingen datakälla fungerar, avsluta programmet
    if data is None:
        print("Kunde inte hämta data från någon datakälla. Avslutar programmet.")
        return

    # Förbered data för backtesting
    try:
        data = prepare_data(data)
    except Exception as e:
        print(f"Ett fel uppstod vid förberedelse av data: {e}")
        return

    # Kör backtest
    try:
        run_backtest(data)
    except Exception as e:
        print(f"Ett fel uppstod under backtesting: {e}")


if __name__ == "__main__":
    main()