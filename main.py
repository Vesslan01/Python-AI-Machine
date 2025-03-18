import yfinance as yf
import pandas as pd
import backtrader as bt
import numpy as np
import logging
from alpha_vantage.timeseries import TimeSeries  # Alternativ datakälla (Alpha Vantage)

# Konfiguration
SYMBOL = "GC=F"  # Symbol för Guld Futures
INTERVAL = "1d"  # Intervall för data (1 dag)
START_DATE = (pd.Timestamp.today() - pd.DateOffset(years=2)).strftime('%Y-%m-%d')  # Senaste 2 åren
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')

# Alpha Vantage API-nyckel
ALPHA_VANTAGE_API_KEY = 'VZ03G1CIRDZQJGPC'  # Din API-nyckel

# Konfigurera logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_yahoo_data():
    """Hämta data från Yahoo Finance."""
    try:
        logging.info("Hämtar data från Yahoo Finance...")
        data = yf.download(SYMBOL, start=START_DATE, end=END_DATE, interval=INTERVAL, auto_adjust=False)
        if data.empty:
            raise ValueError("Ingen data hämtades från Yahoo Finance.")
        return data
    except Exception as e:
        logging.error(f"Ett fel uppstod vid hämtning från Yahoo Finance: {e}")
        return None


def fetch_alpha_vantage_data():
    """Hämta data från Alpha Vantage."""
    try:
        logging.info("Hämtar data från Alpha Vantage...")
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=SYMBOL, outputsize='full')
        data.index = pd.to_datetime(data.index)  # Konvertera index till datetime
        data = data.sort_index()  # Sortera efter datum
        data = data.loc[START_DATE:END_DATE]  # Filtrera för önskad tidsperiod
        if data.empty:
            raise ValueError("Ingen data hämtades från Alpha Vantage.")
        return data
    except Exception as e:
        logging.error(f"Ett fel uppstod vid hämtning från Alpha Vantage: {e}")
        return None


def prepare_data(data):
    """Förbered data för backtesting."""
    # Hantera MultiIndex för kolumner
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)  # Ta bort den andra nivån (t.ex. 'GC=F')

    # Konvertera kolumnnamn till gemener och ta bort mellanslag
    data.columns = data.columns.str.lower().str.strip()

    # Byt namn på kolumner för att matcha Backtraders förväntningar
    column_mapping = {
        'price': 'close',  # Om 'price' är stängningspriset
        'adj close': 'adj_close',  # Ytterligare kolumn, inte nödvändig för Backtrader
        'close': 'close',  # Stängningspris
        'high': 'high',    # Höjdpunkt
        'low': 'low',      # Lågpunkt
        'open': 'open',    # Öppningspris
        'volume': 'volume'  # Volym
    }
    data.rename(columns=column_mapping, inplace=True)

    # Hantera volymkolumnen dynamiskt
    if 'volume' in data.columns:
        data['volume'] = data['volume'].replace(0, np.nan)  # Ersätt 0 med NaN
        data['volume'] = data['volume'].ffill()  # Fyll saknade värden
    else:
        logging.warning("Volymkolumn saknas, skapar en dummy-kolumn.")
        data['volume'] = np.nan

    # Ta bort rader med saknade värden
    data.dropna(inplace=True)

    # Säkerställ att det finns tillräckligt med data
    if data.empty or len(data) < 30:
        raise ValueError("För lite data hämtades. Kontrollera intervallet och datakällan.")

    # Säkerställ att datetime-index är korrekt
    data.index = pd.to_datetime(data.index)

    logging.info("Första raderna av renad data:")
    logging.info(data.head())

    logging.info("Sista raderna av renad data:")
    logging.info(data.tail())

    logging.info("Beskrivande statistik för data:")
    logging.info(data.describe())

    return data


def run_backtest(data):
    """Kör backtest med Backtrader."""

    class MACD_RSI_Strategy(bt.Strategy):
        params = (
            ('macd_fast', 12),  # Korttids EMA-period
            ('macd_slow', 26),  # Långtids EMA-period
            ('macd_signal', 9),  # Signallinjeperiod
            ('rsi_period', 14),  # RSI-period
            ('rsi_overbought', 70),  # Överköpt nivå
            ('rsi_oversold', 30),   # Översåld nivå
            ('take_profit_pips', 1000),  # Take-profit nivå i pips
            ('stop_loss_pips', 500),  # Stop-loss nivå i pips
            ('position_size', 1000)  # Fast positionsstorlek
        )

        def __init__(self):
            self.macd = bt.indicators.MACDHisto(
                period_me1=self.params.macd_fast,
                period_me2=self.params.macd_slow,
                period_signal=self.params.macd_signal
            )
            self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
            self.order = None
            self.trade_count = 0  # Räknare för antalet affärer
            self.net_profit = 0  # Nettoresultat
            self.stop_loss_price = None  # Stop-loss pris
            self.take_profit_price = None  # Take-profit pris

        def notify_order(self, order):
            """Hantera orderutförande."""
            if order.status in [order.Completed, order.Canceled, order.Margin]:
                self.order = None  # Återställ ordern

        def next(self):
            if self.order:
                return  # Vänta på att den väntande ordern ska slutföras

            # Felsökningsutskrifter
            print(f"Datum: {self.data.datetime.date(0)}, Stängning: {self.data.close[0]}")
            print(f"MACD: {self.macd.macd[0]}, Signal: {self.macd.signal[0]}, RSI: {self.rsi[0]}")

            # Kontrollera om vi inte har en öppen position
            if not self.position:
                # Köpvillkor: MACD korsar över Signal och RSI är översåld
                if self.macd.macd[0] > self.macd.signal[0] and self.rsi[0] < self.params.rsi_oversold:
                    print("Skickar köporder.")
                    self.order = self.buy(size=self.params.position_size)
                    self.trade_count += 1

                    # Sätt stop-loss och take-profit nivåer
                    self.stop_loss_price = self.data.close[0] - self.params.stop_loss_pips
                    self.take_profit_price = self.data.close[0] + self.params.take_profit_pips
                    print(f"Stop-loss: {self.stop_loss_price}, Take-profit: {self.take_profit_price}")

            # Om vi har en öppen position
            else:
                # Säljvillkor: MACD korsar under Signal eller RSI är överköpt
                if self.macd.macd[0] < self.macd.signal[0] or self.rsi[0] > self.params.rsi_overbought:
                    print("Skickar säljorder.")
                    self.order = self.sell(size=self.params.position_size)
                    self.trade_count += 1

                # Kontrollera stop-loss och take-profit nivåer
                if self.data.close[0] <= self.stop_loss_price:
                    print(f"Stop-loss träffad vid {self.data.close[0]}. Stänger position.")
                    self.order = self.sell(size=self.params.position_size)
                elif self.data.close[0] >= self.take_profit_price:
                    print(f"Take-profit träffad vid {self.data.close[0]}. Stänger position.")
                    self.order = self.sell(size=self.params.position_size)

        def stop(self):
            # Beräkna nettoresultat
            self.net_profit = self.broker.getvalue() - 10000.00  # Startkapital är 10,000.00

            # Skriv ut resultat
            logging.info("\n--- Backtest Resultat ---")
            logging.info(f"Startkapital: 10000.00")
            logging.info(f"Slutvärde: {self.broker.getvalue():.2f}")
            logging.info(f"Nettoresultat: {self.net_profit:.2f} ({'Vinst' if self.net_profit >= 0 else 'Förlust'})")
            logging.info(f"Antal affärer: {self.trade_count}")

    # Sätt upp backtest-miljön
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000.00)  # Sätt startkapital till 10,000
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
        logging.info("Försöker med alternativ datakälla (Alpha Vantage)...")
        data = fetch_alpha_vantage_data()

    # Om ingen datakälla fungerar, avsluta programmet
    if data is None:
        logging.error("Kunde inte hämta data från någon datakälla. Avslutar programmet.")
        return

    # Förbered data för backtesting
    try:
        data = prepare_data(data)
    except Exception as e:
        logging.error(f"Ett fel uppstod vid förberedelse av data: {e}")
        return

    # Kör backtest
    try:
        run_backtest(data)
    except Exception as e:
        logging.error(f"Ett fel uppstod under backtesting: {e}")


if __name__ == "__main__":
    main()