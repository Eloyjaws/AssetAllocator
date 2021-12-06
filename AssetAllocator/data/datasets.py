import yfinance as yf
from config.tickers import DOW_30_TICKER, NAS_100_TICKER

""" 
Fetch DOW Jones Data from yfinance
"""
DOW_30 = yf.download(DOW_30_TICKER, start='2009-01-02',
                     end='2021-10-27', interval="1d")['Adj Close']
