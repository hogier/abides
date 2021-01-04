import pandas as pd
import sys
import os

sys.path.append('../..')

from realism.realism_utils import make_orderbook_for_analysis, MID_PRICE_CUTOFF
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import timedelta, datetime
import argparse
import json
import matplotlib

matplotlib.rcParams['agg.path.chunksize'] = 10000

from dateutil.parser import parse

import pickle
from tqdm import tqdm


def log_order_book_snapshots(log_dir, symbol, book_freq, wide_book, mkt_open, mkt_close):
    """
    Log full depth quotes (price, volume) from this order book at some pre-determined frequency. Here we are looking at
    the actual log for this order book (i.e. are there snapshots to export, independent of the requested frequency).
    """

    def get_quote_range_iterator(s):
        """ Helper method for order book logging. Takes pandas Series and returns python range() from first to last
          element.
      """
        forbidden_values = [0, 19999900]  # TODO: Put constant value in more sensible place!
        quotes = sorted(s)
        for val in forbidden_values:
            try:
                quotes.remove(val)
            except ValueError:
                pass
        return quotes

    print("Logging order book to file...")
    dfLog = book_log_to_df(log_dir, symbol)
    dfLog.set_index('QuoteTime', inplace=True)
    dfLog = dfLog[~dfLog.index.duplicated(keep='last')]
    dfLog.sort_index(inplace=True)

    if str(book_freq).isdigit() and int(book_freq) == 0:  # Save all possible information
        # Get the full range of quotes at the finest possible resolution.
        quotes = get_quote_range_iterator(dfLog.columns.unique())

        # Restructure the log to have multi-level rows of all possible pairs of time and quote
        # with volume as the only column.
        if not wide_book:
            filledIndex = pd.MultiIndex.from_product([dfLog.index, quotes], names=['time', 'quote'])
            dfLog = dfLog.stack()
            dfLog = dfLog.reindex(filledIndex)

        filename = f'ORDERBOOK_{symbol}_FULL'

    else:  # Sample at frequency self.book_freq
        # With multiple quotes in a nanosecond, use the last one, then resample to the requested freq.
        dfLog = dfLog.resample(book_freq).ffill()
        dfLog.sort_index(inplace=True)

        # Create a fully populated index at the desired frequency from market open to close.
        # Then project the logged data into this complete index.
        time_idx = pd.date_range(mkt_open, mkt_close, freq=book_freq, closed='right')
        dfLog = dfLog.reindex(time_idx, method='ffill')
        dfLog.sort_index(inplace=True)

        if not wide_book:
            dfLog = dfLog.stack()
            dfLog.sort_index(inplace=True)

            # Get the full range of quotes at the finest possible resolution.
            quotes = get_quote_range_iterator(dfLog.index.get_level_values(1).unique())

            # Restructure the log to have multi-level rows of all possible pairs of time and quote
            # with volume as the only column.
            filledIndex = pd.MultiIndex.from_product([time_idx, quotes], names=['time', 'quote'])
            dfLog = dfLog.reindex(filledIndex)

        filename = f'ORDERBOOK_{symbol}_FREQ_{book_freq}'

    # Final cleanup
    if not wide_book:
        dfLog.rename('Volume')
        df = pd.SparseDataFrame(index=dfLog.index)
        df['Volume'] = dfLog
    else:
        df = dfLog
        df = df.reindex(sorted(df.columns), axis=1)

    # Archive the order book snapshots directly to a file named with the symbol, rather than
    # to the exchange agent log.
    file = "{}.bz2".format(filename)

    df.to_pickle(os.path.join(log_dir, file), compression='bz2')

    print("Order book logging complete!")


def book_log_to_df(log_dir, symbol):
    filename = f'BOOK_LOG_{symbol}_CHUNK'
    book_log_files = []
    for file in os.listdir(log_dir):
        if filename in file:
            book_log_files.append(os.path.join(log_dir, file))
    df = None
    for filename in tqdm(book_log_files):
        with open(filename, "rb") as input_file:
            book_log_chunk = pickle.load(input_file)

            if df is None:
                df = pd.DataFrame(book_log_chunk)
            else:
                df = df.append(book_log_chunk)
            quotes_times = df.QuoteTime
            df.drop(columns='QuoteTime', inplace=True)
            df = df.astype("Sparse[float32]")
            df.sort_index(axis=1, inplace=True)
            df.insert(0, 'QuoteTime', quotes_times, allow_duplicates=True)
        # os.remove(filename)

    df.sort_values(by='QuoteTime', inplace=True)
    return df


def main(log_dir, symbol, book_freq, wide_book, mkt_open, mkt_close):
    log_order_book_snapshots(log_dir, symbol, book_freq, wide_book, mkt_open, mkt_close)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI utility for inspecting liquidity issues and transacted volumes '
                                                 'for a day of trading.')

    parser.add_argument('-l',
                        '--log_dir',
                        required=True,
                        help='Log directory name.')
    parser.add_argument('-t',
                        '--ticker',
                        required=True,
                        help='Ticker (symbol) to use.')

    # TODO check type for book_freq
    parser.add_argument('-b', '--book-freq',
                        help="",
                        type=str,
                        default='S'
                        )

    parser.add_argument('-w', '--wide-book',
                        help="",
                        type=bool,
                        default=False
                        )

    parser.add_argument('-d', '--historical-date',
                        required=True,
                        type=parse,
                        help='historical date being simulated in format YYYYMMDD.')
    parser.add_argument('--start-time',
                        default='09:30:00',
                        type=parse,
                        help='Starting time of simulation.'
                        )
    parser.add_argument('--end-time',
                        default='11:30:00',
                        type=parse,
                        help='Ending time of simulation.'
                        )

    args, remaining_args = parser.parse_known_args()
    log_dir = args.log_dir
    symbol = args.ticker

    book_freq = args.book_freq
    wide_book = args.wide_book

    historical_date = pd.to_datetime(args.historical_date)
    mkt_open = historical_date + pd.to_timedelta(args.start_time.strftime('%H:%M:%S'))
    mkt_close = historical_date + pd.to_timedelta(args.end_time.strftime('%H:%M:%S'))

    main(log_dir, symbol, book_freq, wide_book, mkt_open, mkt_close)
