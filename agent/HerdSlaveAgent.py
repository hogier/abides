from agent.TradingAgent import TradingAgent
from util.util import log_print

import pandas as pd


class HerdSlaveAgent(TradingAgent):

    def __init__(self, id, name, type, starting_cash=100000, log_orders=False, random_state=None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)

        # for now let's do that the master is defined from the start of the kernel and it is fixed.
        # In a second moment I think I should do something like: the slave can ask for a list of registered masters
        # the exchange or a copy trading exchange agent which is on the same level of a market maker in terms of
        # agent level in the market. Then decides who to follow and make everything more dynamic.
        self.master_id = None

    def kernelStarting(self, start_time):

        super().kernelStarting(start_time)

        # for now there is only 1 master agent in the configuration.
        self.master_id = self.kernel.findAgentByType(HerdMasterAgent)

    def kernelStopping(self):
        # Always call parent method to be safe.
        super().kernelStopping()

        # Print end of day valuation.
        H = int(round(self.getHoldings(self.symbol), -2) / 100)
        # May request real fundamental value from oracle as part of final cleanup/stats.
        #-------------------------------------------------
        #if self.symbol != 'ETF':
        rT = self.oracle.observePrice(self.symbol, self.currentTime, sigma_n=0, random_state=self.random_state)
        #else:
        #    portfolio_rT, rT = self.oracle.observePortfolioPrice(self.symbol, self.portfolio, self.currentTime,
        #                                                         sigma_n=0,
        #                                                         random_state=self.random_state)
        #-------------------------------------------------

        # Start with surplus as private valuation of shares held.
        if H > 0:
            surplus = sum([self.theta[x + self.q_max - 1] for x in range(1, H + 1)])
        elif H < 0:
            surplus = -sum([self.theta[x + self.q_max - 1] for x in range(H + 1, 1)])
        else:
            surplus = 0

        log_print("surplus init: {}", surplus)

        # Add final (real) fundamental value times shares held.
        surplus += rT * H

        log_print("surplus after holdings: {}", surplus)

        # Add ending cash value and subtract starting cash value.
        surplus += self.holdings['CASH'] - self.starting_cash

        self.logEvent('FINAL_VALUATION', surplus, True)

        log_print(
            "{} final report.  Holdings {}, end cash {}, start cash {}, final fundamental {}, preferences {}, surplus {}",
            self.name, H, self.holdings['CASH'], self.starting_cash, rT, self.theta, surplus)

    # collects the subscription market data and processes meassages from the exchange such as when the market opens and closes
    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)  # receives subscription market data
        self.mid_price_history = self.mid_price_history.append(
            pd.Series({'mid_price': self.getCurrentMidPrice()}, name=currentTime))
        self.mid_price_history.dropna(inplace=True)

    # populated with the orderbook subscription data by the receiveMessage method
    # each a list, where each element is a tuple (price, quantity)
    def getCurrentMidPrice(self):
        try:
            best_bid = self.current_bids[0][0]
            best_ask = self.current_asks[0][0]
            return round((best_ask + best_bid) / 2)
        except (TypeError, IndexError):
            return None

    # next wakeup using randomness or some deterministic formula, it need not be a constant. Here for simplicity we set a constant wakeup.
    def getWakeFrequency(self):

        return pd.Timedelta(self.wake_freq)

    # compute moving averages with the computeMidPriceMovingAverages
    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        short_moving_avg, long_moving_avg = self.computeMidPriceMovingAverages()
        if short_moving_avg is not None and long_moving_avg is not None:
            if short_moving_avg > long_moving_avg:
                self.placeMarketOrder(self.order_size, 0)
            elif short_moving_avg < long_moving_avg:
                self.placeMarketOrder(self.order_size, 1)

    def computeMidPriceMovingAverages(self):
        try:
            short_moving_avg = self.mid_price_history.rolling(self.short_window).mean().iloc[-1]['mid_price']
            long_moving_avg = self.mid_price_history.rolling(self.long_window).mean().iloc[-1]['mid_price']
            return short_moving_avg, long_moving_avg
        except IndexError:
            return None, None

    def placeLimitOrder(self, quantity, is_buy_order, limit_price):
        """ Place a limit order at the exchange.
          :param quantity (int):      order quantity
          :param is_buy_order (bool): True if Buy else False
          :param limit_price: price level at which to place a limit order
          :return:
        """
        super().placeLimitOrder(self.symbol, quantity, is_buy_order, limit_price)

    def placeMarketOrder(self, quantity, is_buy_order):
        """ Place a market order at the exchange.
          :param quantity (int):      order quantity
          :param is_buy_order (bool): True if Buy else False
          :return:
        """
        super().placeMarketOrder(self.symbol, quantity, is_buy_order)

    def cancelAllOrders(self):
        """ Cancels all resting limit orders placed by the experimental agent.
        """
        for _, order in self.orders.items():
            self.cancelOrder(order)

