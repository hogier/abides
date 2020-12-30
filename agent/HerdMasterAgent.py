from agent.TradingAgent import TradingAgent
from util.util import log_print
from agent.HerdSlaveAgent import HerdSlaveAgent

from message.Message import Message
import numpy as np
import pandas as pd


class HerdMasterAgent(TradingAgent):

    def __init__(self, id, name, type, symbol='IBM', starting_cash=100000, sigma_n=0,
                 r_bar=100000, kappa=0.05, sigma_s=100000, future_window=100000,
                 lambda_a=0.005, log_orders=False, random_state=None):

        # Base class init.
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)

        # Store important parameters particular to the ZI agent.
        self.symbol = symbol  # symbol to trade
        self.sigma_n = sigma_n  # observation noise variance
        self.r_bar = r_bar  # true mean fundamental value
        self.kappa = kappa  # mean reversion parameter
        self.sigma_s = sigma_s  # shock variance
        self.lambda_a = lambda_a  # mean arrival rate of ZI agents
        self.future_window = future_window

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state = 'AWAITING_WAKEUP'

        # The agent maintains two priors: r_t and sigma_t (value and error estimates).
        self.r_t = r_bar
        self.sigma_t = 0
        self.oracle = None

        # The agent must track its previous wake time, so it knows how many time
        # units have passed.
        self.prev_wake_time = None

        self.percent_aggr = 0.1                 # percent of time that the agent will aggres the spread
        self.size = 2   # size that the agent will be placing
        self.depth_spread = 2
        self.placed_orders = 0
        # for now let's do that the master is defined from the start of the kernel and it is fixed.
        # In a second moment I think I should do something like: the slave can ask for a list of registered masters
        # the exchange or a copy trading exchange agent which is on the same level of a market maker in terms of
        # agent level in the market. Then decides who to follow and make everything more dynamic.
        self.slave_ids = []
        self.slave_delays = {}

    def kernelStarting(self, startTime):
        self.logEvent('WAKE_FREQUENCY', self.future_window, True)

        super().kernelStarting(startTime)

        self.oracle = self.kernel.oracle

        # for now there is only 1 master agent in the configuration.
        self.slave_ids = self.kernel.findAllAgentsByType(HerdSlaveAgent)

    def kernelStopping(self):
        # Always call parent method to be safe.
        super().kernelStopping()

        # Print end of day valuation.
        h = int(round(self.getHoldings(self.symbol), -2) / 100)
        # May request real fundamental value from oracle as part of final cleanup/stats.

        # marked to fundamental
        r_t = self.oracle.observePrice(self.symbol, self.currentTime, sigma_n=0, random_state=self.random_state)

        # final (real) fundamental value times shares held.
        surplus = r_t * h

        log_print("surplus after holdings: {}", surplus)

        # Add ending cash value and subtract starting cash value.
        surplus += self.holdings['CASH'] - self.starting_cash
        surplus = float(surplus)/self.starting_cash

        self.logEvent('FINAL_VALUATION', surplus, True)

        log_print(
            "{} final report.  Holdings {}, end cash {}, start cash {}, final fundamental {}, surplus {}",
            self.name, h, self.holdings['CASH'], self.starting_cash, r_t, surplus)

    def wakeup(self, currentTime):
        super().wakeup(currentTime)

        self.state = 'INACTIVE'

        if not self.mkt_open or not self.mkt_close:
            for s_id in self.slave_ids:
                self.sendMessage(recipientID=s_id, msg=Message({"msg": "SLAVE_DELAY_REQUEST", "sender": self.id}))
            return
        else:
            if not self.trading:
                self.trading = True
                self.oracle.compute_fundamental_value_series(self.symbol, currentTime, sigma_n=0,
                                                             random_state=self.random_state)
                log_print("{} is ready to start trading now.", self.name)

        if self.mkt_closed and (self.symbol in self.daily_close_price):
            return

        if self.mkt_closed and (self.symbol not in self.daily_close_price):
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
            return

        self.cancelOrders()

        if type(self) == HerdMasterAgent:
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
        else:
            self.state = 'ACTIVE'

    def placeOrder(self):
        self.r_t = self.oracle.observePrice(self.symbol, self.currentTime, sigma_n=self.sigma_n,
                                            random_state=self.random_state)
        #delta = pd.Timedelta(self.random_state.randint(low=self.future_window/2, high=self.future_window), unit='ns')
        delta = pd.Timedelta(self.future_window, unit='ns')
        if self.currentTime+delta < self.mkt_close:
            self.setWakeup(self.currentTime + delta)

        r_f = self.oracle.observePriceSpecial(self.symbol, self.currentTime+delta, sigma_n=self.sigma_n,
                                              random_state=self.random_state)
        bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)
        size = self.size
        if bid and ask:
            spread = abs(ask - bid)

            if np.random.rand() < self.percent_aggr:
                adjust_int = 0
            else:
                adjust_int = np.random.randint(0, self.depth_spread*spread)

            if ask < r_f:
                buy = True
                p = ask - adjust_int
                size = self.getHoldings(self.symbol)*(-1) if self.getHoldings(self.symbol) < 0 else self.size
                if p >= r_f:
                    return
            elif bid > r_f:
                buy = False
                p = bid + adjust_int
                size = self.getHoldings(self.symbol) if self.getHoldings(self.symbol) > 0 else self.size
                if p <= r_f:
                    return
            else:
                return
        else:
            return

        h = self.getHoldings(self.symbol)
        surplus = self.r_t * h
        #print(self.currentTime, self.getHoldings(self.symbol), self.holdings['CASH'], h, surplus,
        #     self.holdings['CASH'] + surplus)
        if self.currentTime+delta < self.mkt_close:
            self.placeLimitOrder(self.symbol, size, buy, p)
            for s_id in self.slave_ids:
                self.sendMessage(recipientID=s_id, msg=Message({"msg": "MASTER_ORDER_PLACED", "sender": self.id,
                                                                "symbol": self.symbol, "quantity": size,
                                                                "is_buy_order": buy, 'limit_price': p}), delay=self.slave_delays[s_id])


    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

        if self.state == 'AWAITING_SPREAD':

            if msg.body['msg'] == 'QUERY_SPREAD':
                if self.mkt_closed:
                    return

                self.placeOrder()
                self.state = 'AWAITING_WAKEUP'

        if msg.body['msg'] == "SLAVE_DELAY_RESPONSE":
            self.slave_delays[msg.body['sender']] = msg.body['delay']
        elif msg.body['msg'] == "ORDER_ACCEPTED":
            order = msg.body['order']
            for s_id in self.slave_ids:
                self.sendMessage(recipientID=s_id, msg=Message({"msg": "MASTER_ORDER_ACCEPTED", "sender": self.id,
                                                                "order": order}), delay=self.slave_delays[s_id])
        elif msg.body['msg'] == "ORDER_CANCELLED":
            order = msg.body['order']
            for s_id in self.slave_ids:
                self.sendMessage(recipientID=s_id, msg=Message({"msg": "MASTER_ORDER_CANCELLED", "sender": self.id,
                                                                "order": order}), delay=self.slave_delays[s_id])

    def cancelOrders(self):
        if not self.orders:
            return False

        for _, order in self.orders.items():
            self.cancelOrder(order)

        return True

    def getWakeFrequency(self):
        return pd.Timedelta(self.random_state.randint(low=10, high=100), unit='ns')
