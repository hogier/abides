from agent.TradingAgent import TradingAgent
from util.util import log_print
from agent.HerdSlaveAgent import HerdSlaveAgent

from message.Message import Message
from math import sqrt
import numpy as np
import pandas as pd


class HerdMasterAgent(TradingAgent):

    def __init__(self, id, name, type, symbol='IBM', starting_cash=100000, sigma_n=0,
                 r_bar=100000, kappa=0.05, sigma_s=100000, future_window = 100000,
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

        # The agent must track its previous wake time, so it knows how many time
        # units have passed.
        self.prev_wake_time = None

        self.percent_aggr = 0.1                 #percent of time that the agent will aggress the spread
        self.size = np.random.randint(20, 50)   #size that the agent will be placing
        self.depth_spread = 2
        self.placed_orders = 0
        # for now let's do that the master is defined from the start of the kernel and it is fixed.
        # In a second moment I think I should do something like: the slave can ask for a list of registered masters
        # the exchange or a copy trading exchange agent which is on the same level of a market maker in terms of
        # agent level in the market. Then decides who to follow and make everything more dynamic.
        self.slave_ids = []
        self.slave_delays = {}

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

        self.oracle = self.kernel.oracle

        # for now there is only 1 master agent in the configuration.
        self.slave_ids = self.kernel.findAllAgentsByType(HerdSlaveAgent)

    def kernelStopping(self):
        # Always call parent method to be safe.
        super().kernelStopping()

        # Print end of day valuation.
        H = int(round(self.getHoldings(self.symbol), -2) / 100)
        # May request real fundamental value from oracle as part of final cleanup/stats.

        #marked to fundamental
        rT = self.oracle.observePrice(self.symbol, self.currentTime, sigma_n=0, random_state=self.random_state)

        # final (real) fundamental value times shares held.
        surplus = rT * H

        log_print("surplus after holdings: {}", surplus)

        # Add ending cash value and subtract starting cash value.
        surplus += self.holdings['CASH'] - self.starting_cash
        surplus = float( surplus )/self.starting_cash

        self.logEvent('FINAL_VALUATION', surplus, True)

        log_print(
            "{} final report.  Holdings {}, end cash {}, start cash {}, final fundamental {}, surplus {}",
            self.name, H, self.holdings['CASH'], self.starting_cash, rT, surplus)

        #print("Final surplus", self.name, surplus)

    def wakeup(self, currentTime):
        # Parent class handles discovery of exchange times and market_open wakeup call.
        super().wakeup(currentTime)

        self.state = 'INACTIVE'

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            for s_id in self.slave_ids:
                self.sendMessage(recipientID = s_id, msg = Message({"msg": "SLAVE_DELAY_REQUEST", "sender": self.id}))
            return
        else:
            if not self.trading:
                self.trading = True
                self.oracle.compute_fundamental_value_series(self.symbol, currentTime, sigma_n=0,
                                         random_state=self.random_state)
                # Time to start trading!
                log_print("{} is ready to start trading now.", self.name)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        if self.mkt_closed and (not self.symbol in self.daily_close_price):
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
        #estimate final value of the fundamental price
        #used for surplus calculation
        self.r_t = self.oracle.observePrice(self.symbol, self.currentTime, sigma_n=self.sigma_n,
                                         random_state=self.random_state)
        delta = pd.Timedelta(self.random_state.randint(low=self.future_window/10, high=self.future_window), unit='ns')
        if self.currentTime+delta < self.mkt_close:
            self.setWakeup(self.currentTime + delta)

        r_f = self.oracle.observePriceSpecial(self.symbol, self.currentTime+delta, sigma_n=self.sigma_n,
                                         random_state=self.random_state)
        bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)

        if bid and ask:
            mid = int((ask+bid)/2)
            spread = abs(ask - bid)

            if np.random.rand() < self.percent_aggr:
                adjust_int = 0
            else:
                adjust_int = np.random.randint( 0, self.depth_spread*spread )
                #adjustment to the limit price, allowed to post inside the spread
                #or deeper in the book as a passive order to maximize surplus

            if self.r_t < r_f:
                #fundamental belief that price will go down, place a sell order
                buy = True
                p = ask - adjust_int #submit a market order to sell, limit order inside the spread or deeper in the book
            elif self.r_t >= r_f:
                #fundamental belief that price will go up, buy order
                buy = False
                p = bid + adjust_int #submit a market order to buy, a limit order inside the spread or deeper in the book
        else:
            # initialize randomly
            buy = np.random.randint(0, 1 + 1)
            p = self.r_t

        # Place the order
        if self.currentTime+delta < self.mkt_close:
            self.placeLimitOrder(self.symbol, self.size, buy, p)
            self.setWakeup(self.currentTime + delta)

    def receiveMessage(self, currentTime, msg):
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receiveMessage(currentTime, msg)

        # We have been awakened by something other than our scheduled wakeup.
        # If our internal state indicates we were waiting for a particular event,
        # check if we can transition to a new state.

        if self.state == 'AWAITING_SPREAD':
            # We were waiting to receive the current spread/book.  Since we don't currently
            # track timestamps on retained information, we rely on actually seeing a
            # QUERY_SPREAD response message.

            if msg.body['msg'] == 'QUERY_SPREAD':
                # This is what we were waiting for.

                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed: return

                # We now have the information needed to place a limit order with the eta
                # strategic threshold parameter.
                self.placeOrder()
                self.state = 'AWAITING_WAKEUP'

        if msg.body['msg'] == "SLAVE_DELAY_RESPONSE":
            # Call the orderAccepted method, which subclasses should extend.
            self.slave_delays[msg.body['sender']] = msg.body['delay']
        elif msg.body['msg'] == "ORDER_ACCEPTED":
            # Call the orderAccepted method, which subclasses should extend.
            order = msg.body['order']
            self.placed_orders += 1
            #print('M', self.currentTime, self.placed_orders)
            for s_id in self.slave_ids:
                self.sendMessage(recipientID = s_id, msg = Message({"msg": "MASTER_ORDER_ACCEPTED", "sender": self.id,
                                                           "order": order}), delay=self.slave_delays[s_id])
        elif msg.body['msg'] == "ORDER_CANCELLED":
            # Call the orderCancelled method, which subclasses should extend.
            order = msg.body['order']
            for s_id in self.slave_ids:
                self.sendMessage(recipientID = s_id, msg = Message({"msg": "MASTER_ORDER_CANCELLED", "sender": self.id,
                                                           "order": order}), delay=self.slave_delays[s_id])
        elif msg.body['msg'] == "ORDER_EXECUTED":
            order = msg.body['order']

        # Cancel all open orders.
        # Return value: did we issue any cancellation requests?

    def cancelOrders(self):
        if not self.orders: return False

        for id, order in self.orders.items():
            self.cancelOrder(order)

        return True

    def getWakeFrequency(self):
        return pd.Timedelta(self.random_state.randint(low=10, high=100), unit='ns')