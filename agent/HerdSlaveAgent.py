from agent.TradingAgent import TradingAgent
from util.util import log_print

import pandas as pd
from message.Message import Message


class HerdSlaveAgent(TradingAgent):

    def __init__(self, id, name, type, symbol='IBM', starting_cash=100000,
                 min_delay=0, max_delay=0, log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)

        self.trading = False
        self.symbol = symbol
        self.master_delay = self.random_state.randint(low=min_delay, high=max_delay)

        self.state = 'AWAITING_WAKEUP'

        self.placed_orders = 0

        self.master_id = None

    def kernelStarting(self, start_time):
        super().kernelStarting(start_time)

    def kernelStopping(self):
        super().kernelStopping()

        cash = self.markToMarket(self.holdings)
        print("Final value for {}: {}.  Having delay: {}".format(self.name, cash, self.master_delay))

    def wakeup(self, currentTime):
        super().wakeup(currentTime)

        self.state = 'INACTIVE'

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return
        else:
            if not self.trading:
                self.trading = True
                # Time to start trading!
                log_print("{} is ready to start trading now.", self.name)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        delta_time = pd.Timedelta(self.random_state.randint(low=1000000, high=10000000), unit='ms')
        if currentTime+delta_time < self.mkt_close:
            self.setWakeup(currentTime + delta_time)

        if self.mkt_closed and (self.symbol not in self.daily_close_price):
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
            return

        if type(self) == HerdSlaveAgent:
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
        else:
            self.state = 'ACTIVE'

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

        if msg.body['msg'] == "SLAVE_DELAY_REQUEST":
            self.master_id = msg.body['sender']
            self.sendMessage(recipientID=self.master_id,
                             msg=Message({"msg": "SLAVE_DELAY_RESPONSE", "sender": self.id,
                                          "delay": self.master_delay}))
        elif msg.body['msg'] == "MASTER_ORDER_ACCEPTED":
            order = msg.body['order'].to_dict()
            self.placed_orders += 1

            self.cancelOrders()
            if order['is_buy_order']:
                quantity = order['quantity']
            else:
                quantity = self.getHoldings(self.symbol) if self.getHoldings(self.symbol) > 0 else order['quantity']
            self.placeLimitOrder(order['symbol'], quantity, order['is_buy_order'], order['limit_price'])
        elif msg.body['msg'] == "MASTER_ORDER_CANCELLED":
            self.cancelOrders()
        elif msg.body['msg'] == "ORDER_EXECUTED":
            order = msg.body['order']
            # print(self.id, order.to_dict())

    def cancelOrders(self):
        if not self.orders: return False
        for id, order in self.orders.items():
            self.cancelOrder(order)

        return True

    def getWakeFrequency(self):
        return pd.Timedelta(self.random_state.randint(low=0, high=100), unit='ns')
