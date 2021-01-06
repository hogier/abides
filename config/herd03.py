# HERD-3 (Herd Behavior Simulation Configuration):
# - 1     Exchange Agent
# - 1     POV Market Maker Agent
# - 1     Herd Master Agent
# - 60    Herd Slave Agent
# - 100   Value Agent
# - 500   Momentum Agent
# - 1000  ZI Agent
# - 500   HBL Agent
# - 5000  Noise Agent
# - 1     (Optional) POV Execution agent

import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt
from dateutil.parser import parse

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle

from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.HerdMasterAgent import HerdMasterAgent
from agent.HerdSlaveAgent import HerdSlaveAgent
from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from agent.HeuristicBeliefLearningAgent import HeuristicBeliefLearningAgent

from agent.market_makers.AdaptiveMarketMakerAgent import AdaptiveMarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent
from agent.execution.POVExecutionAgent import POVExecutionAgent
from model.LatencyModel import LatencyModel

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(description='Detailed options for HERD03 config.')

parser.add_argument('-c',
                    '--config',
                    required=True,
                    help='Name of config file to execute')
parser.add_argument('--start-time',
                    default='09:30:00',
                    type=parse,
                    help='Starting time of simulation.'
                    )
parser.add_argument('-l',
                    '--log_dir',
                    default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-s',
                    '--seed',
                    type=int,
                    default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v',
                    '--verbose',
                    action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')
# Execution agent config
parser.add_argument('-e',
                    '--execution-agents',
                    action='store_true',
                    help='Flag to allow the execution agent to trade.')
parser.add_argument('-p',
                    '--execution-pov',
                    type=float,
                    default=0.1,
                    help='Participation of Volume level for execution agent')
# market maker config
parser.add_argument('--mm-pov',
                    type=float,
                    default=0.025
                    )
parser.add_argument('--mm-window-size',
                    type=util.validate_window_size,
                    default='adaptive'
                    )
parser.add_argument('--mm-min-order-size',
                    type=int,
                    default=1
                    )
parser.add_argument('--mm-num-ticks',
                    type=int,
                    default=10
                    )
parser.add_argument('--mm-wake-up-freq',
                    type=str,
                    default='10S'
                    )
parser.add_argument('--mm-skew-beta',
                    type=float,
                    default=0
                    )
parser.add_argument('--mm-level-spacing',
                    type=float,
                    default=5
                    )
parser.add_argument('--mm-spread-alpha',
                    type=float,
                    default=0.75
                    )
parser.add_argument('--mm-backstop-quantity',
                    type=float,
                    default=50000)


args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

log_dir = args.log_dir  # Requested log directory.
seed = args.seed  # Random seed specification on the command line.
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

exchange_log_orders = True
log_orders = False
book_freq = None

simulation_start_time = dt.datetime.now()
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}\n".format(seed))
########################################################################################################################
############################################### AGENTS CONFIG ##########################################################

# Hyperparameters

historical_date = pd.to_datetime('2021-01-02')
symbol = 'ABM'
end_time = parse("11:30:00").strftime('%H:%M:%S')

starting_cash = 10000000  # Cash in this simulator is always in CENTS.

r_bar = 1e5
sigma_n = r_bar / 10
kappa = 1.67e-15
lambda_a = 7e-11
fund_vol = 1e-8

# Herd hyperparameters

min_delay = 1e+1
max_delay = 1e+9

future_window = 30e+9

size = 1

strategy = 'mixed'

# Historical date to simulate.
mkt_open = historical_date + pd.to_timedelta(args.start_time.strftime('%H:%M:%S'))
mkt_close = historical_date + pd.to_timedelta(end_time)
agent_count, agents, agent_types = 0, [], []

# Oracle
symbols = {symbol: {'r_bar': r_bar,
                    'kappa': 1.67e-16,
                    'agent_kappa': kappa,
                    'sigma_s': 0,
                    'fund_vol': fund_vol,
                    'megashock_lambda_a': 2.77778e-18,
                    'megashock_mean': 1e3,
                    'megashock_var': 5e4,
                    'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}}

oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

# 1) Exchange Agent

#  How many orders in the past to store for transacted volume computation
# stream_history_length = int(pd.to_timedelta(args.mm_wake_up_freq).total_seconds() * 100)
stream_history_length = 25000

agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=exchange_log_orders,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=stream_history_length,
                             book_freq=book_freq,
                             wide_book=True,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))])
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) Noise Agents
num_noise = 5000
noise_mkt_open = historical_date + pd.to_timedelta("09:00:00")  # These times needed for distribution of arrival times
                                                                # of Noise Agents
noise_mkt_close = historical_date + pd.to_timedelta("16:00:00")
agents.extend([NoiseAgent(id=j,
                          name="NoiseAgent {}".format(j),
                          type="NoiseAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          wakeup_time=util.get_wake_time(noise_mkt_open, noise_mkt_close),
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_noise)])
agent_count += num_noise
agent_types.extend(['NoiseAgent'])

# 3) Value Agents
num_value = 100
agents.extend([ValueAgent(id=j,
                          name="Value Agent {}".format(j),
                          type="ValueAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          sigma_n=sigma_n,
                          r_bar=r_bar,
                          kappa=kappa,
                          lambda_a=lambda_a,
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_value)])
agent_count += num_value
agent_types.extend(['ValueAgent'])

# 4) Market Maker Agents

"""
window_size ==  Spread of market maker (in ticks) around the mid price
pov == Percentage of transacted volume seen in previous `mm_wake_up_freq` that
       the market maker places at each level
num_ticks == Number of levels to place orders in around the spread
wake_up_freq == How often the market maker wakes up

"""

# each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
mm_params = [(args.mm_window_size, args.mm_pov, args.mm_num_ticks, args.mm_wake_up_freq, args.mm_min_order_size),
             (args.mm_window_size, args.mm_pov, args.mm_num_ticks, args.mm_wake_up_freq, args.mm_min_order_size)
             ]

num_mm_agents = len(mm_params)
mm_cancel_limit_delay = 50  # 50 nanoseconds

agents.extend([AdaptiveMarketMakerAgent(id=j,
                                name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(j),
                                type='AdaptivePOVMarketMakerAgent',
                                symbol=symbol,
                                starting_cash=starting_cash,
                                pov=mm_params[idx][1],
                                min_order_size=mm_params[idx][4],
                                window_size=mm_params[idx][0],
                                num_ticks=mm_params[idx][2],
                                wake_up_freq=mm_params[idx][3],
                                cancel_limit_delay=mm_cancel_limit_delay,
                                skew_beta=args.mm_skew_beta,
                                level_spacing=args.mm_level_spacing,
                                spread_alpha=args.mm_spread_alpha,
                                backstop_quantity=args.mm_backstop_quantity,
                                log_orders=log_orders,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))
               for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))])
agent_count += num_mm_agents
agent_types.extend('POVMarketMakerAgent')


# 5) Momentum Agents

num_momentum_agents = 500
agents.extend([MomentumAgent(id=j,
                             name="MOMENTUM_AGENT_{}".format(j),
                             type="MomentumAgent",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             min_size=1,
                             max_size=10,
                             log_orders=False,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))
               for j in range(agent_count, agent_count + num_momentum_agents)])
agent_types.extend("MomentumAgent")
agent_count += num_momentum_agents

# 6) Herd Master Agents

num_herd_masters = 1
agents.extend([HerdMasterAgent(id=j,
                          name="Herd Master Agent {}".format(j),
                          type="HerdMasterAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          sigma_n=0,
                          strategy=strategy,
                          future_window=future_window,
                          size=size,
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                    dtype='uint64')))
               for j in range(agent_count, agent_count + num_herd_masters)])
agent_count += num_herd_masters
agent_types.extend(['HerdMasterAgent'])

# 7) Herd Slave Agents

num_herd_slaves = 60
agents.extend([HerdSlaveAgent(id=j,
                          name="Herd Slave Agent {}".format(j),
                          type="HerdSlaveAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          min_delay=min_delay,
                          max_delay=max_delay,
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                    dtype='uint64')))
               for j in range(agent_count, agent_count + num_herd_slaves)])
agent_count += num_herd_slaves
agent_types.extend(['HerdSlaveAgent'])

# 8) Execution Agent

trade = True if args.execution_agents else False

#### Participation of Volume Agent parameters

pov_agent_start_time = mkt_open + pd.to_timedelta('00:30:00')
pov_agent_end_time = mkt_close - pd.to_timedelta('00:30:00')
pov_proportion_of_volume = args.execution_pov
pov_quantity = 12e5
pov_frequency = '1min'
pov_direction = "BUY"

pov_agent = POVExecutionAgent(id=agent_count,
                              name='POV_EXECUTION_AGENT',
                              type='ExecutionAgent',
                              symbol=symbol,
                              starting_cash=starting_cash,
                              start_time=pov_agent_start_time,
                              end_time=pov_agent_end_time,
                              freq=pov_frequency,
                              lookback_period=pov_frequency,
                              pov=pov_proportion_of_volume,
                              direction=pov_direction,
                              quantity=pov_quantity,
                              trade=trade,
                              log_orders=True,  # needed for plots so conflicts with others
                              random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))

execution_agents = [pov_agent]
agents.extend(execution_agents)
agent_types.extend("ExecutionAgent")
agent_count += 1

# 9) Zero Intelligence Agents

num_zi_agents = 1000
agents.extend([ZeroIntelligenceAgent(id=j,
                                     name="ZI_AGENT_{}".format(j),
                                     type="ZeroIntelligenceAgent",
                                     symbol=symbol,
                                     starting_cash=starting_cash,
                                     sigma_n=10000,
                                     sigma_s=symbols[symbol]['fund_vol'],
                                     kappa=symbols[symbol]['agent_kappa'],
                                     r_bar=symbols[symbol]['r_bar'],
                                     q_max=10,
                                     sigma_pv=5e4,
                                     R_min=0,
                                     R_max=100,
                                     eta=1,
                                     lambda_a=1e-12,
                                     log_orders=False,
                                     random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                               dtype='uint64')))
               for j in range(agent_count, agent_count + num_zi_agents)])
agent_types.extend("ZeroIntelligenceAgent")
agent_count += num_zi_agents

# 10) Heuristic Belief Learning Agents
num_hbl_agents = 500
agents.extend([HeuristicBeliefLearningAgent(id=j,
                                            name="HBL_AGENT_{}".format(j),
                                            type="HeuristicBeliefLearningAgent",
                                            symbol=symbol,
                                            starting_cash=starting_cash,
                                            sigma_n=10000,
                                            sigma_s=symbols[symbol]['fund_vol'],
                                            kappa=symbols[symbol]['agent_kappa'],
                                            r_bar=symbols[symbol]['r_bar'],
                                            q_max=10,
                                            sigma_pv=5e4,
                                            R_min=0,
                                            R_max=100,
                                            eta=1,
                                            lambda_a=1e-12,
                                            L=2,
                                            log_orders=False,
                                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                      dtype='uint64')))
               for j in range(agent_count, agent_count + num_hbl_agents)])
agent_types.extend("HeuristicBeliefLearningAgent")
agent_count += num_hbl_agents

########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("HERD03 Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                  dtype='uint64')))

kernelStartTime = historical_date
kernelStopTime = mkt_close + pd.to_timedelta('00:01:00')

defaultComputationDelay = 50  # 50 nanoseconds

# LATENCY

latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2**32))
pairwise = (agent_count, agent_count)

# All agents sit on line from Seattle to NYC
nyc_to_seattle_meters = 3866660
pairwise_distances = util.generate_uniform_random_pairwise_dist_on_line(0.0, nyc_to_seattle_meters, agent_count,
                                                                        random_state=latency_rstate)
pairwise_latencies = util.meters_to_light_ns(pairwise_distances)

model_args = {
    'connected': True,
    'min_latency': pairwise_latencies
}

latency_model = LatencyModel(latency_model='deterministic',
                             random_state=latency_rstate,
                             kwargs=model_args
                             )
# KERNEL

kernel.runner(agents=agents,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              agentLatencyModel=latency_model,
              defaultComputationDelay=defaultComputationDelay,
              oracle=oracle,
              log_dir=args.log_dir)


simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
