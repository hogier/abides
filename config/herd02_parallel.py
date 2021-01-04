import argparse
import os
from multiprocessing import Pool
import psutil
import datetime as dt
import numpy as np


def run_in_parallel(num_simulations, num_parallel, config, log_folder, verbose,
                    symbol, historical_date, end_time, master_window, slave_min_delay, slave_max_delay, fund_vol
                    ):

    global_seeds = np.random.randint(0, 2 ** 32, num_simulations)
    print(f'Global Seeds: {global_seeds}')
    processes = [f'python -u abides.py -c {config} -l {log_folder}_seed_{seed} {"-v" if verbose else ""} -s {seed} -t {symbol} -d {historical_date}  --end-time {end_time} --master-window {master_window} --slave-min-delay {slave_min_delay} --slave-max-delay {slave_max_delay} --fund-vol {fund_vol}'
                 for seed in global_seeds]

    pool = Pool(processes=num_parallel)
    pool.map(run_process, processes)


def run_process(process):
    os.system(process)


if __name__ == "__main__":
    start_time = dt.datetime.now()

    parser = argparse.ArgumentParser(description='Main config to run multiple ABIDES simulations in parallel')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed controlling the generated global seeds')
    parser.add_argument('--num_simulations', type=int, default=1,
                        help='Total number of simulations to run')
    parser.add_argument('--num_parallel', type=int, default=None,
                        help='Number of simulations to run in parallel')
    parser.add_argument('--config', required=True,
                        help='Name of config file to execute')
    parser.add_argument('--log_folder', required=True,
                        help='Log directory name')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Maximum verbosity!')
    parser.add_argument('-t',
                        '--ticker',
                        required=True,
                        help='Ticker (symbol) to use for simulation')
    parser.add_argument('-d', '--historical-date',
                        required=True,
                        type=str,
                        help='historical date being simulated in format YYYYMMDD.')
    parser.add_argument('--end-time',
                        default='11:30:00',
                        type=str,
                        help='Ending time of simulation.'
                        )

    parser.add_argument('--fund-vol',
                        type=float,
                        default=1e-8,
                        help='Volatility of fundamental time series.'
                        )

    parser.add_argument('--master-window',
                        type=float,
                        default=1e+10,
                        help='Herd Master wakeup frequency.'
                        )

    parser.add_argument('--slave-min-delay',
                        type=float,
                        default=1e+2,
                        help='Herd Slave min delay.'
                        )
    parser.add_argument('--slave-max-delay',
                        type=float,
                        default=1e+8,
                        help='Herd Slave max delay.'
                        )

    args, remaining_args = parser.parse_known_args()

    seed = args.seed
    num_simulations = args.num_simulations
    num_parallel = args.num_parallel if args.num_parallel else psutil.cpu_count() # count of the CPUs on the machine
    config = args.config
    log_folder = args.log_folder
    verbose = args.verbose

    symbol = args.ticker
    historical_date = args.historical_date
    end_time = args.end_time
    master_window = args.master_window
    slave_min_delay = args.slave_min_delay
    slave_max_delay = args.slave_max_delay
    fund_vol = args.fund_vol


    print(f'Total number of simulation: {num_simulations}')
    print(f'Number of simulations to run in parallel: {num_parallel}')
    print(f'Configuration: {config}')

    np.random.seed(seed)

    run_in_parallel(num_simulations=num_simulations,
                    num_parallel=num_parallel,
                    config=config,
                    log_folder=log_folder,
                    verbose=verbose,
                    symbol=symbol,
                    historical_date =historical_date,
                    end_time = end_time,
                    master_window = master_window,
                    slave_min_delay = slave_min_delay,
                    slave_max_delay = slave_max_delay,
                    fund_vol = fund_vol
                    )

    end_time = dt.datetime.now()
    print(f'Total time taken to run in parallel: {end_time - start_time}')
