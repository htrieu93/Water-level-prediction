import argparse
import logging.config
from src import config

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger('loggers')

parser = argparse.ArgumentParser(
                    prog='WaterLevelPrediction',
                    description='Setting up global variable')

parser.add_argument('-s', '--scenario', type=int)
parser.add_argument('-n', '--n_steps', type=int)
parser.add_argument('-l', '--lead_time', type=int)
parser.add_argument('-t', '--target', type=int)
parser.add_argument('-m', '--model', type=int)

args = parser.parse_args()

scenario = args.scenario
lead_time = args.lead_time
n_steps = args.n_steps
target = args.target
model_name = args.model