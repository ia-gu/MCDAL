from datetime import datetime

# log file path
CHECKPOINT_PATH = 'logs/budget_fixed'

EPOCH = 200

DATE_FORMAT = '%Y%m%d_%H%M%S'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)