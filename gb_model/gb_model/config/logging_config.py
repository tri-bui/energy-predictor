import sys
import logging
from logging.handlers import TimedRotatingFileHandler

from gb_model.config import config


LOG_FILE = config.LOG_PATH / 'model.log'

FORMATTER = logging.Formatter(
	'%(levelname)s - %(asctime)s - %(name)s - '
	'%(funcName)s:%(lineno)d - %(message)s'
)


def get_console_handler():
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(FORMATTER)
	return console_handler


def get_file_handler():
	file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
	file_handler.setFormatter(FORMATTER)
	file_handler.setLevel(logging.INFO)
	return file_handler


def get_logger(logger_name):
	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.DEBUG)

	logger.addHandler(get_console_handler())
	logger.addHandler(get_file_handler())

	logger.propagate = False
	return logger
