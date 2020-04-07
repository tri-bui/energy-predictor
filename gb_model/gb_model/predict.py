import pandas as pd

from gb_model import __version__
from gb_model.config import config, logging_config
from gb_model.tests import data_validation
from gb_model.processing import preprocessing, pipeline


logger = logging_config.get_logger(__name__)


building = pd.read_pickle(config.DATA_PATH / 'building.pkl')
wpipe = pipeline.wthr_pipe
fpipe = pipeline.feat_pipe


def make_prediction(meter_data, weather_data, time_var='timestamp'):

	"""
	Make predictions from meter and weather data.

	:param meter_data: (dictionary) raw meter data in JSON format
	:param weather_data: (dictionary) raw weather data in JSON format
	:param time_var: (str) name of datetime variable

	:return: (dictionary) predictions in JSON format
	"""

	meter = pd.read_json(meter_data)
	weather = pd.read_json(weather_data)

	data_validation.validate_data(meter, weather,
	                              config.METER_COLS, config.WEATHER_COLS)

	start = weather[time_var].min()
	end = weather[time_var].max()

	weather = wpipe.fit_transform(weather,
	                              time_reindexer__t_start=start,
	                              time_reindexer__t_end=end)

	df = preprocessing.merge_data(meter, weather, building)
	df = fpipe.fit_transform(df)

	assert df.isnull().sum().sum() == 0, 'Missing values detected.'

	pred = pipeline.pred_pipe(df,
							  config.MODEL_PATH / 'rare_enc',
							  config.MODEL_PATH / 'mean_enc',
							  config.MODEL_PATH / 'scaler',
							  config.MODEL_PATH / 'lgb',
							  use_xgb=False)

	logger.info(
		f'Model version: {__version__}'
		f'Input meter: {meter}'
		f'Input weather data: {weather}'
		f'Predictions: {pred}'
	)

	response = {'predictions': pred}
	return response