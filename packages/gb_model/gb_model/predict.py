import pandas as pd

from gb_model import __version__
from gb_model.config import config, logging_config
from gb_model.tests import data_validation
from gb_model.processing import preprocessing, pipeline


logger = logging_config.get_logger(__name__)

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

    building = pd.read_pickle(config.DATA_PATH / 'building.pkl')
    meter = pd.DataFrame(meter_data)
    weather = pd.DataFrame(weather_data)

    data_validation.validate_data(meter, weather,
                                  config.METER_COLS, config.WEATHER_COLS)

    meter[time_var] = pd.to_datetime(meter[time_var], unit='ms')
    weather[time_var] = pd.to_datetime(weather[time_var], unit='ms')

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
                              config.MODEL_PATH / 'lgb')

    logger.info(
        f'Model version: {__version__}'
        f'\nInput meter data: {meter}'
        f'\nInput weather data: {weather}'
        f'\nPredictions: {pred}'
    )

    response = {'predictions': pred, 'version': __version__}
    return response
