import pandas as pd
from gb_model import __version__
from gb_model.config import config, logging_config
from gb_model.tests import data_validation
from gb_model.processing import preprocessing, pipeline


logger = logging_config.get_logger(__name__) # logger
wpipe = pipeline.wthr_pipe # weather data pipeline
fpipe = pipeline.feat_pipe # feature pipeline


def make_prediction(meter_data, weather_data, time_var='timestamp'):

    """
    Make meter reading predictions from meter and weather data.

    Parameters
    ----------
    meter_data : JSON
        Raw meter data
    weather_data : JSON
        Raw weather data
    time_var : str, optional
        Name of timestamp variable, by default 'timestamp'

    Returns
    -------
    JSON
        Meter reading predictions and model version
    """

    # Read and validate data
    building = pd.read_pickle(config.DATA_PATH / 'building.pkl')
    meter = pd.DataFrame(meter_data)
    weather = pd.DataFrame(weather_data)
    data_validation.validate_data(meter, weather,
                                  config.METER_COLS, config.WEATHER_COLS)

    # Convert timestamps to dt
    meter[time_var] = pd.to_datetime(meter[time_var], unit='ms')
    weather[time_var] = pd.to_datetime(weather[time_var], unit='ms')

    # Preprocess weather data
    start = weather[time_var].min()
    end = weather[time_var].max()
    weather = wpipe.fit_transform(weather,
                                  time_reindexer__t_start=start,
                                  time_reindexer__t_end=end)

    # Merge data and select feats
    df = preprocessing.merge_data(meter, weather, building)
    df = fpipe.fit_transform(df)

    # Check for missing vals
    assert df.isnull().sum().sum() == 0, 'Missing values detected.'

    # Make preds
    pred = pipeline.pred_pipe(df,
                              config.MODEL_PATH / 'rare_enc',
                              config.MODEL_PATH / 'mean_enc',
                              config.MODEL_PATH / 'scaler',
                              config.MODEL_PATH / 'lgb', 
                              use_model0=True, use_xgb=False, 
                              inverse_transform_site0=False)

    # Log - model version, input data, and preds
    logger.info(
        f'Model version: {__version__}'
        f'\nInput meter data: {meter}'
        f'\nInput weather data: {weather}'
        f'\nPredictions: {pred}'
    )

    # Format preds
    response = {'predictions': pred, 'version': __version__}
    return response
