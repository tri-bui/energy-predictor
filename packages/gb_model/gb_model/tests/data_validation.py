def validate_cols(df, cols):
    """
	Check if input data has the required columns and that the column does not
	have the "object" type.

	:param df: (Pandas dataframe) input data
	:param cols: (list) required columns
	:return: None
	"""

    for col in cols:
        assert col in df.columns, f'Missing {col} column.'
        assert 'object' not in str(df[col].dtype), \
            f'Type of {col} column not supported.'


def validate_meter_vals(meter_df):
    """
	Check for bad values in the input meter data.

	:param meter_df: (Pandas dataframe) input meter data
	:return: None
	"""

    assert meter_df['building_id'].min() >= 0 and \
           meter_df['building_id'].max() <= 1448, 'Bad building_id value.'
    assert meter_df['meter'].min() >= 0 and \
           meter_df['meter'].max() <= 3, 'Bad meter value.'


def validate_weather_vals(weather_df):
    """
	Check for bad values in the input weather data.

	:param weather_df: (Pandas dataframe) input weather data
	:return: None
	"""

    assert weather_df['site_id'].min() >= 0 and \
           weather_df['site_id'].max() <= 15, 'Bad site_id value.'
    assert weather_df['air_temperature'].min() >= -95 and \
           weather_df['air_temperature'].max() <= 95, \
        'Bad air_temperature value.'
    assert weather_df['dew_temperature'].min() >= -95 and \
           weather_df['dew_temperature'].max() <= 95, \
        'Bad dew_temperature value.'
    assert weather_df['sea_level_pressure'].min() >= 800 and \
           weather_df['sea_level_pressure'].max() <= 1100, \
        'Bad sea_level_pressure value.'
    assert weather_df['wind_speed'].min() >= 0, 'Bad wind_speed value.'
    assert weather_df['wind_direction'].min() >= 0 and \
           weather_df['wind_direction'].max() <= 360, \
        'Bad wind_direction value.'


def validate_data(meter_df, weather_df, meter_cols, weather_cols):
    """
	Validate input meter and weather data.

	:param meter_df: (Pandas dataframe) input meter data
	:param weather_df: (Pandas dataframe) input weather data
	:param meter_cols: (list) required meter columns
	:param weather_cols: (list) required weather columns
	:return: None
	"""

    validate_cols(meter_df, meter_cols)
    validate_cols(weather_df, weather_cols)
    validate_meter_vals(meter_df)
    validate_weather_vals(weather_df)
