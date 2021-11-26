def validate_cols(df, cols):

    """
    Check if the input data has the required columns and that there are no 
    columns of "object" type.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Input data
    cols : list[str]
        Required columns
    """

    for col in cols:
        assert col in df.columns, f'Missing {col} column.'
        assert 'object' not in str(df[col].dtype), \
            f'Type of {col} column not supported.'


def validate_meter_vals(meter_df):

    """
	Check for bad values in the input meter data. A value is bad if it is out 
    of range for that particular feature. The meter feature ranges are:
    - `building_id` - between 0 and 1448
    - `meter` - between 0 and 3

    Parameters
    ----------
    meter_df : pandas.core.frame.DataFrame
        Input meter data
    """

    assert meter_df['building_id'].min() >= 0 and \
           meter_df['building_id'].max() <= 1448, 'Bad building_id value.'
    assert meter_df['meter'].min() >= 0 and \
           meter_df['meter'].max() <= 3, 'Bad meter value.'


def validate_weather_vals(weather_df):

    """
	Check for bad values in the input weather data. A value is bad if it is out 
    of range for that particular features. The weather feature ranges are:
    - `site_id` - between 0 and 15
    - `air_temperature` - between -95 and 95
    - `dew_temperature` - between -95 and 95
    - `sea_level_pressure` - between 800 and 1100
    - `wind_speed` - non-negative
    - `wind_direction` - between 0 and 360

    Parameters
    ----------
    weather_df : pandas.core.frame.DataFrame
        Input weather data
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

    Parameters
    ----------
    meter_df : pandas.core.frame.DataFrame
        Input meter data
    weather_df : pandas.core.frame.DataFrame
        Input weather data
    meter_cols : list[str]
        Required meter columns
    weather_cols : list[str]
        Required weather columns
    """

    validate_cols(meter_df, meter_cols)
    validate_meter_vals(meter_df)
    validate_cols(weather_df, weather_cols)
    validate_weather_vals(weather_df)
