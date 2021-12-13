import pytest
import pathlib
import numpy as np
import pandas as pd
from gb_model.processing import features
from gb_model.config import config


# Data path
TEST_PATH = pathlib.Path(__file__).resolve().parent # gb_model/tests/
DATA_PATH = TEST_PATH / 'datasets' # gb_model/tests/datasets/

# Combined data subset for testing featurization
df = pd.read_pickle(DATA_PATH / 'd_features.pkl')
print(df.info())


"""
This dataset is the combined building, weather, and meter datasets. It contains 
the first 2 days of data from buildings 0 and 1448. There are no missing values 
but there are 2 missing timestamps from building 1448.

Data shape: [94, 16]
"""


@pytest.fixture
def weather_extractor():

    """ Fit a WeatherExtractor that creates a feature for relative humidity and 
    converts wind direction from compass degrees to x- and y-components. """

    we = features.WeatherExtractor()
    we.fit(df)
    return we


@pytest.fixture
def time_extractor():

    """ Instantiate a TimeExtractor that extracts time components from the 
    timestamp feature. """

    return features.TimeExtractor()


@pytest.fixture
def holiday_extractor():

    """ Instantiate a HolidayExtractor that creates a country feature and a 
    holiday binary indicator using the site and date from the timestamp. """

    return features.HolidayExtractor(config.COUNTRIES)


@pytest.fixture
def feat_selector():

    """ Instantiate a FeatSelector that selects the first 4 and last 2 features 
    from the `config` module. """

    return features.FeatSelector(config.FEATS[:4] + config.FEATS[-2:])


def test_WeatherExtractor(weather_extractor):

    """ This transformer drops 1 feature and creates 3 more.
    After the transformation, the number of columns should
    increase to 18. """

    X = weather_extractor.transform(df)
    assert X.shape == (94, 18)
    assert np.round(X.loc[0, 'rel_humidity'], 2) == 83.41
    assert np.round(X.loc[48, 'rel_humidity'], 2) == 81.76
    assert np.round(X.loc[5, 'wind_direction_x'], 2) == 0.17
    assert np.round(X.loc[50, 'wind_direction_x'], 2) == 0.50
    assert np.round(X.loc[1, 'wind_direction_y'], 2) == 0.50
    assert np.round(X.loc[49, 'wind_direction_y'], 2) == -0.87


def test_TimeExtractor(time_extractor):

    """ This transformer creates 4 new features. After the transformation,
    the number of columns should increase to 20. """

    X = time_extractor.transform(df)
    assert X.shape == (94, 20)
    assert X.loc[0, 'dayofyear'] == 1
    assert X.loc[93, 'dayofyear'] == 2
    assert X.loc[1, 'dayofweek'] == 6
    assert X.loc[92, 'dayofweek'] == 0
    assert X.loc[2, 'hour'] == 2
    assert X.loc[91, 'hour'] == 21
    assert X.loc[3, 'is_weekend'] == 1
    assert X.loc[90, 'is_weekend'] == 0


def test_HolidayExtractor(holiday_extractor):

    """ This transformer creates 2 new features. After the transformation,
    the number of columns should increase to 18. """

    X = holiday_extractor.transform(df)
    assert X.shape == (94, 18)
    assert X.loc[0, 'country'] == X.loc[93, 'country'] == 'US'
    assert X.loc[1, 'is_holiday'] == X.loc[92, 'is_holiday'] == 1


def test_FeatSelector(feat_selector):
    X = feat_selector.transform(df)
    assert X.shape == (94, 6)
