import pytest
import pathlib
import numpy as np
import pandas as pd

from gb_model.processing import features
from gb_model.config import config


TEST_PATH = pathlib.Path(__file__).resolve().parent
DATA_PATH = TEST_PATH / 'datasets'

df = pd.read_pickle(DATA_PATH / 'd_features.pkl')
print(df.info())

"""
This dataset is the combined building, weather, and meter datasets. 
It contains the first 2 days of buildings 0 and 1448. There are no
missing values but there are 2 missing timestamps in building 1448.
Data shape: [94, 16]
"""


@pytest.fixture
def weather_extractor():
    we = features.WeatherExtractor()
    we.fit(df)
    return we


@pytest.fixture
def time_extractor():
    return features.TimeExtractor()


@pytest.fixture
def holiday_extractor():
    return features.HolidayExtractor(config.COUNTRIES)


@pytest.fixture
def feat_selector():
    return features.FeatSelector(config.FEATS[:4] + config.FEATS[-2:])


def test_WeatherExtractor(weather_extractor):

    """
    This transformer drops 1 feature and creates 3 more.
    After the transformation, the number of columns should
    increase to 18.
    """

    X = weather_extractor.transform(df)
    assert X.shape == (94, 18)
    assert np.round(X.loc[0, 'rel_humidity'], 2) == 83.41
    assert np.round(X.loc[48, 'rel_humidity'], 2) == 81.76
    assert np.round(X.loc[5, 'wind_direction_x'], 2) == 0.17
    assert np.round(X.loc[50, 'wind_direction_x'], 2) == 0.50
    assert np.round(X.loc[1, 'wind_direction_y'], 2) == 0.50
    assert np.round(X.loc[49, 'wind_direction_y'], 2) == -0.87


def test_TimeExtractor(time_extractor):

    """
    This transformer creates 4 new features. After the transformation,
    the number of columns should increase to 20.
    """

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

    """
    This transformer creates 2 new features. After the transformation,
    the number of columns should increase to 18.
    """

    X = holiday_extractor.transform(df)
    assert X.shape == (94, 18)
    assert X.loc[0, 'country'] == X.loc[93, 'country'] == 'US'
    assert X.loc[1, 'is_holiday'] == X.loc[92, 'is_holiday'] == 1


def test_FeatSelector(feat_selector):
    X = feat_selector.transform(df)
    assert X.shape == (94, 6)
