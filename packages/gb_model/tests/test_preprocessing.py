import pytest
import pathlib
import datetime
import pandas as pd
from gb_model.processing import preprocessing
from gb_model.config import config


# Data path
TEST_PATH = pathlib.Path(__file__).resolve().parent # gb_model/tests/
DATA_PATH = TEST_PATH / 'datasets' # gb_model/tests/datasets

# Meter data subset for testing
meter = pd.read_pickle(DATA_PATH / 'm_preprocessing.pkl')
print(meter.info())

# Weather data subset for testing
weather = pd.read_pickle(DATA_PATH / 'w_preprocessing.pkl')
print(weather.info())

# Building data subset for testing
building = pd.read_pickle(DATA_PATH / 'b_preprocessing.pkl')
print(building.info())


"""
The weather dataset contains the first 2 days of sites 0 and 15. There are 2 
missing timestamps and all columns with weather measurements have some missing 
data. The 'precip_depth_1_hr' column is missing 100% of the data in site 0.

Data shape: [94, 9]
"""


@pytest.fixture
def time_converter():

    """ Fit a TimeConverter that convert UTC to local time. """

    tc = preprocessing.TimeConverter(config.TZ_OFFSETS)
    tc.fit(weather)
    return tc


@pytest.fixture
def time_reindexer():

    """ Fit a TimeReindexer that reindexes the weather data from 
    "2017-01-01 00:00:00" to "2017-01-02 23:00:00". """

    tr = preprocessing.TimeReindexer()
    tr.fit(weather, t_start='2017-01-01 00:00:00', t_end='2017-01-02 23:00:00')
    return tr


@pytest.fixture
def missing_imputer():

    """ Instantiate a MissingImputer that imputes missing data using cubic or 
    linear interpolation, followed by a forward and back fill. """

    return preprocessing.MissingImputer(config.CUB_VARS, config.LIN_VARS)


@pytest.fixture
def data_copier(time_reindexer):

    """ Fit a DataCopier to copy weather data from site 0 to site 15. The 
    weather data is reindexed before the copy. """

    X = time_reindexer.transform(weather)
    dc = preprocessing.DataCopier()
    dc.fit(X, copy_from_site=0, copy_to_site=15)
    X = dc.transform(X)
    return dc, X


def test_TimeConverter(time_converter):

    """
    Both sites have a timezone offset of -5. After converting,
    01/01/2017 00:00:00 should become 12/31/2016 19:00:00 and
    01/02/2017 23:00:00 should become 01/02/2017 18:00:00
    """

    X = time_converter.transform(weather)
    assert X.loc[0, 'timestamp'] == datetime.datetime(2016, 12, 31, 19)
    assert X.loc[93, 'timestamp'] == datetime.datetime(2017, 1, 2, 18)


def test_TimeReindexer(time_reindexer):

    """
    Since there are 2 missing timestamps, reindexing with the same
    start and end should add 2 more rows. The first and last rows
    should be the same as the passed in timestamps.
    """

    X = time_reindexer.transform(weather)
    assert X.shape[0] == 96
    assert X.loc[0, 'timestamp'] == datetime.datetime(2017, 1, 1, 0)
    assert X.loc[95, 'timestamp'] == datetime.datetime(2017, 1, 2, 23)


def test_MissingImputer(missing_imputer):

    """
    This imputer is only able to fill missing values in columns where
    there is some data present by a per-site basis. 1 column is missing
    100% data in site 0 so that is the only one that will not be filled.
    After imputing, there should be 48 missing values left, all of which
    should be coming from the 'precip_depth_1_hr' column.
    """

    X = missing_imputer.transform(weather)
    assert X['precip_depth_1_hr'].isnull().sum() == 48
    assert X.isnull().sum().sum() == 48


def test_DataCopier(data_copier):

    """
    Site 0 has 48 rows and site 15 only has 46 so the data has to be
    reindexed before applying this transformer. After reindexing,
    there should be a total of 96 rows. After copying the data,
    'sea_level_pressure' data should be the same for sites 0 and 15.
    """

    dc, X = data_copier
    s0_slp = weather.loc[weather[dc.site_var] == dc.copy_from_site, dc.var_to_copy]
    s15_slp = X.loc[X[dc.site_var] == dc.copy_to_site, dc.var_to_copy]
    assert X.shape[0] == 96
    assert (s0_slp - s15_slp).sum() == 0


def test_merge_data():
    X = preprocessing.merge_data(meter, weather, building)
    assert X.shape == (94, 16)
