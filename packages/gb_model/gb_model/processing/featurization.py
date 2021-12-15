import holidays
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WeatherExtractor(BaseEstimator, TransformerMixin):

	"""
	Feature extractor for weather-related variables. This is used to:
	1. create a new feature for relative humidity using the air temperature and 
	   dew temperature
	2. convert wind direction (which is in compass direction) into x- and y- 
	   components

	Parameters
	----------
	dir_var : str, optional
		Name of wind direction variable, by default "wind_direction"
	air_var : str, optional
		Name of air temperature variable, by default "air_temperature"
	dew_var : str, optional
		Name of dew temperature variable, by default "dew_temperature"
	"""

	def __init__(self, dir_var='wind_direction', air_var='air_temperature',
	             dew_var='dew_temperature'):
		self.dir_var = dir_var
		self.air_var = air_var
		self.dew_var = dew_var
		self.dir = None
		self.e = None
		self.es = None

	def fit(self, X, y=None):
		self.dir = X[self.dir_var] * np.pi / 180
		self.e = \
			6.11 * 10.0 ** (7.5 * X[self.dew_var] / (237.3 + X[self.dew_var]))
		self.es = \
			6.11 * 10.0 ** (7.5 * X[self.air_var] / (237.3 + X[self.air_var]))
		return self

	def transform(self, X):
		X = X.copy()
		X['rel_humidity'] = self.e * 100 / self.es
		X[f'{self.dir_var}_x'] = np.cos(self.dir)
		X[f'{self.dir_var}_y'] = np.sin(self.dir)
		X.loc[X[self.dir_var] == 0, f'{self.dir_var}_x'] = 0
		X.drop(self.dir_var, axis=1, inplace=True)
		return X


class TimeExtractor(BaseEstimator, TransformerMixin):

	"""
	Feature extractor for the datetime variables. This is used to extract the 
	following datetime components from a timestamp:
	1. day of year
	2. day of week
	3. hour of day
	4. weekend indicator

	Parameters
	----------
	time_var : str, optional
		Name of timestamp variable, by default "timestamp"
	"""

	def __init__(self, time_var='timestamp'):
		self.time_var = time_var

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X = X.copy()
		X['dayofyear'] = X[self.time_var].dt.dayofyear
		X['dayofweek'] = X[self.time_var].dt.dayofweek
		X['hour'] = X[self.time_var].dt.hour
		X['is_weekend'] = X['dayofweek'].apply(lambda d: int(d in [5, 6]))
		return X


class HolidayExtractor(BaseEstimator, TransformerMixin):

	"""
	Feature extractor for site-specific features. This is used to create the 
	followingn features:
	1. country (2-letter code)
	2. holiday indicator based on date and country

	Parameters
	----------
	countries : dict{str: str}
		Mapping of site number to 2-letter country code
	site_var : str, optional
		Name of site variable, by default "site_id"
	time_var : str, optional
		Name of timestamp variable, by default "timestamp"
	"""

	def __init__(self, countries, site_var='site_id', time_var='timestamp'):
		self.countries = countries
		self.site_var = site_var
		self.time_var = time_var
		self.USh = None
		self.CAh = None
		self.UKh = None
		self.IEh = None

	def fit(self, X, y=None):
		self.USh = holidays.UnitedStates()
		self.CAh = holidays.Canada()
		self.UKh = holidays.UnitedKingdom()
		self.IEh = holidays.Ireland()
		return self

	def transform(self, X):
		X = X.copy()
		X['country'] = X[self.site_var].map(self.countries)
		US = X[X['country'] == 'US'].copy()
		US['is_holiday'] = US[self.time_var].map(lambda d: int(d in self.USh))
		CA = X[X['country'] == 'CA'].copy()
		CA['is_holiday'] = CA[self.time_var].map(lambda d: int(d in self.CAh))
		UK = X[X['country'] == 'UK'].copy()
		UK['is_holiday'] = UK[self.time_var].map(lambda d: int(d in self.UKh))
		IE = X[X['country'] == 'IE'].copy()
		IE['is_holiday'] = IE[self.time_var].map(lambda d: int(d in self.IEh))
		X = pd.concat([US, CA, UK, IE]).sort_index()
		return X


class FeatSelector(BaseEstimator, TransformerMixin):

	"""
	Feature filterer.

	Parameters
	----------
	feats : list[str]
		Features to keep
	"""

	def __init__(self, feats):
		self.feats = feats

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X = X.copy()
		return X[self.feats]
