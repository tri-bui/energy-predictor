import holidays
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WeatherExtractor(BaseEstimator, TransformerMixin):

	"""
	Feature extractor for weather-related variables. This is used to create a
	new feature for relative humidity and to convert compass direction into
	x- and y- components.

	:param dir_var: (string) name of direction variable
	:param air_var: (string) name of air temperature variable
	:param dew_var: (string) name of dew temperature variable
	"""

	def __init__(self,
	             dir_var='wind_direction',
	             air_var='air_temperature',
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
	Feature extractor for the datetime variable. This is used to extract
	day of year, day of week, and hour of day components from timestamps as
	well as a weekend boolean feature.

	:param time_var: (string) name of datetime variable
	"""

	def __init__(self,
	             time_var='timestamp'):
		self.time_var = time_var

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X = X.copy()
		X['dayofyear'] = X[self.time_var].dt.dayofyear
		X['dayofweek'] = X[self.time_var].dt.dayofweek
		X['hour'] = X[self.time_var].dt.hour
		X['is_weekend'] = X.dayofweek.apply(lambda d: int(d in [5, 6]))
		return X


class HolidayExtractor(BaseEstimator, TransformerMixin):

	"""
	Feature extractor for site-specific features. This is used to create
	country string and holiday boolean features.

	:param countries: (dictionary) mapping of site to country
	:param site_var: (string) name of site variable
	:param time_var: (string) name of datetime variable
	"""

	def __init__(self, countries,
	             site_var='site_id',
	             time_var='timestamp'):
		self.countries = countries
		self.site_var = site_var
		self.time_var = time_var
		self.USh = holidays.UnitedStates()
		self.CAh = holidays.Canada()
		self.UKh = holidays.UnitedKingdom()
		self.IEh = holidays.Ireland()

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X = X.copy()
		X['country'] = X[self.site_var].map(self.countries)
		US = X[X.country == 'US'].copy()
		US['is_holiday'] = US[self.time_var].map(lambda d: int(d in self.USh))
		CA = X[X.country == 'CA'].copy()
		CA['is_holiday'] = CA[self.time_var].map(lambda d: int(d in self.CAh))
		UK = X[X.country == 'UK'].copy()
		UK['is_holiday'] = UK[self.time_var].map(lambda d: int(d in self.UKh))
		IE = X[X.country == 'IE'].copy()
		IE['is_holiday'] = IE[self.time_var].map(lambda d: int(d in self.IEh))
		X = pd.concat([US, CA, UK, IE]).sort_index()
		return X


class FeatSelector(BaseEstimator, TransformerMixin):

	"""
	Feature selector.

	:param feats: (list of strings) features
	"""

	def __init__(self, feats):
		self.feats = feats

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X = X.copy()
		return X[self.feats]
