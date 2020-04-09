import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TimeConverter(BaseEstimator, TransformerMixin):

	"""
	UTC-to-local timestamp converter.

	:param timezones: (list of integers) timezone offsets
	:param site_var: (string) name of site variable
	:param time_var: (string) name of datetime variable
	"""

	def __init__(self, timezones,
	             site_var='site_id',
	             time_var='timestamp'):
		self.timezones = timezones
		self.site_var = site_var
		self.time_var = time_var
		self.offsets = None

	def fit(self, X, y=None):
		self.offsets = X[self.site_var].map(
			lambda s: np.timedelta64(self.timezones[s], 'h'))
		return self

	def transform(self, X):
		X = X.copy()
		X[self.time_var] += self.offsets
		return X


class TimeReindexer(BaseEstimator, TransformerMixin):

	"""
	Site and datetime reindexer to include a timestamp for every hour within
	the time interval at every site.

	:param site_var: (string) name of site variable
	:param time_var: (string) name of datetime variable

	:param t_start: (datetime string in the format 'YYYY-MM-DD hh:mm:ss')
					first timestamp in new index
	:param t_end: (datetime string in the format 'YYYY-MM-DD hh:mm:ss')
				  last timestamp in new index
	"""

	def __init__(self,
	             site_var='site_id',
	             time_var='timestamp'):
		# t_start and t_end are strings in the format 'YYYY-MM-DD hh:mm:ss'
		self.site_var = site_var
		self.time_var = time_var
		self.sites = None
		self.t_start = None
		self.t_end = None

	def fit(self, X,
	        t_start='2017-01-01 00:00:00',
	        t_end='2018-12-31 23:00:00',
	        y=None):
		self.sites = X[self.site_var].unique()
		self.t_start = t_start
		self.t_end = t_end
		return self

	def transform(self, X):
		X = X.copy()
		X = X.set_index([self.site_var, self.time_var])
		X = X.reindex(
			pd.MultiIndex.from_product([
				self.sites,
				pd.date_range(start=self.t_start, end=self.t_end, freq='H')
			])
		)
		X.index.rename([self.site_var, self.time_var], inplace=True)
		X.reset_index(inplace=True)
		return X


class MissingImputer(BaseEstimator, TransformerMixin):

	"""
	Missing value imputer using cubic and linear interpolation.

	:param cub_vars: (list of strings) variable names for cubic interpolation
	:param lin_vars: (list of strings) variable names for linear interpolation
	:param site_var: (string) name of site variable
	"""

	def __init__(self, cub_vars, lin_vars,
	             site_var='site_id'):
		self.cub_vars = cub_vars
		self.lin_vars = lin_vars
		self.site_var = site_var

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X = X.copy()
		for var in X.columns:
			imethod = 'cubic' if var in self.cub_vars else 'linear'
			if var in self.cub_vars + self.lin_vars:
				X[var] = X.groupby(self.site_var)[var].transform(
					lambda s: s.interpolate(imethod, limit_direction='both'))
			X[var] = X.groupby(self.site_var)[var].transform(
				lambda s: s.fillna(method='ffill').fillna(method='bfill'))
		return X


class DataCopier(BaseEstimator, TransformerMixin):

	"""
	Data copier to a column from one site to another site. This is used to fill
	missing data in sites with 100% missing values.

	:param site_var: (string) name of site variable

	:param var_to_copy: (string) name of variable to copy
	:param copy_from_site: (integer) site to copy data from
	:param copy_to_site: (integer) site to copy data to
	"""

	def __init__(self,
	             site_var='site_id'):
		self.site_var = site_var
		self.var_to_copy = None
		self.copy_from_site = None
		self.copy_to_site = None
		self.from_idx = None
		self.to_idx = None

	def fit(self, X,
	        var_to_copy='sea_level_pressure',
	        copy_from_site=1,
	        copy_to_site=5,
	        y=None):
		self.var_to_copy = var_to_copy
		self.copy_from_site = copy_from_site
		self.copy_to_site = copy_to_site
		self.from_idx = X[X[self.site_var] == self.copy_from_site].index
		self.to_idx = X[X[self.site_var] == self.copy_to_site].index
		return self

	def transform(self, X):
		X = X.copy()
		X.loc[self.to_idx, self.var_to_copy] = \
			X.loc[self.from_idx, self.var_to_copy].values
		return X


def merge_data(meter_df, weather_df, building_df,
               on_mb='building_id', on_mbw=['site_id', 'timestamp']):

	"""
	Combine the meter, weather, and building data.

	:param meter_df: (Pandas dataframe) meter data
	:param weather_df: (Pandas dataframe) weather data
	:param building_df: (Pandas dataframe) building data
	:param on_mb: (string or list of strings) variable name(s) to merge
				  meter_df and building_df on
	:param on_mbw: (string or list of strings) variable name(s) to merge the
				   resulting dataframe and weather_df on

	:return: dataframe containing meter, weather, and building data
	"""

	meter = meter_df.copy()
	weather = weather_df.copy()
	building = building_df.copy()
	mb = pd.merge(meter, building, on=on_mb, how='left')
	mbw = pd.merge(mb, weather, on=on_mbw, how='left')
	return mbw
