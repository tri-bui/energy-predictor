import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TimeConverter(BaseEstimator, TransformerMixin):

	"""
	UTC-to-local timestamp converter.

	Parameters
	----------
	timezones : list[int]
		Timezone offsets to convert UTC to local time
	site_var : str, optional
		Name of site variable, by default "site_id"
	time_var : str, optional
		Name of datetime variable, by default "timestamp"
	"""

	def __init__(self, timezones, site_var='site_id', time_var='timestamp'):
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
	Dataframe reindexer to include a timestamp for every hour within the time 
	interval at every site.

	Ex: 16 sites, 366 days => 16 x 366 x 24 timestamps

	Parameters
	----------
	site_var : str, optional
		Name of site variable, by default "site_id"
	time_var : str, optional
		Name of datetime variable, by default "timestamp"

	t_start : str, optional
		First timestamp in new index with the format "YYYY-MM-DD hh:mm:ss", by 
		default "2017-01-01 00:00:00"
	t_end : str, optional
		Last timestamp in new index with the format "YYYY-MM-DD hh:mm:ss", by 
		default "2018-12-31 23:00:00"
	"""

	def __init__(self, site_var='site_id', time_var='timestamp'):
		self.site_var = site_var
		self.time_var = time_var
		self.sites = None
		self.t_start = None
		self.t_end = None

	def fit(self, X, y=None, t_start='2017-01-01 00:00:00', 
			t_end='2018-12-31 23:00:00'):
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

	Parameters
	----------
	cub_vars : list[str]
		Names of variables to perform cubic interpolation for
	lin_vars : list[str]
		Names of variables to perform linear interpolation for
	site_var : str, optional
		Name of site variable, by default "site_id"
	"""

	def __init__(self, cub_vars, lin_vars, site_var='site_id'):
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
	Data copier of a variable from one site to another site. This is used to 
	fill missing data in sites with 100% missing values. The data being copied 
	is coming from a site where the data is expected to be similar.

	Parameters
	----------
	site_var : str, optional
		Name of site variable, by default "site_id"

	var_to_copy : str, optional
		Name of variable to copy data from, by default "sea_level_pressure"
	from_site : int, optional
		Site of data to copy from, by default 1
	to_site : int, optional
		Site of data to copy to, by default 5
	"""

	def __init__(self, site_var='site_id'):
		self.site_var = site_var
		self.var_to_copy = None
		self.from_site = None
		self.to_site = None
		self.from_idx = None
		self.to_idx = None

	def fit(self, X, y=None, var_to_copy='sea_level_pressure', from_site=1, 
			to_site=5):
		self.var_to_copy = var_to_copy
		self.from_site = from_site
		self.to_site = to_site
		self.from_idx = X[X[self.site_var] == self.from_site].index
		self.to_idx = X[X[self.site_var] == self.to_site].index
		return self

	def transform(self, X):
		X = X.copy()
		X.loc[self.to_idx, self.var_to_copy] = \
			X.loc[self.from_idx, self.var_to_copy].values
		return X


def merge_data(meter_df, weather_df, building_df, on_mb='building_id', 
			   on_mbw=['site_id', 'timestamp']):

	"""
	Merge the meter, weather, and building data.

	Parameters
	----------
	meter_df : pandas.core.frame.DataFrame
		Meter data
	weather_df : pandas.core.frame.DataFrame
		Weather data
	building_df : pandas.core.frame.DataFrame
		Building data
	on_mb : str or list[str], optional
		Name(s) of variable(s) to merge meter and building data on, by default 
		"building_id"
	on_mbw : str or list[str], optional
		Name(s) of variable(s) to merge resulting merged data and weather data 
		on, by default ["site_id", "timestamp"]

	Returns
	-------
	pandas.core.frame.DataFrame
		Merged data containing meter, weather, and building data
	"""

	meter = meter_df.copy()
	weather = weather_df.copy()
	building = building_df.copy()
	mb = pd.merge(meter, building, on=on_mb, how='left')
	mbw = pd.merge(mb, weather, on=on_mbw, how='left')
	return mbw
