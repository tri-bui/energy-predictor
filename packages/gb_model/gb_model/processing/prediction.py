import joblib
import numpy as np
import pandas as pd
# import xgboost as xgb


def split(df, meter_var='meter'):

	"""
	Split data by meter type.

	Parameters
	----------
	df : pandas.core.frame.DataFrame
		Full dataset including meter, weather, and building data
	meter_var : str, optional
		Name of meter type variable, by default 'meter'

	Returns
	-------
	list[pandas.core.frame.DataFrame]
		Dataframe for each meter type
	"""

	df = df.copy()
	dfs = list()
	for m in range(4):
		dfm = df[df[meter_var] == m].drop(meter_var, axis=1)
		dfs.append(dfm)
	return dfs


def transform(df, rare_enc, mean_enc, scaler):

	"""
	Transform data using:
	1. rare label categorical encoding - group rare categories into a single 
	label
	2. target-mean categorical encoding - numerically encode categories with 
	the mean target label (i.e. meter reading) of their corresponding group
	3. standard scaling - scale each feature to have a mean of 0 and standard 
	deviation of 1

	Parameters
	----------
	df : pandas.core.frame.DataFrame
		Dataset with selected variables
	rare_enc : feature_engine.encoding.rare_label.RareLabelEncoder
		Fitted rare label categorical encoder
	mean_enc : feature_engine.encoding.mean_encoding.MeanEncoder
		Fitted target-mean categorical encoder
	scaler : sklearn.preprocessing._data.StandardScaler
		Fitted standard scaler

	Returns
	-------
	pandas.core.frame.DataFrame
		Transformed data with selected features
	"""

	df = df.copy()
	transformed = rare_enc.transform(df)
	transformed = mean_enc.transform(transformed)
	transformed = scaler.transform(transformed)
	transformed = pd.DataFrame(transformed, columns=df.columns)
	return transformed


def predict(df, model=None, model_path=None, use_xgb=False):
	
	"""
	Make predictions using a trained model. Pass in either a trained model or 
	the path to a trained model. If both are passed in, `model_path` has 
	priority.

	Parameters
	----------
	df : pandas.core.frame.DataFrame
		Data with features matching the training data
	model : [type], optional
		Trained model, by default None
	model_path : [type], optional
		Path to trained model, by default None
	use_xgb : bool, optional
		Whether the model is an XGBoost model, by default False

	Returns
	-------
	numpy.ndarray[float]
		Non-negative predictions
	"""

	df = df.copy()
	# if use_xgb:
	# 	df = xgb.DMatrix(df)

	if model_path:
		model = joblib.load(model_path)
	pred = model.predict(df)
	pred[pred < 0] = 0
	return pred


def inverse_transform(df, stdize_sqft=False, sqft_var='square_feet', 
					  target_var='meter_reading'):

	"""
	Inverse transform predictions. The target variable in the training data
	was optionally standardized using the square_feet variable and then 
	log-transformed, so the inverse transformations on predictions will occur 
	in reverse order.

	Parameters
	----------
	df : pandas.core.frame.DataFrame
		Data with square footage annd target variables
	stdize_sqft : bool, optional
		Whether the target variable was standardized by the building's square 
		footage before trainning, by default False
	sqft_var : str, optional
		Name of square footage variable, by default 'square_feet'
	target_var : str, optional
		Name of target variable, by default 'meter_reading'

	Returns
	-------
	pandas.core.frame.DataFrame
		Data with inverse-transformed predictions
	"""

	df = df.copy()
	df[target_var] = np.expm1(df[target_var])
	if stdize_sqft:
		df[target_var] *= df[sqft_var] / df[sqft_var].mean()
	return df


def convert_site0_units(df, site_var='site_id', meter_var='meter',
                        target_var='meter_reading'):

	"""
	Convert site 0 meter 0 readings from kWh back to kBTU and site 0 meter 1
	readings from tons back to kBTU. Site 0 meter readings in the training data
	were recorded in kBTU, but the model was trained on units kWh and tons for
	meter 0 and meter 1 respectively. This function is only used for the Kaggle
	submission, as site 0 units were not consistent.

	Parameters
	----------
	df : pandas.core.frame.DataFrame
		Data with site, meter, and target variables
	site_var : str, optional
		Name of site variable, by default 'site_id'
	meter_var : str, optional
		Name of meter type variable, by default 'meter'
	target_var : str, optional
		Name of target variable, by default 'meter_reading'

	Returns
	-------
	pandas.core.frame.DataFrame
		Data with units of meters 0 and 1 in site 0 converted back to their 
		original units
	"""

	df = df.copy()
	df.loc[(df[site_var] == 0) & (df[meter_var] == 0), target_var] *= 3.4118
	df.loc[(df[site_var] == 0) & (df[meter_var] == 1), target_var] *= 12
	return df
