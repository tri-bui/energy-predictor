import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


def split(df,
          meter_var='meter'):

	"""
	Split data by meter type.

	:param df: (Pandas dataframe) full data
	:param meter_var: (string) name of meter type variable

	:return: list of dataframes split by meter type
	"""

	df = df.copy()
	dfs = []
	for m in range(4):
		dfm = df[df[meter_var] == m].drop(meter_var, axis=1)
		dfs.append(dfm)
	return dfs


def transform(df, rare_enc, mean_enc, scaler):

	"""
	Transform data using rare label categorical encoding, mean
	categorical encoding, and standard scaling. Features will be
	selected on this resulting data.

	:param df: (Pandas dataframe) data with variables to be selected
	:param rare_enc: (Feature-engine categorical encoder object)
					 fitted rare label categorical encoder
	:param mean_enc: (Feature-engine categorical encoder object)
					 fitted mean categorical encoder
	:param scaler: (Scikit-learn preprocessing object)
				   fitted standard scaler

	:return: transformed dataframe with selected features
	"""

	df = df.copy()
	transformed = rare_enc.transform(df)
	transformed = mean_enc.transform(transformed)
	transformed = scaler.transform(transformed)
	transformed = pd.DataFrame(transformed, columns=df.columns)
	return transformed


def predict(df, model_path, use_xgb=True):

	"""
	Make predictions using a trained model.

	:param df: (Pandas dataframe) data with features matching
			   training data
	:param model_path: (string) path to trained model
	:param use_xgb: (boolean) whether or not to predict using a XGBoost model

	:return: non-negative predictions
	"""

	df = df.copy()
	if use_xgb:
		df = xgb.DMatrix(df)

	model = joblib.load(model_path)
	pred = model.predict(df)
	pred[pred < 0] = 0
	return pred


def inverse_transform(df, sqft_var='square_feet', target_var='meter_reading'):

	"""
	Inverse transform predictions. The target variable in the training data
	was standardized using the square_feet variable and then log-transformed.

	:param df: (Pandas dataframe) data with square footage and target variables
	:param sqft_var: (String) name of square footage variable
	:param target_var: (String) name of target variable

	:return: inverse-transformed dataframe
	"""

	df = df.copy()
	df[target_var] = np.expm1(df[target_var])
	df[target_var] *= df[sqft_var] / df[sqft_var].mean()
	return df


def convert_site0_units(df,
                        site_var='site_id',
                        meter_var='meter',
                        target_var='meter_reading'):

	"""
	Convert site 0 meter 0 readings from kWh back to kBTU and site 0 meter 1
	readings from tons back to kBTU. Site 0 meter readings in the training data
	were recorded in kBTU, but the model was trained on units kWh and tons for
	meter 0 and meter 1 respectively.

	:param df: (Pandas dataframe) data with site, meter, and target variables
	:param site_var: (String) name of site variable
	:param meter_var: (String) name of meter type variable
	:param target_var: (String) name of target variable

	:return: dataframe with units in site 0 converted
	"""

	df = df.copy()
	df.loc[(df[site_var] == 0) & (df[meter_var] == 0), target_var] *= 3.4118
	df.loc[(df[site_var] == 0) & (df[meter_var] == 1), target_var] *= 12
	return df


# def pred_pipe(df, rare_path, mean_path, sclr_path, model_path,
#                use_xgb=True,
#                sqft_var='square_feet',
#                target_var='meter_reading'):
#
# 	"""
# 	Make predictions using LightGBM or XGBoost.
#
# 	:param df: (Pandas dataframe) preprocessed data with listed variables
# 	:param rare_path: (string) path to trained rare label categorical encoders
# 	:param mean_path: (string) path to trained mean categorical encoders
# 	:param sclr_path: (string) path to trained standard scalers
# 	:param model_path: (String) path to trained LightGBM models
# 	:param use_xgb: (boolean) whether or not to predict using a XGBoost model
# 	:param sqft_var: (String) name of square footage variable
# 	:param target_var: (String) name of target variable
#
# 	:return: predictions in a list
# 	"""
#
# 	df.reset_index(inplace=True)
# 	df_list = split(df)
# 	preds = []
#
# 	for i in range(4):
# 		re = joblib.load(rare_path + str(i) + '.pkl')
# 		me = joblib.load(mean_path + str(i) + '.pkl')
# 		ss = joblib.load(sclr_path + str(i) + '.pkl')
# 		X = transform(df_list[i], re, me, ss)
#
# 		y_pred = predict(X, model_path + str(i) + '.pkl', use_xgb=use_xgb)
# 		y = df_list[i][[sqft_var]].copy()
# 		y[target_var] = y_pred
# 		y = inverse_transform(y)
# 		preds.append(y)
#
# 	pred = pd.concat(preds).sort_index().reset_index()
# 	pred = pd.merge(df[['index', 'site_id', 'meter']], pred, on='index', how='left')
# 	pred = convert_site0_units(pred)
# 	pred = pred[target_var].tolist()
# 	return pred