import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from gb_model.config import config as cfg
from gb_model.processing import preprocessing as ppg
from gb_model.processing import features as fts
from gb_model.processing import prediction as pdn


# Weather data preprocessing
wthr_pipe = Pipeline(
	[
		('time_converter', ppg.TimeConverter(timezones=cfg.TZ_OFFSETS)),
		('time_reindexer', ppg.TimeReindexer()),
		('missing_imputer', ppg.MissingImputer(cub_vars=cfg.CUB_VARS, 
											   lin_vars=cfg.LIN_VARS)),
		('data_copier', ppg.DataCopier())
	]
)


# Feature engineering
feat_pipe = Pipeline(
	[
		('weather_extractor', fts.WeatherExtractor()),
		('time_extractor', fts.TimeExtractor()),
		('holiday_extractor', fts.HolidayExtractor(countries=cfg.COUNTRIES)),
		('feat_selector', fts.FeatSelector(feats=cfg.FEATS))
	]
)


# Prediction
def pred_pipe(df, rare_encoder_path, mean_encoder_path, scaler_path, 
			  model_path, use_xgb=True, sqft_var='square_feet', 
			  target_var='meter_reading'):

	"""
	Make predictions using LightGBM or XGBoost.

	:param df: (Pandas dataframe) preprocessed data with listed variables
	:param rare_encoder_path: (pathlib Path object) path to trained rare label 
							  categorical encoders
	:param mean_encoder_path: (pathlib Path object) path to trained mean 
							  categorical encoders
	:param scaler_path: (pathlib Path object) path to trained standard scalers
	:param model_path: (pathlib Path object) path to trained LightGBM models
	:param use_xgb: (boolean) whether or not to predict using a XGBoost model
	:param sqft_var: (String) name of square footage variable
	:param target_var: (String) name of target variable

	:return: predictions in a list
	"""

	df = df.reset_index().copy()
	# tmp = df[['index', 'site_id', 'meter']].copy()
	df.drop(['index', 'site_id'], axis=1, inplace=True)
	df_list = pdn.split(df)

	model = joblib.load(model_path / 'lgb0.pkl')
	preds = list()

	for i in range(4):
		re = joblib.load(rare_encoder_path / f'rare_enc{i}.pkl')
		me = joblib.load(mean_encoder_path / f'mean_enc{i}.pkl')
		ss = joblib.load(scaler_path / f'scaler{i}.pkl')
		X = pdn.transform(df_list[i], re, me, ss)

		y_pred = pdn.predict(X, model=model, use_xgb=use_xgb)
		# y_pred = pdn.predict(X, model_path=(model_path / f'lgb{i}.pkl'), 
		# 					   use_xgb=use_xgb)
		y = df_list[i][[sqft_var]].copy()
		y[target_var] = y_pred
		y = pdn.inverse_transform(y)
		preds.append(y)

	pred = pd.concat(preds).sort_index().reset_index()
	# pred = pd.merge(tmp, pred, on='index', how='left')
	# pred = pdn.convert_site0_units(pred)
	
	pred = pred[target_var].tolist()
	return pred
