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


def pred_pipe(df, rare_encoder_path, mean_encoder_path, scaler_path, 
			  model_path, use_model0=True, use_xgb=True, 
			  inverse_transform_site0=False, sqft_var='square_feet', 
			  target_var='meter_reading'):

	"""
	Transform data and make predictions using a trained LightGBM or XGBoost 
	model.

	:param df: (pandas.core.frame.DataFrame) preprocessed data
	:param rare_encoder_path: (str) path to directory containing fitted rare 
							  label categorical encoders
	:param mean_encoder_path: (str) path to directory containing fitted 
							  target-mean categorical encoders
	:param scaler_path: (str) path to directory containing fitted standard 
						scalers
	:param model_path: (str) path to directory containing trained LightGBM 
					   models
	:param use_model0: (bool) whether to use model 0 (electricity) for all 
					   predictions or to use each meter's model separately
	:param use_xgb: (bool) whether to use an XGBoost model to make predictions
	:param inverse_transform_site0: (bool) whether to convert site 0 
									predictions back to original units
	:param sqft_var: (str) name of square footage variable
	:param target_var: (str) name of target variable

	:return: (list[float]) Predictions

	TODO:
	1) Update docstring to np style
	"""

	# Data
	df = df.reset_index().copy()
	tmp = df[['index', 'site_id', 'meter']].copy()
	df.drop(['index', 'site_id'], axis=1, inplace=True)
	df_list = pdn.split(df)

	# Preds
	model = 'xgb' if use_xgb else 'lgb'
	preds = list()

	# Transform data and make preds
	for i in range(4):

		# Transform data
		re = joblib.load(rare_encoder_path / f'rare_enc{i}.pkl')
		me = joblib.load(mean_encoder_path / f'mean_enc{i}.pkl')
		ss = joblib.load(scaler_path / f'scaler{i}.pkl')
		X = pdn.transform(df_list[i], re, me, ss)

		# Model path
		if use_model0:
			model_path = (model_path / f'{model}0.pkl')
		else:
			model_path = (model_path / f'{model}{i}.pkl')

		# Make preds
		y_pred = pdn.predict(X, model_path=model_path, use_xgb=use_xgb)

		# Inverse transform preds
		y = df_list[i][[sqft_var]].copy()
		y[target_var] = y_pred
		y = pdn.inverse_transform(y)

		# Store preds
		preds.append(y)

	# Format preds
	pred = pd.concat(preds).sort_index().reset_index()
	if inverse_transform_site0:
		pred = pd.merge(tmp, pred, on='index', how='left')
		pred = pdn.convert_site0_units(pred)
	pred = pred[target_var].tolist()
	return pred
