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
			  model_path, use_model0=True, use_xgb=False, 
			  inverse_transform_site0=False, sqft_var='square_feet', 
			  target_var='meter_reading'):

	"""
	Transform data and make predictions using a trained LightGBM or XGBoost 
	model.

	Parameters
	----------
	df : pandas.core.frame.DataFrame
		Preprocessed data
	rare_encoder_path : str
		Path to directory containing rare label categorical encoders
	mean_encoder_path : str
		Path to directory containing target-mean categorical encoders
	scaler_path : str
		Path to directory containing standard scalers
	model_path : str
		Path to directory containing trained models
	use_model0 : bool, optional
		Whether to use model 0 (electricity) for all predictions or to use each 
		meter's model separately, by default True
	use_xgb : bool, optional
		Whether to use an XGBoost model to make predictions, by default False
	inverse_transform_site0 : bool, optional
		Whether to convert site 0 predictions back to original units, by 
		default False
	sqft_var : str, optional
		Name of square footage variable, by default "square_feet"
	target_var : str, optional
		Name of target variable, by default "meter_reading"

	Returns
	-------
	list[float]
		Meter reading predictions
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
