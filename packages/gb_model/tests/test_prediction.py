import pytest
import pathlib
import pandas as pd
from gb_model.processing import pipeline
from gb_model.config import config


# Data path
TEST_PATH = pathlib.Path(__file__).resolve().parent # gb_model/tests/
DATA_PATH = TEST_PATH / 'datasets' # gb_model/tests/datasets/

# Combined data subset for testing prediction
df = pd.read_pickle(DATA_PATH / 'd_prediction.pkl')
print(df.info())


"""
This dataset is the combined dataset containing building, weather, and meter 
data. It contains the first day of data for each type of meter. Meter 0 is from 
building 0, meter 1 is from building 175, meter 2 is from building 750, and 
meter 3 is from building 1000. 1 timestamp is missing for buildings 750 and 
1000.

Data shape: [94, 16]
"""


# @pytest.mark.skip(reason='Changed to 1 model')
def test_pred_pipe():
    pred = pipeline.pred_pipe(df,
                              config.MODEL_PATH / 'rare_enc',
                              config.MODEL_PATH / 'mean_enc',
                              config.MODEL_PATH / 'scaler',
                              config.MODEL_PATH / 'lgb',
                              use_xgb=False)
    assert len(pred) == 94
    # assert np.round(pred[48]) == 17152
