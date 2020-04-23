import pytest
import pathlib
import numpy as np
import pandas as pd

from gb_model.processing import pipeline
from gb_model.config import config


TEST_PATH = pathlib.Path(__file__).resolve().parent
DATA_PATH = TEST_PATH / 'datasets'

df = pd.read_pickle(DATA_PATH / 'd_prediction.pkl')
print(df.info())

"""
This dataset is the combined building, weather, and meter datasets. 
It contains the first day of each type of meter. Meter 0 is from 
building 0, meter 1 is from building 175, meter 2 is from building 
750, and meter 3 is from building 1000. 1 timestamp is missing in
buildings 750 and 1000.
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
