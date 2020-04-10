from pathlib import Path

import gb_model


# Directory paths
ROOT_PATH = Path(gb_model.__file__).resolve().parent
DATA_PATH = ROOT_PATH / 'datasets'
MODEL_PATH = ROOT_PATH / 'models'
LOG_PATH = ROOT_PATH / 'logs'
LOG_PATH.mkdir(exist_ok=True)

# Data columns
METER_COLS = ['building_id', 'meter', 'timestamp']
WEATHER_COLS = ['site_id', 'timestamp', 'air_temperature', 'dew_temperature',
                'sea_level_pressure', 'wind_speed', 'wind_direction']

# Selected features
FEATS = ['building_id', 'primary_use', 'square_feet', 'year_built', 'country',
         'dayofyear', 'hour', 'is_weekend', 'is_holiday',
         'rel_humidity', 'dew_temperature', 'sea_level_pressure',
         'wind_speed', 'wind_direction_y', 'site_id', 'meter']

# Site to country mapping
COUNTRIES = {0: 'US', 1: 'UK', 2: 'US', 3: 'US',
             4: 'US', 5: 'UK', 6: 'US', 7: 'CA',
             8: 'US', 9: 'US', 10: 'US', 11: 'CA',
             12: 'IE', 13: 'US', 14: 'US', 15: 'US'}

# Timezone conversion offsets for sites 0 - 15
TZ_OFFSETS = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]

# Variables to be imputed using linear interpolation and cubic interpolation
LIN_VARS = ['wind_direction', 'wind_speed']
CUB_VARS = ['air_temperature', 'dew_temperature', 'sea_level_pressure']
