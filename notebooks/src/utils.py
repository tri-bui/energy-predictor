import joblib
import holidays
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_engine.categorical_encoders import RareLabelCategoricalEncoder, MeanCategoricalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import lightgbm as lgb
import xgboost as xgb
import optuna




####################      GENERAL      ####################


def reduce_mem_usage(df):
    
    '''
    Function:
        Recast column data types to reduce memory usage
        
    Input:
        df - Pandas dataframe
        
    Output:
        Pandas dataframe with reduced size
    '''
    
    # Unsigned integer upper limit
    uint8_lim = 2 ** 8
    uint16_lim = 2 ** 16
    uint32_lim = 2 ** 32

    # Signed integer upper limit
    int8_lim = 2 ** 7
    int16_lim = 2 ** 15
    int32_lim = 2 ** 31

    for col in df.columns:
        # Floats
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
            
        # Unsigned integers
        elif str(df[col].dtype) in 'uint64' and df[col].min() >= 0:
            if df[col].max() < uint8_lim:
                df[col] = df[col].astype('uint8')
            elif df[col].max() < uint16_lim:
                df[col] = df[col].astype('uint16')
            elif df[col].max() < uint32_lim:
                df[col] = df[col].astype('uint32')
                
        # Signed Integers
        elif str(df[col].dtype) == 'int64':
            if df[col].min() >= -int8_lim and df[col].max() < int8_lim:
                df[col] = df[col].astype('int8')
            elif df[col].min() >= -int16_lim and df[col].max() < int16_lim:
                df[col] = df[col].astype('int16')
            elif df[col].min() >= -int32_lim and df[col].max() < int32_lim:
                df[col] = df[col].astype('int32')
                
    return df


def get_summary(df):
    
    '''
    Function:
        Build a table of summary statistics
        
    Input:
        df - Pandas dataframe
        
    Output:
        Pandas dataframe similar to the Pandas.DataFrame.describe() method
    '''
    
    stats = pd.DataFrame(df.count(), columns=['_count'])
    stats['_missing'] = df.shape[0] - stats._count
    stats['_mean'] = df.mean()
    stats['_std'] = df.std()
    stats['_min'] = df.min()
    stats['_max'] = df.max()
    stats['_dtype'] = df.dtypes
    return stats.T.fillna('-')


def get_outlier_threshold(df, col_name, stat='std', multiplier=3):
    
    '''
    Function:
        Calculate the thresholds at which to consider observations an outlier
    
    Input:
        df - Pandas dataframe
        col_name - name of column to calculate threshold for
        stat (optional) - either standard deviation (std) or interquantile range (iqr)
        multiplier (optional) - multiplier for the stat
    
    Output:
        List of the lower and upper outlier thresholds
    '''
    
    # Calculate threshold based on standard deviation
    if stat == 'std':
        avg = df[col_name].mean()
        stdev = df[col_name].std()
        lower = avg - stdev * multiplier
        upper = avg + stdev * multiplier
    
    # Calculate threshold based on interquartile range
    if stat == 'iqr':
        q25 = df[col_name].quantile(0.25)
        q75 = df[col_name].quantile(0.75)
        iqr = q75 - q25
        lower = q25 - iqr * multiplier
        upper = q75 + iqr * multiplier

    return [lower, upper]




####################      CONVERSION      ####################


def convert_readings(df, site_num, meter_num, conversion,
                     site_col='site_id', meter_col='meter', reading_col='meter_reading'):
    
    '''
    Function:
        Convert the meter reading units of a specified meter type in a specified site
        
    Input:
        df - Pandas dataframe with the columns: site, meter type, and meter reading
        site_num - number of site
        meter_num - meter type number
        conversion - string to specify unit conversions in the format "{unit1}_to_{unit2}"
        site_col (optional) - name of site column
        meter_col (optional) - name of meter type column
        reading_col (optional) - name of meter reading column
        
        Note: pass in site_col, meter_col, and reading_col if different from defaults
        
    Output:
        Pandas dataframe with converted units for a given meter type in a given site
    '''
    
    # Conversion multipliers
    kbtu_to_kwh = 0.2931
    kwh_to_kbtu = 3.4118
    kbtu_to_ton = 0.0833
    ton_to_kbtu = 12
    
    # Convert using multiplier
    mult = eval(conversion)
    df.loc[(df[site_col] == site_num) & (df[meter_col] == meter_num), reading_col] *= mult
    return df


def get_rel_humidity(T, Td):
    
    '''
    Function:
        Calculate the relative humidity using air temperature and dew temperature
        
    Input:
        T - air temperature in degrees Celsius
        Td - dew temperature in degrees Celsius
        
    Output:
        Relative humidity
        
    Source:
        https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf
    '''
    
    e = 6.11 * 10.0 ** (7.5 * Td / (237.3 + Td))
    es = 6.11 * 10.0 ** (7.5 * T / (237.3 + T))
    return e * 100 / es


def to_local_time(df, timezones, site_col='site_id', time_col='timestamp'):
    
    '''
    Function:
        Convert timestamps to local time
        
    Input:
        df - Pandas dataframe with columns: site and time
        timezones - list of timezone offsets
        site_col (optional) - name of site column
        time_col (optional) - name of time column
        
        Note: pass in site_col and time_col if different from defaults
        
    Output:
        Pandas dataframe with local time
    '''
    
    offset = df[site_col].map(lambda s: np.timedelta64(timezones[s], 'h'))
    df[time_col] += offset
    return df




####################      MISSING VALUES      ####################


def missing_vals_by_site(df, pct=False, site_col='site_id', time_col='timestamp'):
    
    '''
    Function:
        Count missing values by site
    
    Input:
        df - Pandas dataframe with columns: site and time
        pct (optional) - boolean to indicate whether to convert the output to percentages
        site_col (optional) - name of site column
        time_col (optional) - name of time column
        
        Note: pass in site_col and time_col if different from defaults
    
    Output:
        Pandas dataframe displaying a matrix of missing values by site
    '''

    # # missing
    missing = df.groupby(site_col).count()
    for col in missing.columns[1:]:
        missing[col] = missing.timestamp - missing[col]
    missing.columns = [col if col == time_col else f'missing_{col}' for col in missing.columns]
    
    # % missing
    pct_missing = missing.copy()
    for col in pct_missing.columns[1:]:
        pct_missing[col] = round(100 * missing[col] / missing.timestamp, 2)
    pct_missing.columns = [col if col == time_col else f'pct_{col}' for col in pct_missing.columns]
    
    return pct_missing if pct else missing
    
    
def fill_missing(df, ffill_cols, lin_interp_cols, cub_interp_cols, site_col='site_id'):
    
    '''
    Function:
        Fill missing values by site
        
        Note: sites missing 100% of the values will not be filled
        
    Input:
        df - Pandas dataframe with site column
        ffill_cols - list of columns to perform a simple forward fill and backward fill on
        lin_interp_cols - list of columns to perform linear interpolation on
        cub_interp_cols - list of columns to perform cubic interpolation on
        site_col (optional) - name of site column
        
        Note: cols_to_interp_lin and cols_to_interp_cubic will also be forward-filled and backward-filled (after interpolation) to fill the beginning and end
        Note: pass in site_col if different from default
        
    Output:
        Pandas dataframe with missing data filled
    '''
    
    for col in df.columns:
        
        if col in ffill_cols:
            df[col] = df.groupby(site_col)[col] \
                        .transform(lambda s: s.fillna(method='ffill') \
                                              .fillna(method='bfill'))
                    
        if col in lin_interp_cols:
            df[col] = df.groupby(site_col)[col] \
                        .transform(lambda s: s.interpolate('linear', limit_direction='both') \
                                              .fillna(method='ffill') \
                                              .fillna(method='bfill'))
            
        if col in cub_interp_cols:
            df[col] = df.groupby(site_col)[col] \
                        .transform(lambda s: s.interpolate('cubic', limit_direction='both') \
                                              .fillna(method='ffill') \
                                              .fillna(method='bfill'))

    return df


def print_missing_readings(df, bldg_col='building_id', meter_col='meter', time_col='timestamp'):
    
    '''
    Function:
        Print the details of missing meter readings
        
    Input:
        df - Pandas dataframe with columns: building, meter type, and time
        bldg_col (optional) - name of building column
        meter_col (optional) - name of meter column
        time_col (optional) - name of time column
        
        Note: pass in bldg_col, meter_col, and time_col if different from defaults
        
    Output:
        None
    '''
    
    # Meter types
    types = ['electricity', 'chilledwater', 'steam', 'hotwater']
    
    # Number of readings from each meter
    n_readings = df.groupby([bldg_col, meter_col], as_index=False).count()
    n_bldgs = n_readings[bldg_col].nunique() # number of buildings
    n_meters = n_readings[meter_col].value_counts() # number of readings from each meter
    
    # Meters with missing readings
    meters_with_missing = n_readings[n_readings[time_col] != 366 * 24] # filter out full readings
    pct_meters_with_missing = 100 * meters_with_missing.shape[0] // n_readings.shape[0] # percent
    
    # Buildings with meter(s) with missing readings
    n_bldgs_with_missing = meters_with_missing[bldg_col].nunique() # number
    pct_bldgs_with_missing = 100 * n_bldgs_with_missing // n_readings[bldg_col].nunique() # percent
    
    # Total missing readings by the 4 meter types
    n_missing_by_type = meters_with_missing[meter_col].value_counts() # number
    pct_missing_by_type = 100 * n_missing_by_type // n_meters # percent
    
    # Total
    print('Buildings:', n_bldgs)
    print('Total meters:', n_meters.sum())
    for i in range(len(types)):
        print(types[i].capitalize(), 'meters:', n_meters[i])
    
    # Total missing
    print('\nBuildings with meter(s) with missing readings:',
          n_bldgs_with_missing, f'({pct_bldgs_with_missing}%)')
    print('Total meters with missing readings:',
          meters_with_missing.shape[0], f'({pct_meters_with_missing}%)')
    for i in range(len(types)):
        print(types[i].capitalize(), 'meters with missing readings:',
              n_missing_by_type[i], f'({pct_missing_by_type[i]}%)')
        
        
        
        
####################      DATAFRAME TRANSFORMATION      ####################


def reidx_site_time(df, t_start, t_end, site_col='site_id', time_col='timestamp'):
    
    '''
    Function:
        Reindex dataframe to include a timestamp for every hour within the time interval
        
    Input:
        df - Pandas dataframe with a site column and time column
        t_start - first timestamp in the format '{month}/{day}/{year} hh:mm:ss'
        t_end - last timestamp in the same format
        site_col (optional) - name of column containing sites
        time_col (optional) - name of column containing timestamps
        
        Note: pass in site_col and time_col if different from defaults
        
    Output:
        Pandas dataframe with full timestamp
    '''
    
    sites = df[site_col].unique()
    frame = df.set_index([site_col, time_col])
    
    frame = frame.reindex(
        pd.MultiIndex.from_product([
            sites,
            pd.date_range(start=t_start, end=t_end, freq='H')
        ])
    )
    
    frame.index.rename([site_col, time_col], inplace=True)
    return frame.reset_index()




def get_site(df, site_num, time_idx=False, site_col='site_id', time_col='timestamp'):
    
    '''
    Function:
        Extract the data from 1 site
        
    Input:
        df - Pandas dataframe with a site column and time column
        site_num - site number to extract
        time_idx (optional) - boolean to indicate weather or not to set the time as the index
        site_col (optional) - name of column containing sites
        time_col (optional) - name of column containing timestamps
        
        Note: pass in site_col and time_col if different from defaults
        
    Output:
        Pandas dataframe with data from 1 site
    '''
    
    df = df[df[site_col] == site_num].drop(site_col, axis=1)
    if time_idx:
        df.set_index(time_col, inplace=True)
    return df




def extract_from_dt(df, dt_components, time_col='timestamp'):
    
    '''
    Function:
        Extract datetime components from a datetime column of a dataframe and add them as new columns
        
    Input:
        df - Pandas dataframe with a time column
        dt_components - a list of Pandas datetime components to extract
        time_col (optional) - name of column containing meter readings timestamps
        
        Note: pass in time_col if different from default
        
    Output:
        Pandas dataframe with added columns
    '''
    
    for comp in dt_components:
        if comp == 'dayofyear':
            df[comp] = df[time_col].dt.dayofyear
        if comp == 'month':
            df[comp] = df[time_col].dt.month
        if comp == 'day':
            df[comp] = df[time_col].dt.day
        if comp == 'dayofweek':
            df[comp] = df[time_col].dt.dayofweek
        if comp == 'hour':
            df[comp] = df[time_col].dt.hour
    return df




def deg_to_components(df, deg_col):
    
    '''
    Function:
        Break the polar degree values of a column into x and y components
        
        Note: 0 degrees will be 0 for both components
        
    Input:
        df - Pandas dataframe with a column in polar degrees
        deg_col - name of column with values in polar degrees
        
    Output:
        Pandas dataframe with added columns
    '''
    
    df[f'{deg_col}_x'] = np.cos(df[deg_col] * np.pi / 180)
    df[f'{deg_col}_y'] = np.sin(df[deg_col] * np.pi / 180)
    df.loc[df[deg_col] == 0, f'{deg_col}_x'] = 0
    return df




####################      FEATURES      ####################




def rare_encoder(train, test, var, val=0, tol=0.05, path='../objects/transformers/rare_enc/', name='rare_enc.pkl'):
    
    '''
    Function:
        Apply feature_engine's RareLabelCategoricalEncoder to both a train and test set
    
    Input:
        train - train data
        test - test data
        var - list of features to encode (must be object type)
        tol (optional) - frequency threshold to categorize as a rare label
        val (optional) - validation data
        path (optional) - output directory path
        name (optional) - output file name
    
    Output:
        Transformed train set, transformed test set, dictionary of encoded values
    '''
    
    enc = RareLabelCategoricalEncoder(tol=tol, variables=var)
    enc.fit(train)
    joblib.dump(enc, path + name)
    train = enc.transform(train)
    test = enc.transform(test)
    if type(val) != int:
        val = enc.transform(val)
    return train, val, test, enc.encoder_dict_





def mean_encoder(X_train, y_train, X_test, var, X_val=0, path='../objects/transformers/mean_enc/', name='mean_enc.pkl'):
    
    '''
    Function:
        Apply feature_engine's MeanCategoricalEncoder to both a train and test set
    
    Input:
        X_train - train data
        y_train - train label
        X_test - test data
        var - list of features to encode (must be object type)
        X_val (optional) - validation data
        path (optional) - output directory path
        name (optional) - output file name
    
    Output:
        Transformed train set, transformed test set, dictionary of encoded values
    '''
    
    enc = MeanCategoricalEncoder(variables=var)
    enc.fit(X_train, y_train)
    joblib.dump(enc, path + name)
    X_train = enc.transform(X_train)
    X_test = enc.transform(X_test)
    if type(X_val) != int:
        X_val = enc.transform(X_val)
    return X_train, X_val, X_test, enc.encoder_dict_

    
    
    
def scale_feats(train, test, val=0, path='../objects/transformers/scaler/', name='scaler.pkl'):

    '''
    Function:
        Apply scikit-learn's StandardScaler to both a train and test set
    
    Input:
        train - train data
        test - test data
        val (optional) - validation data
        path (optional) - output directory path
        name (optional) - output file name
    
    Output:
        Transformed train set, transformed test set
    '''
    
    scaler = StandardScaler()
    scaler.fit(train)
    joblib.dump(scaler, path + name)
    train_scaled = scaler.transform(train)
    train_df = pd.DataFrame(train_scaled, columns=train.columns)
    test_scaled = scaler.transform(test)
    test_df = pd.DataFrame(test_scaled, columns=test.columns)
    val_df = 0
    if type(val) != int:
        val_scaled = scaler.transform(val)
        val_df = pd.DataFrame(val_scaled, columns=val.columns)
    return train_df, val_df, test_df
    
    
    
    
def find_rare_cats(df, col):
    
    '''
    Function:
        Plot the value counts of each category of a feature
        
    Input:
        df - Pandas dataframe with a categorical column
        col - name of categorical column
        
    Output:
        Pandas series of value counts
    '''

    counts = df[col].value_counts()
    
    counts.plot.bar()
    plt.axhline(y=df.shape[0] * 0.05, color='red')
    plt.xticks(rotation=45, ha='right')
    
    return counts




def encode_cat(encoder, df, col_to_encode):
    
    '''
    Function:
        Numerically encoded a categorical column of a Pandas dataframe
        
    Input:
        encoder - an encoder class from Sklearn with dense output
        df - Pandas dataframe with a categorical column
        col_to_encode - name of categorical column
        
    Output:
        Pandas dataframe of encoded categories
    '''
    
    encoded = encoder.fit_transform(df[[col_to_encode]])
    encoded = pd.DataFrame(encoded, columns=encoder.categories_)
    return encoded.astype('uint8')
    



def constant_feats(df):
    
    '''
    Function:
        Check a Pandas dataframe for constant and quasi-constant features
        
    Input:
        df - Pandas dataframe
        
    Output:
        Pandas dataframe showing the variance of each variable and wether or not they're a constant/quasi-constant feature
    '''
    
    const = pd.DataFrame(df.var(), columns=['variance'])
    const['constant'] = const.variance == 0
    const['quasiconstant'] = const.variance < 0.01
    return const
    
    
    
    
def duplicated_feats(df):
    
    '''
    Function:
        Check a Pandas dataframe for duplicated features
        
    Input:
        df - Pandas dataframe
        
    Output:
        A list of pairs of duplicated features (in tuples)
    '''
    
    dup = []
    for i, col1 in enumerate(df.columns[:-1]):
        for col2 in df.columns[i+1:]:
            if df[col1].equals(df[col2]):
                dup.append((col1, col2))
    return dup




def correlated_feats(df, threshold):
    
    '''
    Function:
        Check a Pandas dataframe for correlated features
        
    Input:
        df - Pandas dataframe with correlation coefficients
        
    Output:
        A list of pairs of correlated features and their correlation coefficient (in tuples)
    '''

    pairs = []
    for i, feat1 in enumerate(df.columns[:-1]):
        for feat2 in df.columns[i+1:]:
            coef = df.loc[feat1, feat2]
            if coef >= threshold:
                pairs.append((feat1, feat2, coef))
    return pairs




def feats_from_model(X, y, seln_model, ml_model):
    
    '''
    Function:
        Select features using a machine learning model
        
    Input:
        X - data
        y - target
        seln_model - name of scikit-learn model class
        ml_model - scikit-learn model class
        
    Output:
        List of selected features
    '''
    
    sel = seln_model(ml_model)
    sel.fit(X, y)
    return X.columns[sel.get_support()].tolist()




def inc_feat_count(count_df, feats, count_col='count'):
    
    '''
    Function:
        Increment the count for selected features
        
    Input:
        count_df - pandas dataframe keeping count for selected features
        feats - list of selected features
        count_col (optional) - name of count column
        
    Output:
        Pandas dataframe with updated counts
    '''
    
    for feat in feats:
        count_df.loc[feat, count_col] += 1
    return count_df




####################      MODELING      ####################




def objective(trial):
    
    '''
    Function:
        Objective function for LightGBM parameter tuning using Optuna
        
    Input:
        Optuna's trial object
        
    Output:
        Root mean squared log error of the trial
    '''

    
    



####################      PLOTTING      ####################




def plot_dist(df, cols):
    
    '''
    Function:
        Plot the value distribution of certain columns of a dataframe
        
    Input:
        df - Pandas dataframe with numeric columns
        cols - an iterable or columns to plot
        
    Output:
        None
    '''
    
    colors = 'bgrcmykw'
    for col in cols:
        fig = plt.figure(figsize=(16, 4))
        df.iloc[:, col].plot.hist(bins=16, color=colors[cols.index(col)])
        plt.xlabel(df.columns[col].replace('_', ' ').capitalize())




def plot_readings(df, buildings, freq=None, group=None, start=None, end=None, ticks=None, buildings_first=False, time_col='timestamp', building_col='building_id', meter_col='meter', reading_col='meter_reading'):
    
    '''
    Function:
        Plot readings from 1 or more of each type of meter
        
    Input:
        df - Pandas dataframe with a building column, meter type column, and time column
        buildings - an iterable of buildings
        freq (optional) - resampling frequency
        group (optional) - column or list of columns to group by
        start (optional) - the start index to slice buildings on
        end (optional) - the end index to slice buildings on
        ticks (optional) - a range of xtick locations
        buildings_first (optional) - a boolean to indicate whether or not to iterate through buildings first then meters
        time_col (optional) - name of column containing timestamps
        building_col (optional) - name of column containing buildings
        meter_col (optional) - name of column containing meter types
        reading_col (optional) - name of column containing meter readings
        
        Note: pass in freq if resampling or group if aggregating
        Note: pass in time_col, building_col, meter_col, and reading_col if different from defaults
        
    Output:
        None
    '''
    
    df = df.set_index(time_col)
    types = ['electricity', 'chilledwater', 'steam', 'hotwater']
    
    if buildings_first:
        for b in buildings:
            for m in df[df[building_col] == b][meter_col].unique():
                fig = plt.figure(figsize=(16, 4))
                bm = df[(df[building_col] == b) & (df[meter_col] == m)]
                bm.resample('d').mean()[reading_col].plot()
                plt.title(f'Building {b} ({types[m]} meter)')
                plt.ylabel('meter_reading')
    else:
        for m in range(4):
            for b in buildings[m][start:end]:
                fig = plt.figure(figsize=(16, 4))
                bm = df[(df[building_col] == b) & (df[meter_col] == m)]
                if freq:
                    bm = bm.resample(freq).mean()
                if group:
                    bm = bm.groupby(group).mean()
                bm[reading_col].plot(xticks=ticks)
                plt.title(f'Building {b} ({types[m]} meter)')
                plt.ylabel('meter_reading')

                
                
                
def show_elec_readings(df, by, idx='timestamp', vals='meter_reading', freq=None, legend_pos=(1, 1), legend_col=1, cols_to_sep=[], meter_col='meter', type_col='type'):
    
    '''
    Function:
        Plot electric meter readings by a specified feature
        
    Input:
        df - Pandas dataframe with 2 meter type columns (1 is integer-encoded)
        by - name of column to pivot to columns
        idx (optional) - name of column to set as index in pivot table
        vals (optional) - name of column to aggregate from for pivot table
        freq (optional) - resampling frequency
        legend_pos (optional) - a tuple to indicate the legend's anchor position
        legend_col (optional) - number of columns in legend
        cols_to_sep (optional) - list of columns to plot separately
        meter_col (optional) - name of column containing meter type integers
        type_col (optional) - name of column containing meter type strings
        
        Note: plot columns with a different scale separately for a better view
        Note: pass in time_col, building_col, meter_col, and reading_col if different 
        
    Output:
        Pandas dataframe of a pivot table
    '''
    
    elec = df.pivot_table(index=idx, columns=by, values=vals, aggfunc='mean')
    if freq:
        elec = elec.resample(freq).mean()
    
    elec.drop(cols_to_sep, axis=1).plot(figsize=(16, 6))
    plt.title('Electric meter readings')
    plt.ylabel('meter_reading')
    plt.legend(bbox_to_anchor=legend_pos, ncol=legend_col, fancybox=True)
    
    if cols_to_sep:
        elec[cols_to_sep].plot(figsize=(16, 6))
        plt.title('Electric meter readings')
        plt.ylabel('meter_reading')
        plt.legend(bbox_to_anchor=(1, 1), fancybox=True)
        
    return elec