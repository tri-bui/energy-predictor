####################      DEPENDENCIES      ####################


import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engine.encoding import RareLabelEncoder, MeanEncoder
from sklearn.preprocessing import StandardScaler




####################      UTILITY      ####################


def reduce_mem_usage(df):
    
    """
    Recast the data types of numeric columns in a dataframe to reduce memory 
    usage
        
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Data to reduce
        
    Returns
    -------
    pandas.core.frame.DataFrame
        Data with reduced memory usage
    """
    
    # Unsigned integer upper limit
    uint8_lim = 2 ** 8
    uint16_lim = 2 ** 16
    uint32_lim = 2 ** 32

    # Signed integer upper limit
    int8_lim = 2 ** 7
    int16_lim = 2 ** 15
    int32_lim = 2 ** 31

    for col in df.columns:
        if df[col].dtype == 'float64': # floats
            df[col] = df[col].astype('float32')
            
        elif 'int' in str(df[col].dtype): # ints
            if df[col].min() >= 0: # unsigned
                if df[col].max() < uint8_lim:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < uint16_lim:
                    df[col] = df[col].astype('uint16')
                elif df[col].max() < uint32_lim:
                    df[col] = df[col].astype('uint32')
            else: # signed
                if df[col].min() >= -int8_lim and df[col].max() < int8_lim:
                    df[col] = df[col].astype('int8')
                elif df[col].min() >= -int16_lim and df[col].max() < int16_lim:
                    df[col] = df[col].astype('int16')
                elif df[col].min() >= -int32_lim and df[col].max() < int32_lim:
                    df[col] = df[col].astype('int32')
                
    return df


def summarize(df):
    
    """
    Build a custom table of summary statistics for a dataframe. The result is 
    similar to that of the dataframe's .describe() method.
        
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Data to summarize
        
    Returns
    -------
    pandas.core.frame.DataFrame
        Table of summary statistics
    """
    
    stats = pd.DataFrame(df.count(), columns=['_count'])
    stats['_missing'] = df.shape[0] - stats._count
    stats['_mean'] = df.mean()
    stats['_std'] = df.std()
    stats['_min'] = df.min()
    stats['_max'] = df.max()
    stats['_dtype'] = df.dtypes
    return stats.T.fillna('-')


def get_outlier_threshold(df, var_name, use_iqr=True, multiplier=1.5):
    
    """
    Calculate the lower and upper thresholds, for a variable in the data, at 
    which to consider observations an outlier.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Data
    var_name : str
        Name of variable to calculate the thresholds for
    use_iqr : bool, optional
        Whether to use the interquartile range in calculating the thresholds, 
        by default True. If False, the standard deviation is used.
    multiplier : numeric, optional
        Multiplier for the statistic to use in the calculation, by default 1.5
    
    Returns
    -------
    float
        Lower outlier threshold
    float
        Upper outlier threshold
    """
    
    if use_iqr: # use iqr
        q25, q75 = df[var_name].quantile(0.25), df[var_name].quantile(0.75)
        iqr = q75 - q25
        lower, upper = q25 - iqr * multiplier, q75 + iqr * multiplier

    else: # use stdev
        avg, std = df[var_name].mean(), df[var_name].std()
        lower, upper = avg - std * multiplier, avg + std * multiplier
    
    return lower, upper




####################      CONVERSION      ####################


def convert_readings(df, site, meter, convert_from, convert_to,
                     site_col='site_id', meter_col='meter', 
                     reading_col='meter_reading'):
    
    """
    Convert the meter readings of a meter type in a specified site to another 
    unit of measurement. Conversions include
    (1) kbtu to kwh
    (2) kwh to kbtu
    (3) kbtu to ton
    (4) ton to kbtu
        
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Data with the columns: site, meter type, and meter reading
    site : int
        Site number
    meter : int
        Meter type number
    convert_from : str
        Unit to convert from: "kbtu", "kwh", or "ton"
    convert_to : str
        Unit to convert to: "kbtu", "kwh", or "ton"
    site_col : str, optional
        Name of site column, by default "site_id"
    meter_col : str, optional
        Name of meter type column, by default "meter"
    reading_col : str, optional
        Name of meter reading column, by default "meter_reading"
        
    Returns
    -------
    pandas.core.frame.DataFrame
        Data with converted units for the specified meter type in the specified 
        site
    """
    
    # Conversion multipliers
    kbtu_to_kwh = 0.2931
    kwh_to_kbtu = 3.4118
    kbtu_to_ton = 0.0833
    ton_to_kbtu = 12
    
    # Convert units using multiplier
    mult = eval(convert_from + '_to_' + convert_to)
    df.loc[(df[site_col] == site) & (df[meter_col] == meter), 
           reading_col] *= mult
    return df


def polar_to_cartesian(df, deg_col, drop_original=True):
    
    """
    Break the polar degree values of a column into Cartesian x- and y-
    components. As a special case, 0 degrees will yield 0 for both components.
        
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Data with a column in polar degrees
    deg_col : str
        Name of column with values in polar degrees
    drop_original : bool, optional
        Whether to drop the original column after creating the component 
        columns, by default True
        
    Returns
    -------
    pandas.core.frame.DataFrame
        Data with additional columns: x and y
    """
    
    df[f'{deg_col}_x'] = np.cos(df[deg_col] * np.pi / 180).astype('float32')
    df[f'{deg_col}_y'] = np.sin(df[deg_col] * np.pi / 180).astype('float32')
    
    # Update x-component to 0 if y-component is 0
    df.loc[df[deg_col] == 0, f'{deg_col}_x'] = 0
    
    if drop_original:
        df.drop(deg_col, axis=1, inplace=True)
    return df


def calc_rel_humidity(T, Td):
    
    """
    Calculate the relative humidity using air temperature and dew temperature.
        
    Source: https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf
        
    Parameters
    ----------
    T : float
        Air temperature in degrees Celsius
    Td : float
        Dew temperature in degrees Celsius
        
    Returns
    -------
    float
        Relative humidity
    """
    
    e = 6.11 * 10.0 ** (7.5 * Td / (237.3 + Td))
    es = 6.11 * 10.0 ** (7.5 * T / (237.3 + T))
    return e * 100 / es


def to_local_time(df, timezones, site_col='site_id', time_col='timestamp'):
    
    """
    Convert timestamps to local time
        
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Data with columns: site and time
    timezones : list[int]
        Timezone offsets for each site
    site_col : str, optional
        Name of site column, by default "site_id"
    time_col : str, optional
        Name of time column, by default "timestamp"
        
    Returns
    -------
    pandas.core.frame.DataFrame
        Data with local time
    """
    
    offset = df[site_col].map(lambda s: np.timedelta64(timezones[s], 'h'))
    df[time_col] += offset
    return df




####################      MISSING VALUES      ####################


def missing_vals_by_site(df, pct=False, site_col='site_id', 
                         time_col='timestamp'):
    
    """
    Count missing values by site, and optionally display them as percentages.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Data with columns: site and time
    pct : bool, optional
        Whether to convert the output to percentages, by default False
    site_col : str, optional
        Name of site column, by default "site_id"
    time_col : str, optional
        Name of time column, by default "timestamp"
    
    Returns
    -------
    pandas.core.frame.DataFrame
        Table of missing value count/percentage by site
    """

    # # missing
    missing = df.groupby(site_col).count()
    for col in missing.columns[1:]:
        missing[col] = missing[time_col] - missing[col]
    missing.columns = [col if col == time_col else f'missing_{col}' 
                       for col in missing.columns]
    
    # % missing
    pct_missing = missing.copy()
    for col in pct_missing.columns[1:]:
        pct_missing[col] = round(missing[col] / missing[time_col] * 100, 2)
    pct_missing.columns = [col if col == time_col else f'pct_{col}' 
                           for col in pct_missing.columns]
    
    return pct_missing if pct else missing
    
    
def fill_missing(df, ffill_cols, lin_interp_cols, cub_interp_cols, 
                 site_col='site_id'):
    
    """
    Fill a dataframe's missing values by site. Since missing values are being 
    filled using a site's existing data, sites missing 100% of their values 
    will not be filled.
        
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Data with site column
    ffill_cols : list[str]
        Columns to perform a simple forward/backward fill on
    lin_interp_cols : list[str]
        Columns to perform linear interpolation on, followed by a 
        forward/backward fill
    cub_interp_cols : list[str]
        Columns to perform cubic interpolation on, followed by a 
        forward/backward fill
    site_col : str, optional
        Name of site column, by defauult "site_id"
        
    Returns
    -------
    pandas.core.frame.DataFrame
        Dataframe with missing data filled
    """
    
    for col in df.columns:
        if col in ffill_cols: # forward/backward fill
            df[col] = df.groupby(site_col)[col].transform(lambda s: \
                        s.fillna(method='ffill').fillna(method='bfill'))
        if col in lin_interp_cols: # linear interpolation
            df[col] = df.groupby(site_col)[col].transform(lambda s: \
                        s.interpolate('linear', limit_direction='both') \
                         .fillna(method='ffill').fillna(method='bfill'))
        if col in cub_interp_cols: # cubic interpolation
            df[col] = df.groupby(site_col)[col].transform(lambda s: \
                        s.interpolate('cubic', limit_direction='both') \
                         .fillna(method='ffill').fillna(method='bfill'))
    return df


def readings_summary(df, bldg_col='building_id', meter_col='meter', 
                             time_col='timestamp'):
    
    """
    Print the details of missing meter readings.
        
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Meter data with columns: building, meter type, and time
    bldg_col : str, optional
        Name of building column, by default "building_id"
    meter_col : str, optional
        Name of meter column, by default "meter"
    time_col : str, optional
        Name of time column, by default "timestamp"
    """
    
    # Meter types
    types = ['electricity', 'chilledwater', 'steam', 'hotwater']
    
    # Num readings from each meter
    n_readings = df.groupby([bldg_col, meter_col], as_index=False).count()
    n_bldgs = n_readings[bldg_col].nunique()
    meter_readings = n_readings[meter_col].value_counts()
    
    # Meters with missing readings
    meters_with_missing = n_readings[n_readings[time_col] != 366 * 24]
    pct_meters_with_missing = 100 * meters_with_missing.shape[0] // \
                              n_readings.shape[0] # %
    
    # Buildings w/ meter(s) with missing readings
    n_bldgs_with_missing = meters_with_missing[bldg_col].nunique() # count
    pct_bldgs_with_missing = 100 * n_bldgs_with_missing // \
                             n_readings[bldg_col].nunique() # %
    
    # Total missing readings by the 4 meter types
    n_missing_by_type = meters_with_missing[meter_col].value_counts() # count
    pct_missing_by_type = 100 * n_missing_by_type // meter_readings # %
    
    # Total
    print('Buildings:', n_bldgs)
    print('Meters:', meter_readings.sum())
    for i in range(len(types)):
        print(types[i].capitalize(), 'meters:', meter_readings[i])
    
    # Total missing
    print('\nBuildings with missing readings:', 
          n_bldgs_with_missing, f'({pct_bldgs_with_missing}%)')
    print('Meters with missing readings:',
          meters_with_missing.shape[0], f'({pct_meters_with_missing}%)')
    for i in range(len(types)):
        print(types[i].capitalize(), 'meters with missing readings:',
              n_missing_by_type[i], f'({pct_missing_by_type[i]}%)')
        
        
        
        
####################      DATAFRAME TRANSFORMATION      ####################


def extract_dt_components(df, dt_components, time_col='timestamp'):
    
    """
    Extract datetime components from a datetime column of a dataframe and add 
    them as new columns.
        
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Meter data with time column
    dt_components : list[str]
        Names of datetime components to extract
    time_col : str, optional
        Name of time column, by default "timestamp"
        
    Returns
    -------
    pandas.core.frame.DataFrame
        Data with added datetime columns
    """
    
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


def get_site(df, site_num, time_idx=False, site_col='site_id', 
             time_col='timestamp'):
    
    """
    Extract the data from 1 site, and optionally use datetime as the index.
        
    Parameters
    ----------
    df : pandas.core.frame.DataFrame 
        Data with columns: site and time
    site_num : int
        Number of site to extract data for
    time_idx : bool, optional
        Whether to set the time as the index, by default False
    site_col : str, optional
        Name of site column, by default "site_id"
    time_col : str, optional
        Name of time column, by default "timestamp"
        
    Returns
    -------
    pandas.core.frame.DataFrame 
        Data from the specified site
    """
    
    df = df[df[site_col] == site_num].drop(site_col, axis=1)
    if time_idx:
        df.set_index(time_col, inplace=True)
    return df


def reidx_site_time(df, t_start, t_end, site_col='site_id', 
                    time_col='timestamp'):
    
    """
    Fill a dataframe with timestamps for every hour within the specified time 
    interval in every site.
        
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Data with columns: site and time
    t_start : str
        First timestamp in the interval, with the format 
        '{month}/{day}/{year} hh:mm:ss'
    t_end : str
        Last timestamp in the interval, with the format 
        '{month}/{day}/{year} hh:mm:ss'
    site_col : str, optional
        Name of site column, by default "site_id"
    time_col : str, optional
        Name of time column, by default "timestamp"
        
    Returns
    -------
    pandas.core.frame.DataFrame 
        Data with complete timestamps for every site
    """
    
    sites = df[site_col].unique() # unique sites
    frame = df.set_index([site_col, time_col]) # idx site and time
    frame = frame.reindex(pd.MultiIndex.from_product(
        [sites, pd.date_range(start=t_start, end=t_end, freq='H')]
    ))
    
    # Reset idxs back to cols
    frame.index.rename([site_col, time_col], inplace=True)
    return frame.reset_index()




####################      PLOTTING      ####################


def hist_subplots(df, cols, bins=40, subplot_figsize=(15, 3), 
                  colors=sns.color_palette(n_colors=8)):
    
    """
    Plot the value distribution of specified columns of a dataframe.
        
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Data with numeric columns
    cols : list[int]
        Indices of columnns to plot
    bins : int, optional
        Number of bins to use in the histograms
    Colors - list
        Colors to use in the subplots
    """
    
    for i in cols:
        fig = plt.figure(figsize=subplot_figsize)
        df.iloc[:, i].plot.hist(bins=bins, color=colors[i-2])
        plt.xlabel(df.columns[i].replace('_', ' ').capitalize())


def plot_readings(df, bldg_list, start=0, end=1, resample=None, groupby=None, 
                  ticks=None, bldg_first=False, figsize=(15, 3), 
                  time_col='timestamp', bldg_col='building_id', 
                  meter_col='meter', read_col='meter_reading'):
    
    """
    Plot readings from 1 or more of each type of meter
        
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Data with columns: building, meter type, and time
    bldg_list : list[int]
        Buildings to plot readings for
    start : int, optional
        Start index to slice buildings on, by default 0
    end : int, optional
        End index to slice buildings on, by default 1
    resample : str, optional
        Resampling frequency if resampling by time, by default None
    groupby : list[str], optional
        Columns to group by if aggregating, by default None
    ticks : range, optional
        Range of xtick locations, by default None
    bldg_first : bool, optional
        Whether to iterate through buildings before meters when plotting, by 
        default False
    figsize :tuple(int, int), optional
        Width and height of each plot, by default (15, 3)
    time_col : str, optional
        Name of time column, by default "timestamp"
    bldg_col : str, optional
        Name of building column, by default "building_id"
    meter_col : str, optional
        Name of meter type column, by default "meter"
    read_col : str, optional
        Name of meter reading column, by default "meter_reading"
    """
    
    # Set time as index
    df = df.set_index(time_col)
    types = ['electricity', 'chilledwater', 'steam', 'hotwater']
    
    # Plot readings from all meters of each building in the list
    if bldg_first:
        for b in bldg_list: # for each building
            for m in df[df[bldg_col] == b][meter_col].unique(): # for each meter
                fig = plt.figure(figsize=figsize)
                bm = df[(df[bldg_col] == b) & (df[meter_col] == m)]
                bm.resample('d').mean()[read_col].plot() # plot daily mean
                plt.title(f'Building {b} ({types[m]} meter)')
                plt.ylabel('Meter reading')
                plt.autoscale(enable=True, axis='x', tight=True)
                
    # Plot readings from a number of each meter type
    else:
        for m in range(len(types)): # for each meter type
            for b in bldg_list[m][start:end]: # for each building
                fig = plt.figure(figsize=figsize)
                bm = df[(df[bldg_col] == b) & (df[meter_col] == m)]
                if resample: # if resampling time frequency
                    bm = bm.resample(resample).mean()
                if groupby: # if aggregating with mean
                    bm = bm.groupby(groupby).mean()
                bm[read_col].plot(xticks=ticks)
                plt.title(f'Building {b} ({types[m]} meter)')
                plt.ylabel('Meter reading')
                plt.autoscale(enable=True, axis='x', tight=True)

                
def pivot_elec_readings(df, pivot_col, pivot_idx='timestamp', pivot_vals='meter_reading', freq=None,
                        legend_pos=(1, 1), legend_col=1, cols_to_sep=[], add_to_title='', 
                        figsize=(15, 5), meter_col='meter', type_col='type'):
    
    '''
    Function:
        Pivot and plot electric meter readings by a specified feature, optionally resampling by time
        
    Input:
        df - Pandas dataframe with 2 meter type columns (1 is integer-encoded)
        pivot_col - name of column to pivot to columns
        pivot_idx (optional) - name of column to pivot to index
        pivot_vals (optional) - name of column to aggregate for pivot table
        freq (optional) - resampling frequency if resampling by time
        legend_pos (optional) - tuple to indicate the legend's anchor position
        legend_col (optional) - number of columns in legend
        cols_to_sep (optional) - list of columns to plot separately
        add_to_title (optional) - text to add to title
        figsize (optional) - tuple to indicate the size of the figure
        meter_col (optional) - name of meter type number column as integers
        type_col (optional) - name of meter type column as strings
        
        Note: plot columns with a different scale separately for a better view
        Note: pass in time_col, building_col, meter_col, and reading_col if different from defaults
        
    Output:
        Pandas dataframe of the pivot table
    '''
    
    elec = df.pivot_table(index=pivot_idx, columns=pivot_col, values=pivot_vals, aggfunc='mean')
    if freq: # resample by time
        elec = elec.resample(freq).mean()
    
    # Plot main group
    elec.drop(cols_to_sep, axis=1).plot(figsize=figsize)
    plt.title('Electric Meter Readings' + add_to_title)
    plt.ylabel('Meter reading')
    plt.legend(bbox_to_anchor=legend_pos, ncol=legend_col, fancybox=True)
    
    # Plot separated columns
    if cols_to_sep:
        elec[cols_to_sep].plot()
        plt.title('Electric Meter Readings' + add_to_title)
        plt.ylabel('Meter reading')
        plt.legend(bbox_to_anchor=(1, 1), fancybox=True)
        
    return elec




####################      FEATURES      ####################    
    
    
def plot_rare_cats(df, col, tol=0.05):
    
    '''
    Function:
        Plot the value counts of each category of a feature
        
    Input:
        df - Pandas dataframe with a categorical column
        col - name of categorical column
        tol (optional) - frequency threshold to categorize as a rare label
        
    Output:
        Pandas series of value counts
    '''

    counts = df[col].value_counts()
    
    counts.plot.bar()
    plt.axhline(y=df.shape[0] * tol, color='red')
    plt.xticks(rotation=45, ha='right')
    
    return counts


def constant_feats(df):
    
    '''
    Function:
        Check a Pandas dataframe for constant and quasi-constant features
        
    Input:
        df - Pandas dataframe
        
    Output:
        Pandas dataframe showing the variance of each variable and wether
        they're a constant/quasi-constant feature
    '''
    
    const = pd.DataFrame(df.var(), columns=['variance'])
    const['constant'] = const.variance == 0
    const['quasiconstant'] = const.variance < 0.01
    return const.T
    
    
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


def correlated_feats(corr_df, threshold):
    
    '''
    Function:
        Check a Pandas dataframe for correlated features
        
    Input:
        corr_df - Pandas dataframe with correlation coefficients
        threshold - coefficient threshold to consider high corrrelation
        
    Output:
        A list of tuples containing the correlated features and their
        correlation coefficient
    '''

    pairs = []
    for i, feat1 in enumerate(corr_df.columns[:-1]):
        for feat2 in corr_df.columns[i+1:]:
            coef = corr_df.loc[feat1, feat2]
            if abs(coef) >= threshold:
                pairs.append((feat1, feat2, coef))
    return pairs


def inc_feat_count(count_df, feats, count_col='count'):
    
    '''
    Function:
        Increment the count for selected features
        
    Input:
        count_df - Pandas dataframe keeping count of selected features
        feats - list of selected features
        count_col (optional) - name of count column
        
    Output:
        Pandas dataframe with updated counts
    '''
    
    for feat in feats:
        count_df.loc[feat, count_col] += 1
    return count_df


def feats_from_model(X, y, sel_model, ml_model):
    
    '''
    Function:
        Select features using a machine learning model
        
    Input:
        X - data
        y - target
        sel_model - name of scikit-learn feature selection transformer
        ml_model - scikit-learn estimator
        
    Output:
        List of selected features
    '''
    
    sel = sel_model(ml_model).fit(X, y)
    return X.columns[sel.get_support()].tolist()


def rare_encoder(var_list, train, test, val=None, tol=0.05, 
                 path='../models/transformers/rare_enc/', name='rare_enc', suffix=''):
    
    '''
    Function:
        Apply feature_engine's RareLabelEncoder to both a train and test set
    
    Input:
        var_list - list of features to encode (must be object type)
        train - train data
        test - test data
        tol (optional) - frequency threshold to categorize as a rare label
        val (optional) - validation data
        path (optional) - output directory path
        name (optional) - output file name
        suffix (optional) - suffix to add to file name before file extension
    
    Output:
        Transformed train set, transformed validation set, transformed test set, 
        dictionary of encoded values
    '''
    
    enc = RareLabelEncoder(tol=tol, variables=var_list).fit(train)
    joblib.dump(enc, path + name + suffix + '.pkl') # save encoder
    train = enc.transform(train)
    test = enc.transform(test)
    if val is not None:
        val = enc.transform(val)
    return train, val, test, enc.encoder_dict_


def mean_encoder(var_list, X_train, y_train, X_test, X_val=None, 
                 path='../models/transformers/mean_enc/', name='mean_enc', suffix=''):
    
    '''
    Function:
        Apply feature_engine's MeanEncoder to both a train and test set
    
    Input:
        var_list - list of features to encode (must be object type)
        X_train - train data
        y_train - train label
        X_test - test data
        X_val (optional) - validation data
        path (optional) - output directory path
        name (optional) - output file name
        suffix (optional) - suffix to add to file name before file extension
    
    Output:
        Transformed train set, transformed validation set, transformed test set, 
        dictionary of encoded values
    '''
    
    enc = MeanEncoder(variables=var_list).fit(X_train, y_train)
    joblib.dump(enc, path + name + suffix + '.pkl') # save encoder
    X_train = enc.transform(X_train)
    X_test = enc.transform(X_test)
    if X_val is not None:
        X_val = enc.transform(X_val)
    return X_train, X_val, X_test, enc.encoder_dict_

    
def scale_feats(train, test, val=None, 
                path='../models/transformers/scaler/', name='scaler', suffix=''):

    '''
    Function:
        Apply scikit-learn's StandardScaler to both a train and test set
    
    Input:
        train - train data
        test - test data
        val (optional) - validation data
        path (optional) - output directory path
        name (optional) - output file name
        suffix (optional) - suffix to add to file name before file extension
    
    Output:
        Transformed train set, transformed test set
    '''
    
    scaler = StandardScaler().fit(train)
    joblib.dump(scaler, path + name + suffix + '.pkl') # save scaler
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns)
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)
    if val is not None:
        val_scaled = pd.DataFrame(scaler.transform(val), columns=val.columns)
    return train_scaled, val_scaled, test_scaled


def encode_cat(encoder, df, col_to_encode):
    
    '''
    Function:
        Numerically encoded a categorical column of a Pandas dataframe
        
    Input:
        encoder - an encoder transformer from Sklearn with dense output
        df - Pandas dataframe with a categorical column
        col_to_encode - name of categorical column
        
    Output:
        Pandas dataframe of encoded categories
    '''
    
    encoded = encoder.fit_transform(df[[col_to_encode]])
    encoded = pd.DataFrame(encoded, columns=encoder.categories_)
    return encoded.astype('uint8')