import category_encoders
import datetime
import os
import pandas as pd
import pickle
import time
from sklearn import preprocessing

import configura
import utilities

def objects_processing(v):
    '''check if float in an object feature -> cut fraction'''
    try:
        v = str(int(float(v)))
    except ValueError:
        v = str(v)

    return v.lower()

def post_process(df):
    start_time = time.time()
    print(f"preshape: {df.shape}")
    utilities.logger.info("post processing started...")

    df = df[list(configura.features.keys()) + list(configura.target.keys())]
    
    obj_cols = [k for k, v in configura.features.items() if v=="object"]
    int_cols = [k for k, v in configura.features.items() if "int" in v]
    float_cols = [k for k, v in configura.features.items() if "float" in v]
    bool_cols = [k for k, v in configura.features.items() if v=="bool"]
    bool_cols += [k for k, v in configura.target.items() if v=="bool"]

    df_trans = df.copy()
    df_trans[obj_cols] = df[obj_cols].fillna("-1")
    df_trans[int_cols+float_cols] = df[int_cols+float_cols].fillna(-1)
    df_trans[bool_cols] = df[bool_cols].fillna(0)
    
    for f in obj_cols:
        df_trans[f] = df_trans[f].apply(objects_processing)
    df_trans[int_cols+bool_cols] = df_trans[int_cols+bool_cols].astype(int)
    df_trans[float_cols] = df_trans[float_cols].astype(float)

    utilities.logger.info(
        f"post processing completed, time : {int(time.time() - start_time)}s, shape: {df.shape}"
    )
    utilities.logger.info(f"...post processing target counts: {df_trans[configura.target.keys()].value_counts()}")
    utilities.logger.info(f"posthape: {df_trans.shape}")

    return df_trans

def fit_encoder(df, encoder_type="target"):
    object_columns = [k for k,v in configura.features.items() if v=="object"]
    if encoder_type == "target":
        encoder = category_encoders.target_encoder.TargetEncoder(return_df=False)
    elif encoder_type == "catboost":
        encoder = category_encoders.cat_boost.CatBoostEncoder(return_df=False)
    elif encoder_type == "woe":
        encoder = category_encoders.woe.WOEEncoder(return_df=False)    
    encoder.fit(
        df[object_columns].copy().values,
        df[configura.target.keys()].copy().values
    )
    pickle.dump(encoder, open(configura.encoder_path, 'wb'))

    return encoder

def encode_objects(df, encoder):

    df[[k for k,v in configura.features.items() if v=="object"]] = encoder.transform(
        df[[k for k,v in configura.features.items() if v=="object"]].copy().values
    )

    return df

def fit_scaler(df, scaler_type="standard"):
    if scaler_type == "standard":
        scaler = preprocessing.StandardScaler()
    if scaler_type == "minmax":
        scaler = preprocessing.MultiLabelBinarizer()
    if scaler_type == "maxabs":
        scaler = preprocessing.MaxAbsScaler()
    scaler.fit(df.drop(configura.target.keys(), axis=1))

    pickle.dump(scaler, open(configura.scaler_path, 'wb'))

    return scaler

def scale_data(df, scaler):

    df = pd.concat([
        pd.DataFrame(
            scaler.transform(df.drop(configura.target.keys(), axis=1)),
                columns = df.drop(configura.target.keys(), axis=1).columns
            ).reset_index(drop=True),
            df[configura.target.keys()].reset_index(drop=True)], axis=1)

    return df 

def get_weekday(v):  # day of a week matching
    if type(v) == pd._libs.tslibs.timestamps.Timestamp:
        return v.weekday()
    if type(v) == str:
        return datetime.datetime.strptime(v, '%Y-%m-%d %H:%M:%S').weekday()
    else:
        # utilities.logger.error("Unknown type date")
        print("Unknown type date")


def get_hour(v):  # by-hour periods matching
    if type(v) == pd._libs.tslibs.timestamps.Timestamp:
        return v.hour
    if type(v) == str:
        return datetime.datetime.strptime(v, '%Y-%m-%d %H:%M:%S').hour
    else:
        # utilities.logger.error("Unknown type date")
        print("Unknown type date")

def process_rare(df):
    objects = [k for k,v in configura.features.items() if v == "object"]

    for f in objects:
        feature_counts = df[f].value_counts()
        not_sufficient = list(feature_counts[feature_counts < 50].keys())
        df.loc[df[f].isin(not_sufficient), f] = "other"

    return df

def process_device_os(df):
    df.device_os = df.device_os.fillna('-1')
    df.device_os = df.device_os.apply(lambda v: v.lower())
    df.loc[df.device_os.str.contains('windo'), 'device_os'] = 'windows'
    df.loc[df.device_os.str.contains('linux|ubuntu'), 'device_os'] = 'linux'
    df.loc[df.device_os.str.contains('mac|os x|osx'), 'device_os'] = 'mac'
    df.loc[df.device_os.str.contains('playstation'), 'device_os'] = 'playstation'
    df.loc[df.device_os.str.contains('ios'), 'device_os'] = 'ios'
    df.loc[df.device_os.str.contains('chrome'), 'device_os'] = 'chrome'

    return df


def pipe_processing():
    # read data
    df = pd.DataFrame(dtype=object)
    for f in os.listdir(configura.data_path):
        df = pd.concat(
            [df, pd.read_csv(configura.data_path+f)]
        ).reset_index(drop=True)

    # record the dates
    df = df.sort_values(by="created_ts").reset_index(drop=True) # sort by chrono
    configura.dates_train = df.iloc[:int(len(df)*0.8), :].created_ts.apply(
        lambda v: str(v)[:10]
    ).unique()
    configura.dates_train = f"{configura.dates_train[0]}->{configura.dates_train[-1]}"
    configura.dates_test = df.iloc[int(len(df)*0.8):, :].created_ts.apply(
        lambda v: str(v)[:10]
    ).unique()
    configura.dates_test = f"{configura.dates_test[0]}->{configura.dates_test[-1]}"

    # enrich data
    df['hour'] = df.created_ts.apply(get_hour)
    #df['weekday'] = df.created_ts.apply(get_weekday)
    #df.site_domain = df.site_domain.fillna("-1")
    #df.site_domain = df.site_domain.apply(lambda v: ''.join([x for x in v.split(".") if x != "www"]))

    # clean data
    df = df[df.is_view == 1].reset_index(drop=True) # drop specific db errors
    df = process_device_os(df)
    df = process_rare(df) # rare -> others
    df = post_process(df)

    # encode & scale
    encoder = fit_encoder(df)
    df = encode_objects(df, encoder)
    scaler = fit_scaler(df)
    df = scale_data(df, scaler)

    # split 
    dftr = df.iloc[:int(len(df)*0.8), :]
    dfts = df.iloc[int(len(df)*0.8):, :]
    utilities.logger.info(f"train shape: {dftr.shape}, test shape: {dfts.shape}")
    utilities.logger.info(f"train target counts:\n{dftr.is_click.value_counts()}")
    utilities.logger.info(f"test target counts:\n{dfts.is_click.value_counts()}")

    return dftr, dfts, encoder, scaler
