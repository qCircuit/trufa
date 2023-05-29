import cloudpickle
import datetime
import json
import logging
import mlflow
import os
import requests
from sys import version_info

import configura

logging.basicConfig(filename=configura.logs_path,
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger('urbanGUI')
logger.addHandler(logging.StreamHandler())

def get_period(start_day: str, n_days: int) -> list:
    datelst = [datetime.datetime.strptime(start_day, '%Y-%m-%d') - datetime.timedelta(days=x) for x in range(n_days)]
    datelst = [x.strftime('%Y-%m-%d') for x in datelst]
    
    return datelst

def init_enviroment(aws:bool=False):
    idx = configura.idx_traffic if any([configura.idx_campaign == 0, configura.idx_campaign == "0"]) else configura.idx_campaign
    experiment_name = f"{idx}_{list(configura.target.keys())[0]}_trainer"
    if aws:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.amazonaws.com"
        os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow.truffle.bid"
        os.environ["AWS_ACCESS_KEY_ID"] = "AKIA2YLIHQ7IPX3ELY7O"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
    
    else:
        mlflow.set_tracking_uri("sqlite:///mlruns.db")

    mlflow.set_experiment(experiment_name)

    PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)
    
    conda_env = {
        'channels': ['defaults'],
        'dependencies': [
        'python={}'.format(PYTHON_VERSION),
        'pip',
        {
            'pip': [
            'mlflow',
            'torch',
            'numpy',
            'pandas',
            'category_encoders',
            'urllib',
            'cloudpickle=={}'.format(cloudpickle.__version__)
            ],
        },
        ],
        'name': 'torch_env'
    }

    return conda_env


def collect_artifacts():
    artifacts = {
        "encoder": configura.encoder_path,
        "features": configura.features_list_path,
        "log": configura.logs_path,
        "model": configura.model_path,
        "probabilities": configura.probabilities_path,
        "scaler": configura.scaler_path,
        "test result": configura.test_plot_path
    }

    return artifacts

def sendMessage(text):
    creds = json.load(open(configura.creds_path, 'r'))
    api_key = creds['api_key']
    chat_id = creds['chat_id_group']
    try:
        response = requests.post(
            url='https://api.telegram.org/bot{0}/sendMessage'.format(api_key),
            data={'chat_id': chat_id, 'text': text},
            timeout=3
        ).json()
    except:
        pass

    return None


def sendImage(img):
    img_f = open(img, 'rb')
    creds = json.load(open(configura.creds_path, 'r'))    
    url = f'https://api.telegram.org/bot{creds["api_key"]}/sendPhoto?chat_id={creds["chat_id_group"]}'
    response = requests.post(url, files={'photo':img_f}, timeout=3)

    return None
