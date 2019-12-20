#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:09:36 2019

@author: mulderg
"""

from logging import basicConfig, getLogger
#from logging import DEBUG as log_level
from logging import INFO as log_level
basicConfig(level = log_level,
            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)

import numpy as np
import pandas as pd
from pprint import pformat
from datetime import date
from math import sqrt
#from statistics import mean

from hyperopt import fmin, tpe, hp, space_eval, STATUS_FAIL, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from os import environ


########################################################################################################
        
def detrend(insample_data):
    """
    Calculates a & b parameters of LRL
    :param insample_data:
    :return:
    """
    x = np.arange(len(insample_data))
    a, b = np.polyfit(x, insample_data, 1)
    return a, b

def acf(data, k):
    """
    Autocorrelation function
    :param data: time series
    :param k: lag
    :return:
    """
    m = np.mean(data)
    s1 = 0
    for i in range(k, len(data)):
        s1 = s1 + ((data[i] - m) * (data[i - k] - m))

    s2 = 0
    for i in range(0, len(data)):
        s2 = s2 + ((data[i] - m) ** 2)

    return float(s1 / s2)

def seasonality_test(original_ts, ppy, tcrit):
    """
    Seasonality test
    :param original_ts: time series
    :param ppy: periods per year
    :param tcrit: seasonality critical cutoff
    :return: boolean value: whether the TS is seasonal
    """
    s = acf(original_ts, 1)
    for i in range(2, ppy):
        s = s + (acf(original_ts, i) ** 2)

    limit = tcrit * (sqrt((1 + 2 * s) / len(original_ts)))

    return abs(acf(original_ts, ppy)) > limit

def deseasonalize(original_ts, ppy, tcrit):
    """
    Calculates and returns seasonal indices
    :param original_ts: original data
    :param ppy: periods per year
    :param tcrit: seasonality critical cutoff
    :return:
    """
    """
    # === get in-sample data
    original_ts = original_ts[:-out_of_sample]
    """
    if seasonality_test(original_ts, ppy, tcrit):
        logger.debug("seasonal")
        # ==== get moving averages
        ma_ts = moving_averages(original_ts, ppy)

        # ==== get seasonality indices
        le_ts = original_ts * 100.0 / ma_ts
        le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
        le_ts = np.reshape(le_ts, (-1, ppy))
        si = np.nanmean(le_ts, 0)
        norm = np.sum(si) / (ppy * 100.0)
        si = si / norm
    else:
        logger.debug("NOT seasonal")
        si = np.full(ppy, 100.0)

    return si

def moving_averages(ts_init, window):
    """
    Calculates the moving averages for a given TS
    :param ts_init: the original time series
    :param window: window length
    :return: moving averages ts
    """
    """
    As noted by Professor Isidro Lloret Galiana:
    line 82:
    if len(ts_init) % 2 == 0:
    
    should be changed to
    if window % 2 == 0:
    
    This change has a minor (less then 0.05%) impact on the calculations of the seasonal indices
    In order for the results to be fully replicable this change is not incorporated into the code below
    """
    
#    if len(ts_init) % 2 == 0:
#        ts_ma = pd.rolling_mean(ts_init, window, center=True)
#        ts_ma = pd.rolling_mean(ts_ma, 2, center=True)
#        ts_ma = np.roll(ts_ma, -1)
#    else:
#        ts_ma = pd.rolling_mean(ts_init, window, center=True)

    if window % 2 == 0:
        ts_ma = pd.Series(ts_init).rolling(window, center=True).mean()
        ts_ma = ts_ma.rolling(2, center=True).mean()
        ts_ma = np.roll(ts_ma.values, -1)
    else:
        ts_ma = pd.Series(ts_init).rolling(window, center=True).values
        
    return ts_ma

def smape(a, b):
    """
    Calculates sMAPE
    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()

def mase(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE
    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = np.mean(abs(insample[freq:] - y_hat_naive))

    return np.mean(abs(y_test - y_hat_test)) / masep

#######################################################################################

rand_seed = 42

if "VERSION" in environ:    
    version = environ.get("VERSION")
    logger.info("Using version : %s" % version)
    
    use_cluster = True
else:
    version = "final"
    logger.warning("VERSION not set, using: %s" % version)
    
    use_cluster = False

if "DATASET" in environ:    
    dataset_name = environ.get("DATASET")
    logger.info("Using dataset : %s" % dataset_name)
    
    use_cluster = True
else:
    dataset_name = "final"
    logger.warning("DATASET not set, using: %s" % dataset_name)
    
freq_pd = "M"
freq = 12
prediction_length = 1

#def goodness_of_fit(ts_train, ts_train_fit):
#    return np.mean((ts_train-ts_train_fit)**2)*100/(np.mean(ts_train)^2)

def score_model(model, model_type, data, season_coeffs):
    import mxnet as mx
    from gluonts.dataset.common import ListDataset
    from gluonts.evaluation.backtest import make_evaluation_predictions #, backtest_metrics
    from gluonts.model.predictor import Predictor
    from tempfile import mkdtemp
    from pathlib import Path

#    gluon_train = ListDataset(data['train'].copy(), freq=freq_pd)
    gluon_test = ListDataset(data['test'].copy(), freq=freq_pd)    
    
#    metrics = backtest_metrics(gluon_train, gluon_test, forecaster=model, num_samples=1, logging_file=None)
#    logger.info("\n(%s, %s)" % metrics)
    
    if model_type != "DeepStateEstimator":
        forecast_it, ts_it = make_evaluation_predictions(dataset=gluon_test, predictor=model, num_samples=1)
    else:
        temp_dir_path = mkdtemp()
        model.serialize(Path(temp_dir_path))
        model_cpu = Predictor.deserialize(Path(temp_dir_path), ctx=mx.cpu())
        logger.info("Loaded DeepState model")
        forecast_it, ts_it = make_evaluation_predictions(dataset=gluon_test, predictor=model_cpu, num_samples=1)
        logger.info("Evaluated DeepState model")
        
    forecasts = list(forecast_it)
    
    # Add back seasonality and compute error metrics
    (model_fits, mases, smapes) = ([], [], [])
    for j in range(len(forecasts)):
        ts_train = data['train'][j]['target']
        for i in range(0, len(ts_train)):
            ts_train[i] = ts_train[i] * season_coeffs[j][i % freq] / 100

        ts_test = data['test'][j]['target']
        for i in range(0, len(ts_test)):
            ts_test[i] = ts_test[i] * season_coeffs[j][i % freq] / 100
            
        # Get forecast and set any negative forecasts to 0.0
        y_hat_test = forecasts[j].samples.reshape(-1)
        for i in range(0, len(y_hat_test)):
            if y_hat_test[i] < 0.0:
                logger.debug("Negative forecast")
                y_hat_test[i] == 0.0
                
        for i in range(len(ts_train), len(ts_train) + prediction_length):
            y_hat_test[i - len(ts_train)] = y_hat_test[i - len(ts_train)] * season_coeffs[j][i % freq] / 100

#        model_fits.append(goodness_of_fit(ts_train, y_hat_test[:-prediction_length]))
        mases.append(mase(np.array(ts_test[:-prediction_length]), np.array(ts_test[-prediction_length:]), y_hat_test, freq))
        smapes.append(smape(np.array(ts_test[-prediction_length:]), y_hat_test))

    return {
#        'gof'     : np.mean(model_fits),
        'mase'    : np.mean(mases),
        'smape'   : float(100 * float(np.mean(smapes)))
    }

def load_plos_m3_data(path, tcrit, model_type):
    from json import loads
    
    data = {}
    for dataset in ["train", "test"]:
        data[dataset] = []
        fname = "%s/%s/data.json" % (path, dataset)
        logger.info("Reading data from: %s" % fname)
        with open(fname) as fp:
            for line in fp:
               ts_data = loads(line)               
               data[dataset].append(ts_data)
    logger.info("Loaded %d time series" % len(data["train"]))
    season_coeffs = []
    for j in range(len(data["train"])):        
        ts_train = data["train"][j]["target"]
        ts_test  = data["test"][j]["target"]
        
        # Remove static features if not supported by model
        if model_type in ['SimpleFeedForwardEstimator',
                          'DeepFactorEstimator',
                          'GaussianProcessEstimator',
                          'WaveNetEstimator']:
            del(data["train"][j]['feat_static_cat'])
            del(data["test"][j]['feat_static_cat'])
            
        # Determine seasonality coeffs
        if tcrit > 0.0:
            seasonality_in = deseasonalize(np.array(ts_train), freq, tcrit)
        else:
            seasonality_in = np.full(freq, 100)
        season_coeffs.append(seasonality_in)
        
        #  Deaseasonalise training data
        for i in range(0, len(ts_train)):
            ts_train[i] = ts_train[i] * 100 / seasonality_in[i % freq]
            
        #  Deaseasonalise test data
        for i in range(0, len(ts_test)):
            ts_test[i] = ts_test[i] * 100 / seasonality_in[i % freq]
        
        data["train"][j]["target"] = ts_train
        data["test"][j]["target"]  = ts_test
        
    return data, season_coeffs
    
def forecast(cfg):    
    import mxnet as mx
    from gluonts.dataset.common import ListDataset
    from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
    from gluonts.model.deep_factor import DeepFactorEstimator
    from gluonts.model.gp_forecaster import GaussianProcessEstimator
#    from gluonts.kernels import RBFKernelOutput, KernelOutputDict
    from gluonts.model.wavenet import WaveNetEstimator
    from gluonts.model.transformer import TransformerEstimator
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.model.deepstate import DeepStateEstimator
    from gluonts.trainer import Trainer
    from gluonts import distribution
    
    logger.info("Params: %s " % cfg)
    mx.random.seed(rand_seed, ctx='all')
    np.random.seed(rand_seed)

    # Load training data
    train_data, train_season_coeffs  = load_plos_m3_data("/var/tmp/m4_monthly", cfg['tcrit'], cfg['model']['type'])
    gluon_train = ListDataset(train_data['train'].copy(), freq=freq_pd)
    
#    trainer=Trainer(
#        epochs=3,
#        hybridize=False,
#    )

    trainer=Trainer(
        mx.Context("gpu"),
        hybridize=False,
        epochs=cfg['trainer']['max_epochs'],
        num_batches_per_epoch=cfg['trainer']['num_batches_per_epoch'],
        batch_size=cfg['trainer']['batch_size'],
        patience=cfg['trainer']['patience'],
        learning_rate=cfg['trainer']['learning_rate'],
        learning_rate_decay_factor=cfg['trainer']['learning_rate_decay_factor'],
        minimum_learning_rate=cfg['trainer']['minimum_learning_rate'],
        weight_decay=cfg['trainer']['weight_decay'],
    )

    if cfg['box_cox']:
        distr_output=distribution.TransformedDistributionOutput(distribution.GaussianOutput(),
                                                                    [distribution.InverseBoxCoxTransformOutput(lb_obs=-1.0E-5)])
    else:
        distr_output=distribution.StudentTOutput()
        
    if cfg['model']['type'] == 'SimpleFeedForwardEstimator':
        estimator = SimpleFeedForwardEstimator(
            freq=freq_pd,
            prediction_length=prediction_length,
            num_hidden_dimensions = cfg['model']['num_hidden_dimensions'],
            num_parallel_samples=1,
            trainer=trainer,
            distr_output=distr_output)
        
    if cfg['model']['type'] == 'DeepFactorEstimator': 
         estimator = DeepFactorEstimator(
            freq=freq_pd,
            prediction_length=prediction_length,
            num_hidden_global=cfg['model']['num_hidden_global'], 
            num_layers_global=cfg['model']['num_layers_global'], 
            num_factors=cfg['model']['num_factors'], 
            num_hidden_local=cfg['model']['num_hidden_local'], 
            num_layers_local=cfg['model']['num_layers_local'],
            trainer=trainer,
            distr_output=distr_output)

    if cfg['model']['type'] == 'GaussianProcessEstimator':
#        if cfg['model']['rbf_kernel_output']:
#            kernel_output = RBFKernelOutput()
#        else:
#            kernel_output = KernelOutputDict()
        estimator = GaussianProcessEstimator(
            freq=freq_pd,
            prediction_length=prediction_length,
            cardinality=len(train_data['train']),
            max_iter_jitter=cfg['model']['max_iter_jitter'],
            sample_noise=cfg['model']['sample_noise'],
            num_parallel_samples=1,
            trainer=trainer)

    if cfg['model']['type'] == 'WaveNetEstimator':            
        estimator = WaveNetEstimator(
            freq=freq_pd,
            prediction_length=prediction_length,
            cardinality=[len(train_data['train']), 6],
            embedding_dimension=cfg['model']['embedding_dimension'],
            num_bins=cfg['model']['num_bins'],        
            n_residue=cfg['model']['n_residue'],
            n_skip=cfg['model']['n_skip'],
            dilation_depth=cfg['model']['dilation_depth'], 
            n_stacks=cfg['model']['n_stacks'],
            act_type=cfg['model']['wn_act_type'],
            num_parallel_samples=1,
            trainer=trainer)
                 
    if cfg['model']['type'] == 'TransformerEstimator':
        if cfg['model']['tf_use_xreg']:
            cardinality=[len(train_data['train']), 6]
        else:
            cardinality=None
            
        estimator = TransformerEstimator(
            freq=freq_pd,
            prediction_length=prediction_length,
            use_feat_static_cat=cfg['model']['tf_use_xreg'],
            cardinality=cardinality,
            model_dim=cfg['model']['model_dim_heads'][0], 
            inner_ff_dim_scale=cfg['model']['inner_ff_dim_scale'],
            pre_seq=cfg['model']['pre_seq'], 
            post_seq=cfg['model']['post_seq'], 
            act_type=cfg['model']['tf_act_type'], 
            num_heads=cfg['model']['model_dim_heads'][1], 
            dropout_rate=cfg['model']['trans_dropout_rate'],
            num_parallel_samples=1,
            trainer=trainer,
            distr_output=distr_output)

    if cfg['model']['type'] == 'DeepAREstimator':
        if cfg['model']['da_use_xreg']:
            cardinality=[len(train_data['train']), 6]
        else:
            cardinality=None
            
        estimator = DeepAREstimator(
            freq=freq_pd,
            prediction_length=prediction_length,        
            use_feat_static_cat=cfg['model']['da_use_xreg'],
            cardinality=cardinality,
            cell_type=cfg['model']['da_cell_type'],
            num_cells=cfg['model']['da_num_cells'],
            num_layers=cfg['model']['da_num_layers'],        
            dropout_rate=cfg['model']['da_dropout_rate'],
            num_parallel_samples=1,
            trainer=trainer,
            distr_output=distr_output)

    if cfg['model']['type'] == 'DeepStateEstimator':            
        estimator = DeepStateEstimator(
            freq=freq_pd,
            prediction_length=prediction_length,
#            cell_type=cfg['model']['ds_cell_type'],
#            add_trend=cfg['model']['add_trend'],     
#            num_cells=cfg['model']['ds_num_cells'],
#            num_layers=cfg['model']['ds_num_layers'],    
#            num_periods_to_train=cfg['model']['num_periods_to_train'],    
#            dropout_rate=cfg['model']['ds_dropout_rate'],
            use_feat_static_cat=True,
            cardinality=[len(train_data['train']), 6],
            num_parallel_samples=1,
            trainer=trainer)
                    
    logger.info("Fitting: %s" % cfg['model']['type'])
    logger.info(estimator)
    model = estimator.train(gluon_train)
    
    train_errs = score_model(model, cfg['model']['type'], train_data, train_season_coeffs)
    logger.info("Training error: %s" % train_errs)

    test_data, test_season_coeffs = load_plos_m3_data("/var/tmp/m4_monthly", cfg['tcrit'], cfg['model']['type'])
    test_errs = score_model(model,  cfg['model']['type'], test_data,  test_season_coeffs)
    logger.info("Testing error: %s" % test_errs)
    
    return {
        'train' : train_errs,
        'test'  : test_errs,
    }

def gluonts_fcast(cfg):   
    from traceback import format_exc
    from os import environ as local_environ
    
    try:
        err_metrics = forecast(cfg)
        if np.isnan(err_metrics['train']['mase']):
            raise ValueError("Training MASE is NaN")
        if np.isinf(err_metrics['train']['mase']):
           raise ValueError("Training MASE is infinite")
           
    except Exception as e:                    
        exc_str = format_exc()
        logger.error('\n%s' % exc_str)
        return {
            'loss'        : None,
            'status'      : STATUS_FAIL,
            'cfg'         : cfg,
            'exception'   : exc_str,
            'build_url'   : local_environ.get("BUILD_URL")
        }
    logger.info(err_metrics)
#    logger.info("Error metrics:" % pformat(err_metrics, indent=4, width=160))
    return {
        'loss'        : err_metrics['train']['mase'],
        'status'      : STATUS_OK,
        'cfg'         : cfg,
        'err_metrics' : err_metrics,
        'build_url'   : local_environ.get("BUILD_URL")
    }

def call_hyperopt():
    dropout_rate = {
        'min' : 0.07,
        'max' : 0.13
    }

    space = {
        # Preprocessing
        'tcrit'   : hp.choice('tcrit', [-1.0, 1.645-0.2, 1.645, 1.645+0.2]), # < 0.0 == no deseasonalisation
        'box_cox' : hp.choice('box_cox', [True, False]),
        
        'trainer' : {
            'max_epochs'                 : hp.choice('max_epochs', [32, 64, 128, 256, 512, 1024, 2048]),
            'num_batches_per_epoch'      : hp.choice('num_batches_per_epoch', [32, 64, 128, 256, 512, 1024, 2048]),
            'batch_size'                 : hp.choice('batch_size', [32, 64, 128, 256, 512]),
            'patience'                   : hp.choice('patience', [8, 16, 32, 64]),
            
            'learning_rate'              : hp.loguniform('learning_rate', np.log(05e-04), np.log(50e-04)),
            'learning_rate_decay_factor' : hp.uniform('learning_rate_decay_factor', 0.10, 0.75),
            'minimum_learning_rate'      : hp.loguniform('minimum_learning_rate', np.log(005e-06), np.log(100e-06)),
            'weight_decay'               : hp.loguniform('weight_decay', np.log(01e-09), np.log(100e-09)),
        },

        'model' : hp.choice('model', [
            {
                'type'                       : 'SimpleFeedForwardEstimator',
                'num_hidden_dimensions'      : hp.choice('num_hidden_dimensions', [[2], [4], [8], [16], [32], [64], [128],
                                                                                   [2, 2], [4, 2], [8, 8], [8, 4], [16, 16], [16, 8], [32, 16], [64, 32],
                                                                                   [64, 32, 16], [128, 64, 32]]),
            },

            {
                'type'                       : 'DeepFactorEstimator',
                'num_hidden_global'          : hp.choice('num_hidden_global', [2, 4, 8, 16, 32, 64, 128, 256]),
                'num_layers_global'          : hp.choice('num_layers_global', [1, 2, 3]),
                'num_factors'                : hp.choice('num_factors', [2, 4, 8, 16, 32]),
                'num_hidden_local'           : hp.choice('num_hidden_local', [2, 4, 8]),
                'num_layers_local'           : hp.choice('num_layers_local', [1, 2, 3]),
            },
                    
            {
                'type'                       : 'GaussianProcessEstimator',
#                'rbf_kernel_output'          : hp.choice('rbf_kernel_output', [True, False]),
                'max_iter_jitter'            : hp.choice('max_iter_jitter', [4, 8, 16, 32]),
                'sample_noise'               : hp.choice('sample_noise', [True, False]),
            },
                    
            {
                'type'                       : 'WaveNetEstimator',
                'embedding_dimension'        : hp.choice('embedding_dimension', [2, 4, 8, 16, 32, 64]),
                'num_bins'                   : hp.choice('num_bins', [256, 512, 1024, 2048]),
                'n_residue'                  : hp.choice('n_residue', [22, 23, 24, 25, 26]),
                'n_skip'                     : hp.choice('n_skip', [4, 8, 16, 32, 64, 128]),
                'dilation_depth'             : hp.choice('dilation_depth', [None, 1, 2, 3, 4, 5, 7, 9]),
                'n_stacks'                   : hp.choice('n_stacks', [1, 2, 3]),
                'wn_act_type'                : hp.choice('wn_act_type', ['elu', 'relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']),
            },
                   
            {
                'type'                       : 'TransformerEstimator',
                'tf_use_xreg'                : hp.choice('tf_use_xreg', [True, False]),
                'model_dim_heads'            : hp.choice('model_dim_heads', [[2, 2], [4, 2], [8, 2], [16, 2], [32, 2], [64, 2],
                                                                             [4, 4], [8, 4], [16, 4], [32, 4], [64, 4],
                                                                             [8, 8], [16, 8], [32, 8], [64, 8],
                                                                             [16, 16], [32, 16], [64, 16]]),
                'inner_ff_dim_scale'         : hp.choice('inner_ff_dim_scale', [2, 3, 4, 5]),
                'pre_seq'                    : hp.choice('pre_seq', ['d', 'n', 'dn', 'nd']),
                'post_seq'                   : hp.choice('post_seq', ['d', 'r', 'n', 'dn', 'nd', 'rn', 'nr', 'dr', 'rd', 'drn', 'dnr', 'rdn', 'rnd', 'nrd', 'ndr']),
                'tf_act_type'                : hp.choice('tf_act_type', ['relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']),               
                'trans_dropout_rate'         : hp.uniform('trans_dropout_rate', dropout_rate['min'], dropout_rate['max']),
            },

            {
                'type'                       : 'DeepAREstimator',
                 'da_cell_type'              : hp.choice('da_cell_type', ['lstm', 'gru']),
                'da_use_xreg'                : hp.choice('da_use_xreg', [True, False]),
                'da_num_cells'               : hp.choice('da_num_cells', [2, 4, 8, 16, 32, 64, 128, 256, 512]),
                'da_num_layers'              : hp.choice('da_num_layers', [1, 2, 3, 4, 5, 7, 9]),

                
                'da_dropout_rate'            : hp.uniform('da_dropout_rate', dropout_rate['min'], dropout_rate['max']),
            },

#            {
#                'type'                       : 'DeepStateEstimator',
#                'ds_cell_type'               : hp.choice('ds_cell_type', ['lstm', 'gru']),
#                'add_trend'                  : hp.choice('add_trend', [True, False]),
#                'ds_num_cells'               : hp.choice('ds_num_cells', [2, 4, 8, 16, 32, 64, 128, 256, 512]),
#                'ds_num_layers'              : hp.choice('ds_num_layers', [1, 2, 3, 4, 5, 7, 9]),
#                'num_periods_to_train'       : hp.choice('num_periods_to_train', [2, 3, 4, 5, 6]),   
#                'ds_dropout_rate'            : hp.uniform('ds_dropout_rate', dropout_rate['min'], dropout_rate['max']),
#            },
        ])
    }

## 18 month
#    space = {
#        'tcrit' : hp.choice('tcrit', [-1.0]), # < 0.0 == no deseasonalisation
#        
#        'trainer' : {
#            'max_epochs'                 : hp.choice('max_epochs', [800, 900, 1000, 1100, 1200]),
#            'num_batches_per_epoch'      : hp.choice('num_batches_per_epoch', [60, 70, 80, 90, 100]),
#            'batch_size'                 : hp.choice('batch_size', [160, 180, 200, 220, 240]),
#            'patience'                   : hp.choice('patience', [60, 70, 80, 90, 100]),
#            
#            'learning_rate'              : hp.uniform('learning_rate', 4.6e-04, 14.6e-04),
#            'learning_ratf_decay_factor' : hp.uniform('learning_ratf_decay_factor', 0.60, 0.68),
#            'minimum_learning_rate'      : hp.uniform('minimum_learning_rate', 0.86e-06, 2.86e-06),
#            'weight_decay'               : hp.uniform('weight_decay', 4.2e-08, 12.2e-08),
#        },
#
#        'model' : hp.choice('model', [                    
#            {
#                'type'                       : 'DeepAREstimator',
#                'da_num_cells'                  : hp.choice('da_num_cells', [512-128, 512-64, 512, 512+64, 512+128]),
#                'da_num_layers'                 : hp.choice('da_num_layers', [2, 3, 4, 5, 6]),
#
#                
#                'da_dropout_rate'           : hp.uniform('da_dropout_rate', 0.094, 0.134),
#            },
#        ])
#    }
    
## 1 month
#    space = {
#        'tcrit' : hp.choice('tcrit', [-1.0]), # < 0.0 == no deseasonalisation
#        'box_cox' : hp.choice('box_cox', [True, False]),
#        'trainer' : {
#            'max_epochs'                 : hp.choice('max_epochs', [128-32, 128-16, 128, 128+16, 128+32]),
#            'num_batches_per_epoch'      : hp.choice('num_batches_per_epoch', [1024-256, 1024-128, 1024, 1024+128, 1024+256]),
#            'batch_size'                 : hp.choice('batch_size', [256-64, 256-32, 256, 256+32, 256+64]),
#            'patience'                   : hp.choice('patience', [32-8, 32-4, 32, 32+4, 32+8]),
#            
#            'learning_rate'              : hp.uniform('learning_rate', 0.000287, 0.001287),
#            'learning_ratf_decay_factor' : hp.uniform('learning_ratf_decay_factor', 0.227, 0.327),
#            'minimum_learning_rate'      : hp.uniform('minimum_learning_rate', 1.01e-05, 2.01e-05),
#            'weight_decay'               : hp.uniform('weight_decay', 4.9e-09, 14.9e-09),
#        },
#
#        'model' : hp.choice('model', [                    
#            {
#                'type'                       : 'DeepAREstimator',
#                'da_num_cells'                  : hp.choice('da_num_cells', [256-64, 256-32, 256, 256+32, 256+64]),
#                'da_num_layers'                 : hp.choice('da_num_layers', [8, 9, 10, 11, 12]),
#
#                
#                'da_dropout_rate'           : hp.uniform('da_dropout_rate', 0.0898, 0.0998),
#            },
#        ])
#    }
                            
    if use_cluster:
        exp_key = "%s" % str(date.today())
        logger.info("exp_key for this job is: %s" % exp_key)
        trials = MongoTrials('mongo://heika:27017/%s-%s/jobs' % (dataset_name, version), exp_key=exp_key)
        best = fmin(gluonts_fcast, space, rstate=np.random.RandomState(rand_seed), algo=tpe.suggest, show_progressbar=False, trials=trials, max_evals=1000)
    else:
        best = fmin(gluonts_fcast, space, algo=tpe.suggest, show_progressbar=False, max_evals=20)
         
    return space_eval(space, best) 
    
if __name__ == "__main__":
    params = call_hyperopt()
    logger.info("Best params:\n%s" % pformat(params, indent=4, width=160))
