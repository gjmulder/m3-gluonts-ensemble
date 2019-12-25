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
from pprint import pformat
from datetime import date

from hyperopt import fmin, tpe, hp, space_eval, STATUS_FAIL, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from os import environ

########################################################################################################
        
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
    dataset_name = "m4_daily"
    logger.warning("DATASET not set, using: %s" % dataset_name)
    
freq_pd = "D"
freq = 7
prediction_length = 14

def score_model(model, model_type, gluon_test_data, num_ts):
    import mxnet as mx
    from gluonts.evaluation.backtest import make_evaluation_predictions #, backtest_metrics
    from gluonts.evaluation import Evaluator
    from gluonts.model.predictor import Predictor
    from tempfile import mkdtemp
    from pathlib import Path
 
    
    if model_type != "DeepStateEstimator":
        forecast_it, ts_it = make_evaluation_predictions(dataset=gluon_test_data, predictor=model, num_samples=1)
    else:
        temp_dir_path = mkdtemp()
        model.serialize(Path(temp_dir_path))
        model_cpu = Predictor.deserialize(Path(temp_dir_path), ctx=mx.cpu())
        logger.info("Loaded DeepState model")
        forecast_it, ts_it = make_evaluation_predictions(dataset=gluon_test_data, predictor=model_cpu, num_samples=1)
        logger.info("Evaluated DeepState model")

    agg_metrics, _ = Evaluator()(ts_it, forecast_it, num_series=num_ts)
    
    return agg_metrics

def load_data(path, model_type):
    from json import loads
    
    data = {}
    for dataset in ["train", "test"]:
        data[dataset] = []
        fname = "%s/%s/data.json" % (path, dataset)
        logger.info("Reading data from: %s" % fname)
        with open(fname) as fp:
            for line in fp:
               ts_data = loads(line)
               
               # Remove static features if not supported by model
               if model_type in ['SimpleFeedForwardEstimator',
                                 'DeepFactorEstimator',
                                 'GaussianProcessEstimator']:
                   del(ts_data['feat_static_cat'])
                   
               data[dataset].append(ts_data)
               
        logger.info("Loaded %d time series from %s/%s" % (len(data[dataset]), dataset_name, dataset))

    return data
    
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
    train_data  = load_data("/var/tmp/%s" % dataset_name, cfg['model']['type'])
    num_ts = len(train_data['train'])
    
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
        clip_gradient=cfg['trainer']['clip_gradient'],
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
            cardinality=num_ts,
            max_iter_jitter=cfg['model']['max_iter_jitter'],
            sample_noise=cfg['model']['sample_noise'],
            num_parallel_samples=1,
            trainer=trainer)

    if cfg['model']['type'] == 'WaveNetEstimator':            
        estimator = WaveNetEstimator(
            freq=freq_pd,
            prediction_length=prediction_length,
            cardinality=[num_ts, 6],
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
            cardinality=[num_ts, 6]
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
            cardinality=[num_ts, 6]
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
            cardinality=[num_ts, 6],
            num_parallel_samples=1,
            trainer=trainer)
    
    logger.info("Fitting: %s" % estimator)
    gluon_train = ListDataset(train_data['train'].copy(), freq=freq_pd)
    gluon_validate = ListDataset(train_data['test'].copy(), freq=freq_pd)
    model = estimator.train(gluon_train, validation_data=gluon_validate)
    train_errs = score_model(model, cfg['model']['type'], gluon_validate, num_ts)
    logger.info("Training error: %s" % train_errs)

    test_data = load_data("/var/tmp/%s_all" % dataset_name, cfg['model']['type'])
    gluon_test = ListDataset(test_data['test'].copy(), freq=freq_pd)
    test_errs = score_model(model, cfg['model']['type'], gluon_test, num_ts)
    logger.info("Testing error : %s" % test_errs)
    
    return {
        'train' : train_errs,
        'test'  : test_errs,
    }

def gluonts_fcast(cfg):   
    from traceback import format_exc
    from os import environ as local_environ
    
    try:
        err_metrics = forecast(cfg)
        if np.isnan(err_metrics['train']['MASE']):
            raise ValueError("Training MASE is NaN")
        if np.isinf(err_metrics['train']['MASE']):
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
        
    return {
        'loss'        : err_metrics['train']['MASE'],
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
            'clip_gradient'              : hp.uniform('clip_gradient', 1, 10),              
        },

        'model' : hp.choice('model', [
            {
                'type'                       : 'SimpleFeedForwardEstimator',
                'num_hidden_dimensions'      : hp.choice('num_hidden_dimensions', [[2], [4], [8], [16], [32], [64], [128],
                                                                                   [2, 2], [4, 2], [8, 8], [8, 4], [16, 16], [16, 8], [32, 16], [64, 32],
                                                                                   [64, 32, 16], [128, 64, 32]]),
            },

#            {
#                'type'                       : 'DeepFactorEstimator',
#                'num_hidden_global'          : hp.choice('num_hidden_global', [2, 4, 8, 16, 32, 64, 128, 256]),
#                'num_layers_global'          : hp.choice('num_layers_global', [1, 2, 3]),
#                'num_factors'                : hp.choice('num_factors', [2, 4, 8, 16, 32]),
#                'num_hidden_local'           : hp.choice('num_hidden_local', [2, 4, 8]),
#                'num_layers_local'           : hp.choice('num_layers_local', [1, 2, 3]),
#            },
#                    
#            {
#                'type'                       : 'GaussianProcessEstimator',
##                'rbf_kernel_output'          : hp.choice('rbf_kernel_output', [True, False]),
#                'max_iter_jitter'            : hp.choice('max_iter_jitter', [4, 8, 16, 32]),
#                'sample_noise'               : hp.choice('sample_noise', [True, False]),
#            },
                  
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
