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
    dataset_name = "m3_monthly"
    logger.warning("DATASET not set, using: %s" % dataset_name)
    
freq_pd = "M"
freq = 12
prediction_length = 18

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

def get_trainer_hyperparams(model_cfg):
    # Trainer hyperparams have a "+" in them so we can pick them off
    trainer_cfg = {}
    for key in model_cfg.keys():
         if '+' in key:
            key_split = key.split('+', 1)[1]
            trainer_cfg[key_split] = model_cfg[key]
    return trainer_cfg

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
    
    trainer_cfg = get_trainer_hyperparams(cfg['model'])
    trainer=Trainer(
        mx.Context("gpu"),
        hybridize=False,
        epochs=trainer_cfg['max_epochs'],
        num_batches_per_epoch=trainer_cfg['num_batches_per_epoch'],
        batch_size=trainer_cfg['batch_size'],
        patience=trainer_cfg['patience'],
        
        learning_rate=trainer_cfg['learning_rate'],
        learning_rate_decay_factor=trainer_cfg['learning_rate_decay_factor'],
        minimum_learning_rate=trainer_cfg['minimum_learning_rate'],
        clip_gradient=trainer_cfg['clip_gradient'],
        weight_decay=trainer_cfg['weight_decay'],
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
            dropout_rate=cfg['model']['tf_dropout_rate'],
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
    validate_errs = score_model(model, cfg['model']['type'], gluon_validate, num_ts)
    logger.info("Validation error: %s" % validate_errs)

    test_data = load_data("/var/tmp/%s_all" % dataset_name, cfg['model']['type'])
    gluon_test = ListDataset(test_data['test'].copy(), freq=freq_pd)
    test_errs = score_model(model, cfg['model']['type'], gluon_test, num_ts)
    logger.info("Testing error : %s" % test_errs)
    
    return {
        'validate' : validate_errs,
        'test'     : test_errs,
    }

def gluonts_fcast(cfg):   
    from traceback import format_exc
    from os import environ as local_environ
    
    try:
        err_metrics = forecast(cfg)
        if np.isnan(err_metrics['validate']['MASE']):
            raise ValueError("Validation MASE is NaN")
        if np.isinf(err_metrics['validate']['MASE']):
           raise ValueError("Validation MASE is infinite")
           
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
        'loss'        : err_metrics['validate']['MASE'],
        'status'      : STATUS_OK,
        'cfg'         : cfg,
        'err_metrics' : err_metrics,
        'build_url'   : local_environ.get("BUILD_URL")
    }

def call_hyperopt():

    # Trainer hyperparams common to all models
    max_epochs = [32, 64, 128, 256, 512, 1024]
    num_batches_per_epoch = [32, 64, 128, 256, 512]
    batch_size = [32, 64, 128, 256]
    patience = [8, 16, 32, 64]
    learning_rate = {
        'min' : np.log(05e-04),
        'max' : np.log(50e-04)
    }
    learning_rate_decay_factor = {
        'min' : 0.10,
        'max' : 0.75
    }
    minimum_learning_rate = {
        'min' : np.log(005e-06),
        'max' : np.log(100e-06)
    }
    weight_decay = {
        'min' : np.log(01e-09),
        'max' : np.log(100e-09)
    }
    clip_gradient = {
        'min' :  1,
        'max' : 10
    }
    
    dropout_rate = {
        'min' : 0.07,
        'max' : 0.13
    }
    
    space = {
        'box_cox' : hp.choice('box_cox', [True, False]),
        
        'model' : hp.choice('model', [
            {
                'type'                           : 'SimpleFeedForwardEstimator',
                'num_hidden_dimensions'          : hp.choice('num_hidden_dimensions', [[2], [4], [8], [16], [32], [64], [128],
                                                                                       [2, 2], [4, 2], [8, 8], [8, 4], [16, 16], [16, 8], [32, 16], [64, 32],
                                                                                       [64, 32, 16], [128, 64, 32]]),
                   
                'sff+max_epochs'                 : hp.choice('sff+max_epochs', max_epochs),
                'sff+num_batches_per_epoch'      : hp.choice('sff+num_batches_per_epoch', num_batches_per_epoch),
                'sff+batch_size'                 : hp.choice('sff+batch_size', batch_size),
                'sff+patience'                   : hp.choice('sff+patience', patience),
                
                'sff+learning_rate'              : hp.loguniform('sff+learning_rate', learning_rate['min'], learning_rate['max']),
                'sff+learning_rate_decay_factor' : hp.uniform('sff+learning_rate_decay_factor', learning_rate_decay_factor['min'], learning_rate_decay_factor['max']),
                'sff+minimum_learning_rate'      : hp.loguniform('sff+minimum_learning_rate', minimum_learning_rate['min'], minimum_learning_rate['max']),
                'sff+weight_decay'               : hp.loguniform('sff+weight_decay', weight_decay['min'], weight_decay['max']),
                'sff+clip_gradient'              : hp.uniform('sff+clip_gradient', clip_gradient['min'], clip_gradient['max']), 
            },

#            {
#                'type'                           : 'DeepFactorEstimator',
#                'num_hidden_global'              : hp.choice('num_hidden_global', [2, 4, 8, 16, 32, 64, 128, 256]),
#                'num_layers_global'              : hp.choice('num_layers_global', [1, 2, 3]),
#                'num_factors'                    : hp.choice('num_factors', [2, 4, 8, 16, 32]),
#                'num_hidden_local'               : hp.choice('num_hidden_local', [2, 4, 8]),
#                'num_layers_local'               : hp.choice('num_layers_local', [1, 2, 3]),
#
#                'df+max_epochs'                  : hp.choice('df+max_epochs', max_epochs),
#                'df+num_batches_per_epoch'       : hp.choice('df+num_batches_per_epoch', num_batches_per_epoch),
#                'df+batch_size'                  : hp.choice('df+batch_size', batch_size),
#                'df+patience'                    : hp.choice('df+patience', patience),
#                
#                'df+learning_rate'               : hp.loguniform('df+learning_rate', learning_rate['min'], learning_rate['max']),
#                'df+learning_rate_decay_factor'  : hp.uniform('df+learning_rate_decay_factor', learning_rate_decay_factor['min'], learning_rate_decay_factor['max']),
#                'df+minimum_learning_rate'       : hp.loguniform('df+minimum_learning_rate', minimum_learning_rate['min'], minimum_learning_rate['max']),
#                'df+weight_decay'                : hp.loguniform('df+weight_decay', weight_decay['min'], weight_decay['max']),
#                'df+clip_gradient'               : hp.uniform('df+clip_gradient', clip_gradient['min'], clip_gradient['max']), 
#            },
                    
#            {
#                'type'                           : 'GaussianProcessEstimator',
##                'rbf_kernel_output'              : hp.choice('rbf_kernel_output', [True, False]),
#                'max_iter_jitter'                : hp.choice('max_iter_jitter', [4, 8, 16, 32]),
#                'sample_noise'                   : hp.choice('sample_noise', [True, False]),
#                
#                'gp+max_epochs'                  : hp.choice('gp+max_epochs', max_epochs),
#                'gp+num_batches_per_epoch'       : hp.choice('gp+num_batches_per_epoch', num_batches_per_epoch),
#                'gp+batch_size'                  : hp.choice('gp+batch_size', batch_size),
#                'gp+patience'                    : hp.choice('gp+patience', patience),
#                
#                'gp+learning_rate'               : hp.loguniform('gp+learning_rate', learning_rate['min'], learning_rate['max']),
#                'gp+learning_rate_decay_factor'  : hp.uniform('gp+learning_rate_decay_factor', learning_rate_decay_factor['min'], learning_rate_decay_factor['max']),
#                'gp+minimum_learning_rate'       : hp.loguniform('gp+minimum_learning_rate', minimum_learning_rate['min'], minimum_learning_rate['max']),
#                'gp+weight_decay'                : hp.loguniform('gp+weight_decay', weight_decay['min'], weight_decay['max']),
#                'gp+clip_gradient'               : hp.uniform('gp+clip_gradient', clip_gradient['min'], clip_gradient['max']), 
#
#            },
                  
            {
                'type'                           : 'WaveNetEstimator',
                'embedding_dimension'            : hp.choice('embedding_dimension', [2, 4, 8, 16]),
                'num_bins'                       : hp.choice('num_bins', [256, 512]),
                'n_residue'                      : hp.choice('n_residue', [22, 23, 24, 25, 26]),
                'n_skip'                         : hp.choice('n_skip', [4, 8, 16, 32, 64]),
                'dilation_depth'                 : hp.choice('dilation_depth', [None, 1, 2, 3, 4]),
                'n_stacks'                       : hp.choice('n_stacks', [1, 2, 3]),
                'wn_act_type'                    : hp.choice('wn_act_type', ['elu', 'relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']),
                
                'wn+max_epochs'                  : hp.choice('wn+max_epochs', max_epochs),
                'wn+num_batches_per_epoch'       : hp.choice('wn+num_batches_per_epoch', num_batches_per_epoch),
                'wn+batch_size'                  : hp.choice('wn+batch_size', batch_size),
                'wn+patience'                    : hp.choice('wn+patience', patience),
                
                'wn+learning_rate'               : hp.loguniform('wn+learning_rate', learning_rate['min'], learning_rate['max']),
                'wn+learning_rate_decay_factor'  : hp.uniform('wn+learning_rate_decay_factor', learning_rate_decay_factor['min'], learning_rate_decay_factor['max']),
                'wn+minimum_learning_rate'       : hp.loguniform('wn+minimum_learning_rate', minimum_learning_rate['min'], minimum_learning_rate['max']),
                'wn+weight_decay'                : hp.loguniform('wn+weight_decay', weight_decay['min'], weight_decay['max']),
                'wn+clip_gradient'               : hp.uniform('wn+clip_gradient', clip_gradient['min'], clip_gradient['max']),
            },
                   
            {
                'type'                           : 'TransformerEstimator',
                'tf_use_xreg'                    : hp.choice('tf_use_xreg', [True, False]),
                'model_dim_heads'                : hp.choice('model_dim_heads', [[2, 2], [4, 2], [8, 2], [16, 2], [32, 2], [64, 2],
                                                                                 [4, 4], [8, 4], [16, 4], [32, 4]]),
                'inner_ff_dim_scale'             : hp.choice('inner_ff_dim_scale', [2, 3, 4, 5]),
                'pre_seq'                        : hp.choice('pre_seq', ['d', 'n', 'dn', 'nd']),
                'post_seq'                       : hp.choice('post_seq', ['d', 'r', 'n', 'dn', 'nd', 'rn', 'nr', 'dr', 'rd', 'drn', 'dnr', 'rdn', 'rnd', 'nrd', 'ndr']),
                'tf_act_type'                    : hp.choice('tf_act_type', ['relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']),               
                'tf_dropout_rate'                : hp.uniform('tf_dropout_rate', dropout_rate['min'], dropout_rate['max']),
                
                'tf+max_epochs'                  : hp.choice('tf+max_epochs', max_epochs),
                'tf+num_batches_per_epoch'       : hp.choice('tf+num_batches_per_epoch', num_batches_per_epoch),
                'tf+batch_size'                  : hp.choice('tf+batch_size', batch_size),
                'tf+patience'                    : hp.choice('tf+patience', patience),
                
                'tf+learning_rate'               : hp.loguniform('tf+learning_rate', learning_rate['min'], learning_rate['max']),
                'tf+learning_rate_decay_factor'  : hp.uniform('tf+learning_rate_decay_factor', learning_rate_decay_factor['min'], learning_rate_decay_factor['max']),
                'tf+minimum_learning_rate'       : hp.loguniform('tf+minimum_learning_rate', minimum_learning_rate['min'], minimum_learning_rate['max']),
                'tf+weight_decay'                : hp.loguniform('tf+weight_decay', weight_decay['min'], weight_decay['max']),
                'tf+clip_gradient'               : hp.uniform('tf+clip_gradient', clip_gradient['min'], clip_gradient['max']),
            },

            {
                'type'                           : 'DeepAREstimator',
                'da_cell_type'                   : hp.choice('da_cell_type', ['lstm', 'gru']),
                'da_use_xreg'                    : hp.choice('da_use_xreg', [True, False]),
                'da_num_cells'                   : hp.choice('da_num_cells', [2, 4, 8, 16, 32, 64, 128, 256, 512]),
                'da_num_layers'                  : hp.choice('da_num_layers', [3]),
                
                'da_dropout_rate'                : hp.uniform('da_dropout_rate', dropout_rate['min'], dropout_rate['max']),
                
                'da+max_epochs'                  : hp.choice('da+max_epochs', max_epochs),
                'da+num_batches_per_epoch'       : hp.choice('da+num_batches_per_epoch', num_batches_per_epoch),
                'da+batch_size'                  : hp.choice('da+batch_size', batch_size),
                'da+patience'                    : hp.choice('da+patience', patience),
                
                'da+learning_rate'               : hp.loguniform('da+learning_rate', learning_rate['min'], learning_rate['max']),
                'da+learning_rate_decay_factor'  : hp.uniform('da+learning_rate_decay_factor', learning_rate_decay_factor['min'], learning_rate_decay_factor['max']),
                'da+minimum_learning_rate'       : hp.loguniform('da+minimum_learning_rate', minimum_learning_rate['min'], minimum_learning_rate['max']),
                'da+weight_decay'                : hp.loguniform('da+weight_decay', weight_decay['min'], weight_decay['max']),
                'da+clip_gradient'               : hp.uniform('da+clip_gradient', clip_gradient['min'], clip_gradient['max']),
            },

#            {
#                'type'                           : 'DeepStateEstimator',
#                'ds_cell_type'                   : hp.choice('ds_cell_type', ['lstm', 'gru']),
#                'add_trend'                      : hp.choice('add_trend', [True, False]),
#                'ds_num_cells'                   : hp.choice('ds_num_cells', [2, 4, 8, 16, 32, 64, 128, 256, 512]),
#                'ds_num_layers'                  : hp.choice('ds_num_layers', [1, 2, 3, 4, 5, 7, 9]),
#                'num_periods_to_train'           : hp.choice('num_periods_to_train', [2, 3, 4, 5, 6]),   
#                'ds_dropout_rate'                : hp.uniform('ds_dropout_rate', dropout_rate['min'], dropout_rate['max']),
#                
#                'ds+max_epochs'                  : hp.choice('ds+max_epochs', max_epochs),
#                'ds+num_batches_per_epoch'       : hp.choice('ds+num_batches_per_epoch', num_batches_per_epoch),
#                'ds+batch_size'                  : hp.choice('ds+batch_size', batch_size),
#                'ds+patience'                    : hp.choice('ds+patience', patience),
#                
#                'ds+learning_rate'               : hp.loguniform('ds+learning_rate', learning_rate['min'], learning_rate['max']),
#                'ds+learning_rate_decay_factor'  : hp.uniform('ds+learning_rate_decay_factor', learning_rate_decay_factor['min'], learning_rate_decay_factor['max']),
#                'ds+minimum_learning_rate'       : hp.loguniform('ds+minimum_learning_rate', minimum_learning_rate['min'], minimum_learning_rate['max']),
#                'ds+weight_decay'                : hp.loguniform('ds+weight_decay', weight_decay['min'], weight_decay['max']),
#                'ds+clip_gradient'               : hp.uniform('ds+clip_gradient', clip_gradient['min'], clip_gradient['max']),
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
