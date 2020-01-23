#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 08:17:31 2019

@author: mulderg
"""

# awk 'BEGIN {print "date.time,series,value"} /metric..epoch_loss/ {print $1, $2 ",training_loss,", substr($NF, 14)} /metric..validation_epoch_loss/ {print $1, $2 ",final_loss,", substr($NF, 25)} /Learning/ {print $1, $2 ",learning.rate,", $NF}' 1440epochs.out > final_run_training_loss.csv
# read_csv("final_run_training_loss.csv") %>% filter(value < 10) %>% ggplot(aes(x = date.time, y = value)) + geom_point(size=0.1) + geom_smooth() + facet_grid(series ~ ., scales = "free") + scale_y_log10()

from logging import basicConfig, getLogger
from logging import DEBUG as log_level
#from logging import INFO as log_level
basicConfig(level = log_level,
            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)

from m4_nextgen import gluonts_fcast
from pprint import pformat
    
if __name__ == "__main__":
#            {
#                'type'                           : 'DeepAREstimator',
#                'da_cell_type'                   : hp.choice('da_cell_type', ['lstm']),
#                'da_use_xreg'                    : hp.choice('da_use_xreg', [True]),
#                'da_num_cells'                   : hp.choice('da_num_cells', [512]),
#                'da_num_layers'                  : hp.choice('da_num_layers', [3]),
#                
#                'da_dropout_rate'                : hp.choice('da_dropout_rate', [0.07138173893167203]),
#                
#                'da+max_epochs'                  : hp.choice('da+max_epochs', [128]),
#                'da+num_batches_per_epoch'       : hp.choice('da+num_batches_per_epoch', [256]),
#                'da+batch_size'                  : hp.choice('da+batch_size', [32]),
#                'da+patience'                    : hp.choice('da+patience', [8]),
#                
#                'da+learning_rate'               : hp.choice('da+learning_rate', [0.0024662237237407514]),
#                'da+learning_rate_decay_factor'  : hp.choice('da+learning_rate_decay_factor', [0.16667217194294942]),
#                'da+minimum_learning_rate'       : hp.choice('da+minimum_learning_rate', [0.00003181948013529227]),
#                'da+weight_decay'                : hp.choice('da+weight_decay', [1.538154835980446e-8]),
#                'da+clip_gradient'               : hp.choice('da+clip_gradient', [6.121571111575984]),
#            },
    cfg = {
            "box_cox" : False,
            'rand_seed' : 4242,
            "model" : {
                    "da_cell_type" : "lstm",
                    "da_dropout_rate" : 0.07138173893167203,
                    "da_num_cells" : 512,
                    "da_num_layers" : 3,
                    "da_use_xreg" : True,
                    "type" : "DeepAREstimator",
                    "da+batch_size" : 32,
                    "da+clip_gradient" : 6.121571111575984,
                    "da+learning_rate" : 0.0024662237237407514,
                    "da+learning_rate_decay_factor" : 0.16667217194294942,
                    "da+max_epochs" : 128,
                    "da+minimum_learning_rate" : 0.00003181948013529227,
                    "da+num_batches_per_epoch" : 256,
                    "da+patience" : 8,
                    "da+weight_decay" : 1.538154835980446e-8
            }
        }

    results = gluonts_fcast(cfg)
    logger.info("Final results:\n%s" % pformat(results, indent=4, width=160))
