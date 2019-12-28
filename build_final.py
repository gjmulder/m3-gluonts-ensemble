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

from plos_m3 import rand_seed, gluonts_fcast
import mxnet as mx
import numpy as np
from pprint import pformat

mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)
    
if __name__ == "__main__":
#                "loss" : 4.346287863706254,
#                "status" : "ok",
                cfg = {
                        "box_cox" : false,
                        "model" : {
                                "da_cell_type" : "lstm",
                                "da_dropout_rate" : 0.07352792063237133,
                                "da_num_cells" : 128,
                                "da_num_layers" : 4,
                                "da_use_xreg" : true,
                                "type" : "DeepAREstimator"
                        },
                        "trainer" : {
                                "batch_size" : 32,
                                "clip_gradient" : 4.201733453345678,
                                "learning_rate" : 0.0014595770409799346,
                                "learning_rate_decay_factor" : 0.675310144018115,
                                "max_epochs" : 1024,
                                "minimum_learning_rate" : 0.00001158227886748996,
                                "num_batches_per_epoch" : 128,
                                "patience" : 64,
                                "weight_decay" : 1.8916583050673646e-8
                        }
             }

    results = gluonts_fcast(cfg)
    logger.info("Final results:\n%s" % pformat(results, indent=4, width=160))
