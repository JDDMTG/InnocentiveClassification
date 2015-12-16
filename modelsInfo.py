# -*- coding: utf-8 -*-
from scikit_learn import RandomForest
models = {
    1:{
        'modelName':'rForest_n_10',
        'modelParameters': {
            'n':10
            }, 
        'model': RandomForest.model,
        'savedModelFileName':'rForest_n_10.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None, 
            }
        }