# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

MLPClassifier = None

models = {
    'RandomForest200':{
        'modelName':'rForest_n_200',
        'modelArrParameters': None,
        'modelDictParameters': {
            'n_estimators':200,
            'n_jobs': -1
            },
    'RandomForest200Entropy':{
        'modelName':'rForest_n_200_Ent',
        'modelArrParameters': None,
        'savedModelFileName':'rForest_n_200Ent.mod',
        'modelDictParameters': {
            'criterion': 'entropy',
            'n_estimators':200,
            'n_jobs': -1
    },
        'model': RandomForestClassifier,
        'savedModelFileName':'rForest_n_200.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None, 
            },
    'NaiveBayesGaussian':{
        'modelName':'nBayes',
        'modelArrParameters': None,
        'modelDictParameters': None, 
        'model': GaussianNB,
        'savedModelFileName':'naive_bayes.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None, 
            },
    'MultilayerPerceptronStandard':{
        'modelName':'mlpStandard',
        'modelArrParameters': None,
        'modelDictParameters': None, 
        'model': MLPClassifier,
        'savedModelFileName':'mlpStandard.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None,     
    },
    'MultilayerPerceptron10050':{
        'modelName':'mlp10050',
        'modelArrParameters': None,
        'modelDictParameters': {
            'hidden_layer_sizes': (100,50),
        }, 
        'model': MLPClassifier,
        'savedModelFileName':'mlp10050.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None,     
    },
    'MultilayerPerceptron20050':{
        'modelName':'mlp20050',
        'modelArrParameters': None,
        'modelDictParameters': {
            'hidden_layer_sizes': (200,50),
        }, 
        'model': MLPClassifier,
        'savedModelFileName':'mlp20050.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None,     
    },
    'MultilayerPerceptron200100':{
        'modelName':'mlp10050',
        'modelArrParameters': None,
        'modelDictParameters': {
            'hidden_layer_sizes': (200,100),
        }, 
        'model': MLPClassifier,
        'savedModelFileName':'mlp200100.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None,         
    },
    'AdaboostDecisionTree':{
        'modelName': 'adaboostDT',
        'modelArrParameters': [DecisionTreeClassifier(max_depth=5)],
        'modelDictParameters': {
            'n_estimators': 200
        }, 
        'model': AdaBoostClassifier,
        'savedModelFileName':'adaboost_decision_tree.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None, 
        
    }
        }