local Network = require 'Network'
io.stdout:setvbuf("no")
-- Options can be overrided on command line run.
local cmd = torch.CmdLine()
cmd:option('-loadModel', true, 'Load previously saved model')
cmd:option('-saveModel', true, 'Save model after training/testing')
cmd:option('-modelName', 'DeepSpeechModel', 'Name of class containing architecture')
cmd:option('-nGPU', 1, 'Number of GPUs, set -1 to use CPU')
cmd:option('-trainingSetLMDBPath','./dysarthric_dataset/torgo_norm_lmdb/train/', 'Path to LMDB training dataset')
cmd:option('-validationSetLMDBPath','./dysarthric_dataset/torgo_norm_lmdb/val/', 'Path to LMDB test dataset')
cmd:option('-logsTrainPath', './logs/TrainingLoss/all_rnn_tune/', ' Path to save Training logs')
cmd:option('-logsValidationPath', './logs/ValidationScores/all_rnn_tune/', ' Path to save Validation logs')
cmd:option('-saveModelInTraining', true, 'save model periodically through training')
--cmd:option('-modelTrainingPath', './models_lstm/', ' Path to save periodic training models')
cmd:option('-modelTrainingPath', './models/', ' Path to save periodic training models')
cmd:option('-saveModelIterations', 10, 'When to save model through training')
cmd:option('-modelPath', 'model_with_BN_all_rnn_finetune.t7', 'Path of final model to save/load')
--cmd:option('-modelPath','deepspeech_lstm.t7','Path to save model')
--cmd:option('-modelPath','model_epoch_10_20161111_234846_gru_deep_speech.t7','Path to save model')

cmd:option('-dictionaryPath', './dictionary_phoneme', ' File containing the dictionary to use')
cmd:option('-epochs', 20, 'Number of epochs for training')
cmd:option('-learningRate', 3e-4, ' Training learning rate')
--cmd:option('-learningRate', 0.0001, ' Training learning rate')
cmd:option('-learningRateAnnealing', 1.1, 'Factor to anneal lr every epoch')
cmd:option('-maxNorm', 400, 'Max norm used to normalize gradients')
cmd:option('-momentum', 0.90, 'Momentum for SGD')
cmd:option('-batchSize', 10, 'Batch size in training')
cmd:option('-validationBatchSize', 10, 'Batch size for validation')
cmd:option('-hiddenSize', 1760, 'RNN hidden sizes')
--cmd:option('-hiddenSize', 968, 'RNN hidden sizes')
cmd:option('-nbOfHiddenLayers', 7, 'Number of rnn layers')
cmd:option('-gru', false, 'gru activation')
cmd:option('-lstm', false, 'lstm activation')
cmd:option('-bn',false, 'use batch normalization')
cmd:option('-dropout',0,'apply dropout')

local opt = cmd:parse(arg)
--local batches_done = 25
--opt.learningRate = opt.learningRate * batches_done 
--Parameters for the stochastic gradient descent (using the optim library).
local optimParams = {
    learningRate = opt.learningRate,
    learningRateAnnealing = opt.learningRateAnnealing,
    momentum = opt.momentum,
    dampening = 0,
    nesterov = true
}


local optimParams_table ={}

--[[
table.insert(
 optimParams_table,{    
                    learningRate = opt.learningRate,
                    learningRateAnnealing = opt.learningRateAnnealing,
                    momentum = opt.momentum,
                    dampening = 0,
                    nesterov = true
                    }   
            )
--]]

--optimParams.learningRate = optimParams.learningRate/(optimParams.learningRateAnnealing * optimParams.learningRateAnnealing)
--Create and train the network based on the parameters and training data.
Network:init(opt)

print (opt.nbOfHiddenLayers)
Network:trainNetwork(opt.epochs, optimParams)

--Creates the loss plot.
Network:createLossGraph()
