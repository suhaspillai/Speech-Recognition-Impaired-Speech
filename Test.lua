local Network = require 'Network'

-- Load the network from the saved model. Options can be overrided on command line run.
local cmd = torch.CmdLine()
cmd:option('-loadModel', true, 'Load previously saved model')
cmd:option('-saveModel', false, 'Save model after training/testing')
cmd:option('-modelName', 'DeepSpeechModel', 'Name of class containing architecture')
cmd:option('-modelTrainingPath', './models/', ' Path to save periodic training models')
cmd:option('-nGPU', 1, 'Number of GPUs, set -1 to use CPU')
cmd:option('-trainingSetLMDBPath', './dysarthric_dataset/torgo_norm_lmdb/train/', 'Path to LMDB training dataset')
cmd:option('-validationSetLMDBPath', './dysarthric_dataset/torgo_norm_lmdb/test/', 'Path to LMDB test dataset')
cmd:option('-logsTrainPath', './logs/TrainingLoss/', ' Path to save Training logs')
cmd:option('-logsValidationPath', './logs/Validation_test/', ' Path to save Validation logs')
--cmd:option('-modelPath', 'model_epoch_5_20161117_040347_final_model__20161116_130513_model_epoch_25_20161027_205220_model_epoch_5_20161027_113814_final_model__20161020_214733_deepspeech.t7', 'Path of final model to save/load')
--cmd:option('-modelPath', 'model_epoch_10_20161117_021126_final_model__20161116_130513_model_epoch_25_20161027_205220_model_epoch_5_20161027_113814_final_model__20161020_214733_deepspeech.t7', 'Path of final model to save/load')
cmd:option('-modelPath','model_epoch_20_20161121_022303_model_with_BN_all_rnn_finetune.t7','Path for model')
cmd:option('-dictionaryPath', './dictionary_phoneme', ' File containing the dictionary to use')
cmd:option('-batchSize', 10, 'Batch size in training')
cmd:option('-validationBatchSize', 10, 'Batch size for validation')

local opt = cmd:parse(arg)

Network:init(opt)

print("Testing network...")
local wer, per = Network:testNetwork()
print(string.format('Avg WER: %2.f  Avg PER: %.2f', 100 * wer, 100 * per))
print(string.format('More information written to log file at %s', opt.logsValidationPath))
