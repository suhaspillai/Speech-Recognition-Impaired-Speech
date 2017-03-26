require 'optim'
require 'nnx'
require 'gnuplot'
require 'lfs'
require 'xlua'
require 'UtilsMultiGPU'
require 'Loader'
require 'nngraph'
require 'Mapper'
require 'ModelEvaluator'

local suffix = '_' .. os.date('%Y%m%d_%H%M%S')
local threads = require 'threads'
local Network = {}
local loss_inf = math.huge
local mean_layers={}
local std_layers = {}
local indices_to_layers = {}
--Training parameters
seed = 10
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(seed)

function Network:init(opt)
    self.fileName = opt.modelPath -- The file name to save/load the network from.
    self.nGPU = opt.nGPU
    self.gpu = self.nGPU > 0

    if not self.gpu then
        require 'rnn'
    else
        require 'cutorch'
        require 'cunn'
        require 'cudnn'
        require 'BatchBRNNReLU'
        cutorch.manualSeedAll(seed)
    end
    self.saveProbMatrix = opt.saveProbMatrix
    self.dropout = opt.dropout
    self.dropoutProb = opt.dropoutProb
    self.trainingSetLMDBPath = opt.trainingSetLMDBPath
    self.validationSetLMDBPath = opt.validationSetLMDBPath
    self.logsTrainPath = opt.logsTrainPath or nil
    self.logsValidationPath = opt.logsValidationPath or nil
    self.modelTrainingPath = opt.modelTrainingPath or nil

    self:makeDirectories({ self.logsTrainPath, self.logsValidationPath, self.modelTrainingPath })
    self.mapper = Mapper(opt.dictionaryPath)
    self.tester = ModelEvaluator(self.gpu, self.validationSetLMDBPath, self.mapper,
        opt.validationBatchSize, self.logsValidationPath)
    self.saveModel = opt.saveModel
    self.saveModelInTraining = opt.saveModelInTraining or false
    self.loadModel = opt.loadModel
    self.saveModelIterations = opt.saveModelIterations or 10 -- Saves model every number of iterations.
    self.maxNorm = opt.maxNorm or 400 -- value chosen by Baidu for english speech.
    -- setting model saving/loading
    if self.loadModel then
        assert(opt.modelPath, "modelPath hasn't been given to load model.")
        local model_path = opt.modelTrainingPath..opt.modelPath
        self:loadNetwork(model_path, opt.modelName)
    else
        assert(opt.modelName, "Must have given a model to train.")
        self:prepSpeechModel(opt.modelName, opt)
    end
    assert((opt.saveModel or opt.loadModel) and opt.modelPath, "To save/load you must specify the modelPath you want to save to")
    -- setting online loading
    self.indexer = indexer(opt.trainingSetLMDBPath, opt.batchSize)
    print (string.format("indexer value = %f" ,self.indexer.nbOfBatches))
    self.pool = threads.Threads(1, function() require 'Loader' end)
    self.logger = optim.Logger(self.logsTrainPath .. 'train' .. suffix .. '.log')
    self.logger:setNames { 'loss', 'WER', 'CER' }
    self.logger:style { '-', '-', '-' }
end

function Network:prepSpeechModel(modelName, opt)
    local model = require(modelName)
    self.model = model[1](opt)
    self.calSize = model[2]
end

function Network:testNetwork(epoch)
    -- This actually makes train mode as false and usefule for batch normalization and dropout
    self.model:evaluate()

    -- The below calculates WER & CER along with an input matrix, whihc is probabilities * timesteps, it is later used to apply beam search with character language model. 
    local wer, cer = self.tester:runEvaluation_v1(self.model, self.saveProbMatrix,true, epoch or 1) -- details in log        
    self.model:zeroGradParameters()   --zero the parameters
    self.model:training()
    return wer, cer
end


function Network:trainNetwork(epochs, optimizerParams)
  
    local lossHistory = {}
    local validationHistory = {}
    local criterion = nn.CTCCriterion(true)  -- call ctc loss method
    local x, gradParameters_norm = self.model:getParameters() --gives learnable parametrs and grads with respect to learnable parameters.
    local parameters,gradParameters = self.model:parameters()  
    local optimParamsLayerWise={}
    local average_norm=0 
    -- Fine tune any layer
    
    for i =1,#parameters do
      if i >48 then
        table.insert(
                    optimParamsLayerWise,{    
                      learningRate = 3e-4,
                      learningRateAnnealing = 1,
                      learningRateDecay = 0.0,
                      momentum = 0.9,
                      dampening = 0,
                      nesterov = true
                      }   
                    )
      else
        table.insert(
                    optimParamsLayerWise,{    
                      learningRate = 0.0001,--opt.learningRate,
                      learningRateAnnealing = 1, --opt.learningRateAnnealing,
                      learningRateDecay = 0.0,
                      momentum = 0.9, --opt.momentum,
                      dampening = 0,
                      nesterov = true
                      }   
                    )
      end
    end
    
    local inputs = torch.Tensor()
    local sizes = torch.Tensor()
    if self.gpu then
        criterion = criterion:cuda()
        inputs = inputs:cuda()
        sizes = sizes:cuda()
    end
    
    local loader = Loader(self.trainingSetLMDBPath, self.mapper)
    local specBuf, labelBuf, sizesBuf

    -- load first batch
    local inds = self.indexer:nextIndices()
    self.pool:addjob(function()
        return loader:nextBatch(inds)
    end,
        function(spect, label, sizes)
            specBuf = spect
            labelBuf = label
            sizesBuf = sizes
        end)
      
    
    local function feval() 
        self.pool:synchronize() -- wait previous loading
        local inputsCPU, sizes, targets = specBuf, sizesBuf, labelBuf -- move buf to training data
        inds = self.indexer:nextIndices() -- load next batch whilst training
        self.pool:addjob(function()
            return loader:nextBatch(inds)
        end,
            function(spect, label, sizes)
                specBuf = spect
                labelBuf = label
                sizesBuf = sizes
            end)
        
        inputs:resize(inputsCPU:size()):copy(inputsCPU) -- transfer over to GPU
        sizes = self.calSize(sizes)
        local predictions = self.model:forward(inputs)
        local loss =  criterion:forward(predictions, targets, sizes)
        if loss== math.huge or loss == -math.huge then loss = 0 print("Recieved an inf cost!") end  
        self.model:zeroGradParameters()
        local gradOutputs = criterion:backward(predictions,targets)
        self.model:backward(inputs,gradOutputs)
        
        local norm = gradParameters_norm:norm()
        average_norm = average_norm + norm
        if norm > self.maxNorm then
           gradParameters_norm:mul(self.maxNorm / norm)
        end
             
        -- updating gradients
        local parameters_model, gradParameters_model = self.model:parameters()
        for i =1,#parameters_model do
          local feval_layerwise = function(x)
            return loss, gradParameters_model[i]
          end
          optim.sgd(feval_layerwise,parameters_model[i],optimParamsLayerWise[i])
      end
      return gradParameters_model, {loss}   -- just to match the previous code
    end  

    -- training--
    local currentLoss
    local startTime = os.time()
    local temp = math.huge    
    local min_wer = math.huge
    local min_cer = math.huge
    local prev_cer = math.huge
    local prev_wer = math.huge
    local diff_conse_epochs = 0
    local check_param,check_gradParam = self.model:parameters()
    for i = 1, epochs do
        local averageLoss = 0
        average_norm = 0
        for j = 1, self.indexer.nbOfBatches-1 do
            currentLoss = 0
            local _,fs = feval()
            if self.gpu then cutorch.synchronize() end           
            currentLoss = currentLoss + fs[1]
            xlua.progress(j, self.indexer.nbOfBatches)
            averageLoss = averageLoss + currentLoss
            print (string.format('batch: %d --- averageLoss: %f WER: %f',j, averageLoss, min_wer * 100))
        end
        print (string.format('Avrage Norm: %d',average_norm/self.indexer.nbOfBatches))        
        self.indexer:permuteBatchOrder()
        averageLoss = averageLoss / self.indexer.nbOfBatches -- Calculate the average loss at this epoch.
        
        -- Update validation error rates
        local wer, cer = self:testNetwork(i)
        -- check if the error is reducing or not between 2 consecutive epochs else break
          if prev_wer - wer <=0.01 then           
              diff_conse_epochs = diff_conse_epochs + 1   
              if diff_conse_epochs>1 then
                  print ('Break because error decreased < than 0.01 between consecutive epochs')
                   break
              end
          else
              diff_conse_epochs = 0
          end
        prev_wer = wer
        
        -- Saving the best model based on WER score
        if min_wer>wer then
          print ('saving best model')
          min_wer = wer
          min_cer = cer
          self:saveNetwork(self.modelTrainingPath .. 'model_epoch_' .. i .. suffix .. '_' ..'best_model'..'_'.. self.fileName)
        end       
        print(string.format("Training Epoch: %d Average Loss: %f Average Validation WER: %.2f Average Validation CER: %.2f  Minimum Validation WER: %.2f Minimum Validation CER :%.2f",
            i, averageLoss, 100 * wer, 100 * cer, 100 * min_wer, 100 * min_cer))

        table.insert(lossHistory, averageLoss) -- Add the average loss value to the logger.
        table.insert(validationHistory, 100 * wer)
        self.logger:add { averageLoss, 100 * wer, 100 * cer }
        
        --anneal learning rate
        for i =1,#parameters do
          optimParamsLayerWise [i]['learningRate'] = optimParamsLayerWise [i]['learningRate'] / optimizerParams.learningRateAnnealing 
        end
        
    end
    local endTime = os.time()
    local secondsTaken = endTime - startTime
    local minutesTaken = secondsTaken / 60
    print("Minutes taken to train: ", minutesTaken)
    
    return lossHistory, validationHistory, minutesTaken
end

function Network:createLossGraph()
    self.logger:plot()
end

function Network:saveNetwork(saveName)
    self.model:clearState()
    saveDataParallel(saveName, self.model)
end

--Loads the model into Network.
function Network:loadNetwork(saveName, modelName)
   
    self.model = loadDataParallel(saveName, self.nGPU)
    if self.dropout==true then
      self:addDropout(self.dropoutProb)
    end
    print (self.model)
    local model = require(modelName)
    self.calSize = model[2]
end

function Network:makeDirectories(folderPaths)
    for index, folderPath in ipairs(folderPaths) do
        if (folderPath ~= nil) then os.execute("mkdir -p " .. folderPath) end
    end
end

function Network:addDropout(prob)
  
  -- After convolutional layers
  self.model:get(1):insert(nn.Dropout(prob):cuda(),4)
  self.model:get(1):insert(nn.Dropout(prob):cuda(),7)
  
end

return Network
