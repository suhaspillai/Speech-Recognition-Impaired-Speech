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
local Network_SA = {}
local loss_inf = math.huge
seed = 10
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(seed)

function Network_SA:init(opt)
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
    self.adaptLayers = opt.adaptLayers
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

function Network_SA:prepSpeechModel(modelName, opt)
    local model = require(modelName)
    self.model = model[1](opt)
    self.calSize = model[2]
end

function Network_SA:testNetwork(epoch)
    -- This actually makes train mode as false and useful for batch normalization and dropout
    self.model:evaluate()
    -- For saving predictions to be used for char language model
    local wer, cer = self.tester:runEvaluation_v1(self.model, self.saveProbMatrix,true, epoch or 1) -- details in log        
    self.model:zeroGradParameters()   --zero the parameters
    self.model:training()
    return wer, cer
end

function Network_SA:trainNetwork(epochs, optimizerParams)

    local lossHistory = {}
    local validationHistory = {}
    local criterion = nn.CTCCriterion(true)  -- call ctc loss method
    local x, gradParameters_norm = self.model:getParameters() --gives learnable parameters and grads with respect to learnable parameters.
    local parameters,gradParameters = self.model:parameters()  
    local optimParamsLayerWise={}
    local average_norm=0 
    
    -- Initializing learning rate to zero for all layers, only LHU layer is learned
    for i =1,#parameters do    
      table.insert(
                  optimParamsLayerWise,{    
                    learningRate = 0,
                    learningRateAnnealing = 1,
                    learningRateDecay = 0.0,
                    momentum = 0.9,
                    dampening = 0,
                    nesterov = true
                    }   
                  )
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
        self.pool:synchronize() -- wait for previous loading
        local inputsCPU, sizes, targets = specBuf, sizesBuf, labelBuf -- move buf to training data
        inds = self.indexer:nextIndices() -- load next batch while training
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
        
        -- Check the norm for exploding gradients (Very important when dealing with long sequences)
        local norm = gradParameters_norm:norm()
        average_norm = average_norm + norm
        if norm > self.maxNorm then
           gradParameters_norm:mul(self.maxNorm / norm)
        end      
        local parameters_model, gradParameters_model = self.model:parameters()
        
        -- Update the gradients
        for i =1,#parameters_model do
          local feval_layerwise = function(x)
            return loss, gradParameters_model[i]
          end
          optim.sgd(feval_layerwise,parameters_model[i],optimParamsLayerWise[i])
      end
      return gradParameters_model, {loss}   
    end  

    -- Training--
    local currentLoss
    local startTime = os.time()
    local temp = math.huge    
    local min_wer = math.huge
    local min_cer = math.huge
    local prev_cer = math.huge
    local prev_wer = math.huge
    local diff_conse_epochs = 0

    local check_param,check_gradParam = self.model:parameters()
    local no_of_adaptLayers = self.adaptLayers
    local learningRate_adapt = optimizerParams.adaptLayerLearningRate
    for i = 1, epochs do
        local averageLoss = 0
        average_norm = 0
        local layer_no = 13 -- Check where LHU layer is added 
        -- Update the learning rates for LHU layer
        for iter=1,no_of_adaptLayers do 
          local lhu = check_param[layer_no]
          lhu[1][1][lhu[1][1]:ge(2)] = 2
          lhu[1][1][lhu[1][1]:le(0)] = 0
          optimParamsLayerWise [layer_no]['learningRate'] = learningRate_adapt
          layer_no = layer_no + 7  --Based on my network configuration
        end
        
        for j = 1, self.indexer.nbOfBatches do
            currentLoss = 0
            local _,fs = feval()
            if self.gpu then cutorch.synchronize() end           
            currentLoss = currentLoss + fs[1]
            xlua.progress(j, self.indexer.nbOfBatches)
            averageLoss = averageLoss + currentLoss
            print (string.format('batch: %d --- averageLoss: %f',j, averageLoss))
        end
        
        --print (string.format('Avrage Norm: %d',average_norm/self.indexer.nbOfBatches))        
        self.indexer:permuteBatchOrder()
        averageLoss = averageLoss / self.indexer.nbOfBatches -- Calculate the average loss at this epoch.
        
        local wer, cer = self:testNetwork(i)
        -- Check if the (Word Error Rate) wer is reducing or not between 2 consecutive epochs else break
        if (prev_wer - wer)*100 <=0.01 then           
            diff_conse_epochs = diff_conse_epochs + 1   
            if diff_conse_epochs>1 then
                break
            end
        else
            diff_conse_epochs = 0
        end
         
        -- This is to save the best model
        if min_wer>wer then
          min_wer  =wer
          min_cer = cer
          print ('saving best model')
          self:saveNetwork(self.modelTrainingPath .. 'model_epoch_' .. i .. suffix .. '_' ..'best_model'..'_'.. self.fileName)
        end      
        print(string.format("Training Epoch: %d Average Loss: %f Average Validation WER: %.2f Average Validation CER: %.2f  Minimum Validation WER: %.2f Minimum Validation CER :%.2f",
            i, averageLoss, 100 * wer, 100 * cer, 100 * min_wer, 100 * min_cer))

        table.insert(lossHistory, averageLoss) -- Add the average loss value to the logger.
        table.insert(validationHistory, 100 * wer)
        self.logger:add { averageLoss, 100 * wer, 100 * cer }
        
        --update the learning rate for LHU layers
        learningRate_adapt = learningRate_adapt/optimizerParams.learningRateAnnealing
        prev_wer = wer
    end

    local endTime = os.time()
    local secondsTaken = endTime - startTime
    local minutesTaken = secondsTaken / 60
    print("Minutes taken to train: ", minutesTaken)
    return lossHistory, validationHistory, minutesTaken
end

function Network_SA:createLossGraph()
    self.logger:plot()
end

function Network_SA:saveNetwork(saveName)
    self.model:clearState()
    saveDataParallel(saveName, self.model)
end

--Loads the model.
function Network_SA:loadNetwork(saveName, modelName)
    self.model = loadDataParallel(saveName, self.nGPU)
    self:addLayers(self.adaptLayers)
    local model = require(modelName)
    self.calSize = model[2]
end

function Network_SA:makeDirectories(folderPaths)
    for index, folderPath in ipairs(folderPaths) do
        if (folderPath ~= nil) then os.execute("mkdir -p " .. folderPath) end
    end
end

-- Adding LHU Layer
function Network_SA:addLayers(noOfLayers)
  local new_module = nn.CMul(1,1,1760):cuda()
  local layer_no = 0 
  for i = 1 , noOfLayers do
    layer_no = 1 + 2*(i-1)  --(Based on the network configuration)
    self.model:get(2):get(layer_no):insert(new_module) -- This will add module in the end
    print (self.model)
  end
  
end
return Network_SA
  