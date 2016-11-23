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
local loss_inf = 1/0
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
    -- You can change this to dropout
    self.maxNorm = opt.maxNorm or 400 -- value chosen by Baidu for english speech.
    -- setting model saving/loading
    print ('Before going to loadModel')
    print (self.loadModel)
    if self.loadModel then
        assert(opt.modelPath, "modelPath hasn't been given to load model.")
        local model_path = opt.modelTrainingPath..opt.modelPath
        print ('---Inside Load modelpath---')
        print (model_path)
        --self:loadNetwork(opt.modelPath, opt.modelName)
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
    local wer, cer = self.tester:runEvaluation(self.model, true, epoch or 1) -- details in log
    self.model:zeroGradParameters()   --zero the parameters
    self.model:training()
    return wer, cer
end

function Network:trainNetwork(epochs, optimizerParams)
--function Network:trainNetwork(epochs, optimParamsLayerWise)
    self.model:training()

    local lossHistory = {}
    local validationHistory = {}
    local criterion = nn.CTCCriterion(true)  -- call ctc loss method
    --local x, gradParameters = self.model:getParameters() --gives learnable parametrs and grads with respect to learnable parameters.
    local parameters,gradParameters = self.model:parameters()  
    local optimParamsLayerWise={}
    for i =1,#parameters do
      if i >8 then
      --if i < 9 then 
        
      table.insert(
                  optimParamsLayerWise,{    
                    learningRate = 0.0001,
                    learningRateAnnealing = 1,
                    learningRateDecay = 0.0,
                    momentum = 0.9,
                    dampening = 0,
                    nesterov = true
                    --weightDecay = 5e-4
                    }   
                  )
      else
      table.insert(
                  optimParamsLayerWise,{    
                    learningRate = 0.0,--opt.learningRate,
                    learningRateAnnealing = 1, --opt.learningRateAnnealing,
                    learningRateDecay = 0.0,
                    momentum = 0.9, --opt.momentum,
                    dampening = 0,
                    nesterov = true
                    --weightDecay = 5e-4
                    }   
                  )
      end
    end
    --print("Number of parameters: ", gradParameters:size(1))
    print (optimParamsLayerWise)
    -- inputs (preallocate)
    local inputs = torch.Tensor()
    local sizes = torch.Tensor()
    if self.gpu then
        criterion = criterion:cuda()
        inputs = inputs:cuda()
        sizes = sizes:cuda()
    end

    -- def loading buf and loader
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
    
    -- define the feval
    --inp_err = torch.load('inf_inputs.dat')
    --[[
    local function feval(x_new)
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
      
        --inputs = inp_err
        local predictions = self.model:forward(inputs)
        --print (predictions)
        local loss = criterion:forward(predictions, targets, sizes)
        self.model:zeroGradParameters()
        if loss == loss_inf then 
          local file_name = 'inf_inputs.dat'
          local file_labels = 'inf_labels.dat'
          local file_sizes = 'inf_sizes.dat'
          torch.save(file_name,inputsCPU)
          torch.save(file_labels,targets)
          torch.save(file_sizes,sizes)
          print ('---incorrect inputs saved---')
          return 0, gradParameters:zero() 
        else
          local gradOutput = criterion:backward(predictions, targets)
          self.model:backward(inputs, gradOutput)
          local norm = gradParameters:norm()
          if norm > self.maxNorm then
              gradParameters:mul(self.maxNorm / norm)
          end
          return loss, gradParameters 
        end
    end
     --]] 
  
      
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
        self.model:zeroGradParameters()
        local gradOutputs = criterion:backward(predictions,targets)
        self.model:backward(inputs,gradOutputs)
        
        local parameters_model, gradParameters_model = self.model:parameters()
        for i =1,#parameters_model do
          local feval_layerwise = function(x)
            return loss, gradParameters_model[i]
          end
          optim.sgd(feval_layerwise,parameters_model[i],optimParamsLayerWise[i])
 
      end
      return gradParameters_model, {loss}   -- just to match the previous code
    end  

    -- training
    local currentLoss
    local startTime = os.time()
    local temp = 1/0    
    local min_wer = 1000
    
    local check_param,check_gradParam = self.model:parameters()
    --print (check_param:size())
    --print (check_param[1][1][1][1])
     
    for i = 1, epochs do
        local averageLoss = 0
      
        for j = 1, self.indexer.nbOfBatches do
            currentLoss = 0
            --local _, fs = optim.sgd(feval, x, optimizerParams)
            local _,fs = feval()
            
           --[[ local _, gradParameters = self.model:getParameters()

            local norm = gradParameters:norm()
            if norm > self.maxNorm then
              gradParameters:mul(self.maxNorm / norm)
            end
            --]]
            if self.gpu then cutorch.synchronize() end
            if fs[1] == 0 then
              print ('---------------------Inisde==0------------------')
              local file_name = 'err_model.t7'
              torch.save(file_name,self.model)
            end
            
            currentLoss = currentLoss + fs[1]
            xlua.progress(j, self.indexer.nbOfBatches)
            averageLoss = averageLoss + currentLoss
            print (string.format('batch: %d --- averageLoss: %f',j, averageLoss))
            --local dummy_param, dummy_gradParam = self.model:parameters()
            --print(dummy_param[1][1][1][1])  
            --print (dummy_gradParam[1][1][1][1]) 
        end
        
        
        self.indexer:permuteBatchOrder()

        averageLoss = averageLoss / self.indexer.nbOfBatches -- Calculate the average loss at this epoch.

--[[
        if (i%5==0) then
          print('After 5 epochs')
        -- anneal learningRate
          optimizerParams.learningRate = optimizerParams.learningRate / (optimizerParams.learningRateAnnealing or 1)
        end
    --]]
     for j = 1,#parameters do
       if j > 48 then
        optimParamsLayerWise[j]['learningRate'] = (optimParamsLayerWise[j]['learningRate'])/(optimizerParams.learningRateAnnealing)
      end
    end
    
    --print(optimParamsLayerWise)
     
      
        --optimizerParams.learningRate = optimizerParams.learningRate / (optimizerParams.learningRateAnnealing or 1)
        
        -- Update validation error rates
        local wer, cer = self:testNetwork(i)
        if min_wer > wer then
          min_wer  =wer
          --print ('Saving Model')      
          --self:saveNetwork('Best_model')
        end
        
        print(string.format("Training Epoch: %d Average Loss: %f Average Validation WER: %.2f Minimum Validation WER: %.2f Average Validation CER: %.2f",
            i, averageLoss, 100 * wer, 100 * min_wer, 100 * cer))

        table.insert(lossHistory, averageLoss) -- Add the average loss value to the logger.
        table.insert(validationHistory, 100 * wer)
        self.logger:add { averageLoss, 100 * wer, 100 * cer }

        -- periodically save the model
        if self.saveModelInTraining and i % self.saveModelIterations == 0 then
          print("Saving model..")
            self:saveNetwork(self.modelTrainingPath .. 'model_epoch_' .. i .. suffix .. '_' .. self.fileName)
        end
        
        
     for i =1,#parameters do
     -- if i < 9 and i>50 then
        optimParamsLayerWise [i]['learningRate'] = optimParamsLayerWise [i]['learningRate'] / optimizerParams.learningRateAnnealing or 1
      --end
     
    end       
       
        
        
        
    end

    local endTime = os.time()
    local secondsTaken = endTime - startTime
    local minutesTaken = secondsTaken / 60
    print("Minutes taken to train: ", minutesTaken)

    if self.saveModel then
     print("Saving model..")
        self:saveNetwork(self.modelTrainingPath .. 'final_model_' .. suffix .. '_' .. self.fileName) 
    end

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
    local model = require(modelName)
    self.calSize = model[2]
end

function Network:makeDirectories(folderPaths)
    for index, folderPath in ipairs(folderPaths) do
        if (folderPath ~= nil) then os.execute("mkdir -p " .. folderPath) end
    end
end

return Network
