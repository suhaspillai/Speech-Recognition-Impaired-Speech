require 'nn'
require 'cudnn'
require 'BatchBRNNReLU'
require 'audio'
require 'torch'
require 'Mapper'
require 'cunn'

local m_obj = Mapper('./dictionary_phoneme')
local sampling_rate = 16000
local sampling_stride = 0.01
local sampling_window = 0.02
local input_sample ='./dysarthric_dataset/Normal/val/MC01_Session2_arryMic_0163.wav'
local input_audio = audio.load(input_sample)
local input_spect = audio.spectrogram(input_audio,sampling_rate * sampling_window, 'hamming', sampling_rate * sampling_stride)
print (input_spect:size())
local mean = input_spect:mean()
local std = input_spect:std()
input_spect:add(-mean)
input_spect:div(std)

--print (input_spect[1])
local model_path ='./models/model_epoch_20_20161121_022303_model_with_BN_all_rnn_finetune.t7'
local model = torch.load(model_path)
local input = torch.Tensor(1,input_spect:size(1),input_spect:size(2))
input[1]=input_spect
input = input:cuda()
model:evaluate()
local predictions = model:forward(input)
token = m_obj:decodeOutput(predictions[1])
--print (token)
text = m_obj:tokensToText(token)
print (text)

local max, max_indices = predictions[1]:max(2)
max_indices = max_indices:transpose(1,2)
--print (max_indices:size())
--print (max_indices)
local start_pos  = 1
local end_pos = 1
local phoneme_timestep = {}
local counter_to_phoneme={}
local counter = 1

for i = 1, max_indices:size(2)-1 do
  
  if max_indices[1][i]~=max_indices[1][i+1] then
    local index = max_indices[1][i]
    --print (index)
    end_pos = i
    table.insert(counter_to_phoneme,counter,index)
    table.insert(phoneme_timestep,counter,{start_pos,end_pos})
    start_pos = i+1
    counter = counter + 1
  end
end



-- If only the same label across all the time steps (takes care of only one symbol and last repeating symbol)
--if start_pos == 1 then
  local index = max_indices[1][start_pos]
  --print (index:size())
  end_pos = max_indices:size(2)
  table.insert(phoneme_timestep,counter,{start_pos,end_pos})
  table.insert(counter_to_phoneme,counter,index)  

--[[
else
  index = max_indices[1][start_pos]
  end_pos = max_indices:size(2)
  table.insert(phoneme_timestep,counter,{start_pos,end_pos})
  table.insert(counter_to_phoneme,counter,index)  
  
--]]

local tot_timesteps_per_phn = {}
local conv_layers={}
conv_layers[1] = {11,2}
conv_layers[2] = {11,2}

for k = 1,#counter_to_phoneme do
  
  start_pos = phoneme_timestep[k][1]
  end_pos = phoneme_timestep[k][2]
    
    for layer, layers_values in ipairs(conv_layers) do
      local filter_w = layers_values[1]
      local stride = layers_values[2]
      if k == 1 then
        start_pos = 1
        end_pos =  end_pos * stride + stride - 1
      elseif k>1 and k< #counter_to_phoneme then
        start_pos =  start_pos * stride
        end_pos =  end_pos * stride + stride - 1
      else
        start_pos = start_pos * stride 
        end_pos = end_pos * stride + filter_w-1
      end
    end
    table.insert(tot_timesteps_per_phn,k, {start_pos,end_pos})
  
end
print (tot_timesteps_per_phn)
print (counter_to_phoneme)

for i  = 1,max_indices:size(2) do
  print (m_obj.token2alphabet[max_indices[1][i]-1])
  --print (max_indices[1][i])
end

--[[
for k,v in ipairs(m_obj.token2alphabet) do
  print (k,v)
end--]]