require 'torch'
require 'nn'

require 'torch-rnn.VanillaRNN'
require 'torch-rnn.LSTM'

local utils = require 'torch-rnn.util.utils'   --changed
--local utils = require 'util.utils'

local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')


function LM:__init(kwargs)
  self.idx_to_token = utils.get_kwarg(kwargs, 'idx_to_token')
  self.token_to_idx = {}
  self.vocab_size = 0
  for idx, token in pairs(self.idx_to_token) do
    self.token_to_idx[token] = idx
    self.vocab_size = self.vocab_size + 1
  end

  self.model_type = utils.get_kwarg(kwargs, 'model_type')
  self.wordvec_dim = utils.get_kwarg(kwargs, 'wordvec_size')
  self.rnn_size = utils.get_kwarg(kwargs, 'rnn_size')
  self.num_layers = utils.get_kwarg(kwargs, 'num_layers')
  self.dropout = utils.get_kwarg(kwargs, 'dropout')
  self.batchnorm = utils.get_kwarg(kwargs, 'batchnorm')

  local V, D, H = self.vocab_size, self.wordvec_dim, self.rnn_size

  self.net = nn.Sequential()
  self.rnns = {}
  self.bn_view_in = {}
  self.bn_view_out = {}

  self.net:add(nn.LookupTable(V, D))
  for i = 1, self.num_layers do
    local prev_dim = H
    if i == 1 then prev_dim = D end
    local rnn
    if self.model_type == 'rnn' then
      rnn = nn.VanillaRNN(prev_dim, H)
    elseif self.model_type == 'lstm' then
      rnn = nn.LSTM(prev_dim, H)
    end
    rnn.remember_states = true
    table.insert(self.rnns, rnn)
    self.net:add(rnn)
    if self.batchnorm == 1 then
      local view_in = nn.View(1, 1, -1):setNumInputDims(3)
      table.insert(self.bn_view_in, view_in)
      self.net:add(view_in)
      self.net:add(nn.BatchNormalization(H))
      local view_out = nn.View(1, -1):setNumInputDims(2)
      table.insert(self.bn_view_out, view_out)
      self.net:add(view_out)
    end
    if self.dropout > 0 then
      self.net:add(nn.Dropout(self.dropout))
    end
  end

  -- After all the RNNs run, we will have a tensor of shape (N, T, H);
  -- we want to apply a 1D temporal convolution to predict scores for each
  -- vocab element, giving a tensor of shape (N, T, V). Unfortunately
  -- nn.TemporalConvolution is SUPER slow, so instead we will use a pair of
  -- views (N, T, H) -> (NT, H) and (NT, V) -> (N, T, V) with a nn.Linear in
  -- between. Unfortunately N and T can change on every minibatch, so we need
  -- to set them in the forward pass.
  self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
  self.view2 = nn.View(1, -1):setNumInputDims(2)

  self.net:add(self.view1)
  self.net:add(nn.Linear(H, V))
  self.net:add(self.view2)
end


function LM:updateOutput(input)
  local N, T = input:size(1), input:size(2)
  self.view1:resetSize(N * T, -1)
  self.view2:resetSize(N, T, -1)

  for _, view_in in ipairs(self.bn_view_in) do
    view_in:resetSize(N * T, -1)
  end
  for _, view_out in ipairs(self.bn_view_out) do
    view_out:resetSize(N, T, -1)
  end

  return self.net:forward(input)
end


function LM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end


function LM:parameters()
  return self.net:parameters()
end


function LM:training()
  self.net:training()
  parent.training(self)
end


function LM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end


function LM:resetStates()
  for i, rnn in ipairs(self.rnns) do
      rnn:resetStates()
  end
end


function LM:encode_string(s)
  local encoded = torch.LongTensor(#s)
  for i = 1, #s do
    local token = s:sub(i, i)
    local idx = self.token_to_idx[token]
    assert(idx ~= nil, 'Got invalid idx')
    encoded[i] = idx
  end
  return encoded
end


function LM:decode_string(encoded)
  assert(torch.isTensor(encoded) and encoded:dim() == 1)
  local s = ''
  for i = 1, encoded:size(1) do
    local idx = encoded[i]
    local token = self.idx_to_token[idx]
    s = s .. token
  end
  return s
end


--[[
Sample from the language model. Note that this will reset the states of the
underlying RNNs.

Inputs:
- init: String of length T0
- max_length: Number of characters to sample

Returns:
- sampled: (1, max_length) array of integers, where the first part is init.
--]]
function LM:sample(kwargs)
  local T = utils.get_kwarg(kwargs, 'length', 100)
  local start_text = utils.get_kwarg(kwargs, 'start_text', '')
  local verbose = utils.get_kwarg(kwargs, 'verbose', 0)
  local sample = utils.get_kwarg(kwargs, 'sample', 1)
  local temperature = utils.get_kwarg(kwargs, 'temperature', 1)

  local sampled = torch.LongTensor(1, T)
  self:resetStates()

  local scores, first_t
  if #start_text > 0 then
    if verbose > 0 then
      print('Seeding with: "' .. start_text .. '"')
    end
    local x = self:encode_string(start_text):view(1, -1)
    local T0 = x:size(2)
    --print ('x_size')
    --print (x:size())
    sampled[{{}, {1, T0}}]:copy(x)
    --print (sampled)
    --print (T0)
    scores = self:forward(x)[{{}, {T0, T0}}]
    
    --print ('inside start text > 0 ')
    print ('scores------',scores:max(3))
    first_t = T0 + 1
  else
    if verbose > 0 then
      print('Seeding with uniform probabilities')
    end
    local w = self.net:get(1).weight
    scores = w.new(1, 1, self.vocab_size):fill(1)
    first_t = 1
  end
  
  local _, next_char = nil, nil
  for t = first_t, T do
    if sample == 0 then
      _, next_char = scores:max(3)
      print 'hi'
      next_char = next_char[{{}, {}, 1}]
    else
       local probs = torch.div(scores, temperature):double():exp():squeeze()
       probs:div(torch.sum(probs))
       next_char = torch.multinomial(probs, 1):view(1, 1)
       --print 'samople_else'
    end
    print ('next_char',next_char)
    sampled[{{}, {t, t}}]:copy(next_char)
    scores = self:forward(next_char)
  end
  --print (scores:size())
  --print (scores)
  local temp = scores:max(3)
  token = self.idx_to_token[temp]
  --print (token)
  self:resetStates()
  return self:decode_string(sampled[1])
end


function LM:clearState()
  self.net:clearState()
end



function LM:sample_for_ctc(kwargs,output_char)
  --[[   
  local T = utils.get_kwarg(kwargs, 'length', 100)
  local start_text = utils.get_kwarg(kwargs, 'start_text', '')
  local verbose = utils.get_kwarg(kwargs, 'verbose', 0)
  local sample = utils.get_kwarg(kwargs, 'sample', 1)
  local temperature = utils.get_kwarg(kwargs, 'temperature', 1)

  local sampled = torch.LongTensor(1, T)
  self:resetStates()

  local scores, first_t
  if #start_text > 0 then
    if verbose > 0 then
      print('Seeding with: "' .. start_text .. '"')
    end
    local x = self:encode_string(start_text):view(1, -1)
    local T0 = x:size(2)
    --print ('x_size')
    --print (x:size())
    sampled[{{}, {1, T0}}]:copy(x)
    --print (sampled)
    --print (T0)
    scores = self:forward(x)[{{}, {T0, T0}}]
    
    --print ('inside start text > 0 ')
    --print ('scores------',scores:max(3))
    first_t = T0 + 1
  else
    if verbose > 0 then
      print('Seeding with uniform probabilities')
    end
    local w = self.net:get(1).weight
    scores = w.new(1, 1, self.vocab_size):fill(1)
    first_t = 1
  end
  --print (scores:size())
  
 scores = scores:double()
  --print (scores[1])
  m=nn.SoftMax()
  scores = m:forward(scores[1])
  --print (scores)
  local new_char_prob =  scores[1][self.token_to_idx[output_char]]
  --print (self.token_to_idx[output_char])
  --print (new_char_prob)
  --]] 
  --[[
  local _, next_char = nil, nil
  local new_char_prob = 0
  for t = first_t, T do
    if sample == 0 then
      --_, next_char = scores:max(3)
      --print 'hi'
      --next_char = next_char[{{}, {}, 1}]
      new_char_prob =  scores[1][1][self.token_to_idx[output_char]
      
    else
       local probs = torch.div(scores, temperature):double():exp():squeeze()
       probs:div(torch.sum(probs))
       next_char = torch.multinomial(probs, 1):view(1, 1)
       print 'samople_else'
    end
    print ('next_char',next_char)
    sampled[{{}, {t, t}}]:copy(next_char)
    scores = self:forward(next_char)
  end
  --]]

  --local temp = scores:max(3)
  --token = self.idx_to_token[temp]
  --print (token)
  --self:resetStates()
  --print (next_char[1][1])
  --print (self.idx_to_token[next_char[1][1]])
  --return new_char_prob
  
end

--[[
function LM:sample_for_ctc(kwargs)
  
  local T = utils.get_kwarg(kwargs, 'length', 100)
  local start_text = utils.get_kwarg(kwargs, 'start_text', '')
  local verbose = utils.get_kwarg(kwargs, 'verbose', 0)
  local sample = utils.get_kwarg(kwargs, 'sample', 1)
  local temperature = utils.get_kwarg(kwargs, 'temperature', 1)

  local sampled = torch.LongTensor(1, T)
  self:resetStates()

  local scores, first_t
  if #start_text > 0 then
    if verbose > 0 then
      print('Seeding with: "' .. start_text .. '"')
    end
    local x = self:encode_string(start_text):view(1, -1)
    local T0 = x:size(2)
    --print ('x_size')
    --print (x:size())
    sampled[{{}, {1, T0}}]:copy(x)
    --print (sampled)
    --print (T0)
    scores = self:forward(x)[{{}, {T0, T0}}]
    
    --print ('inside start text > 0 ')
    --print ('scores------',scores:max(3))
    first_t = T0 + 1
  else
    if verbose > 0 then
      print('Seeding with uniform probabilities')
    end
    local w = self.net:get(1).weight
    scores = w.new(1, 1, self.vocab_size):fill(1)
    first_t = 1
  end
  --print (scores)
  scores = scores:double()
  --m_log_softmax = nn.LogSoftMax()
  --scores_softmax=  m_log_softmax:forward(scores[1][1])
  
  m=nn.SoftMax()
  scores = m:forward(scores[1])
  scores = (scores[1]):log()
  
  --print (scores:size())
  --print (scores-scores_softmax)
  
  self:resetStates()
  --print (scores[1])
  return scores
end
--]]
-- for stored hidden activations 
function LM:sample_for_ctc(kwargs, prev_hidden_state)
  
  local T = utils.get_kwarg(kwargs, 'length', 100)
  local start_text = utils.get_kwarg(kwargs, 'start_text', '')
  local verbose = utils.get_kwarg(kwargs, 'verbose', 0)
  local sample = utils.get_kwarg(kwargs, 'sample', 1)
  local temperature = utils.get_kwarg(kwargs, 'temperature', 1)
  local sampled = torch.LongTensor(1, T)
  local rnn
  if prev_hidden_state == nil then 
    --print ('--from start--')
    self:resetStates()
  else
    -- for LSTM cell, can modify for rnn which ahs only onde hidden o/p
    
    for rnn_count in ipairs(self.rnns) do
      --print ('--inside prev rnn----')
      rnn = self.rnns[rnn_count]
      rnn.output[1][rnn.output:size(2)] =  prev_hidden_state[rnn_count].h0
      rnn.cell[1][rnn.output:size(2)] = prev_hidden_state[rnn_count].c0
    end
  end
  
  local scores, first_t
  if #start_text > 0 then
    if verbose > 0 then
      print('Seeding with: "' .. start_text .. '"')
    end
    local x = self:encode_string(start_text):view(1, -1)
    local T0 = x:size(2)
    --print ('x_size')
    --print (x:size())
    sampled[{{}, {1, T0}}]:copy(x)
    --print (sampled)
    --print (T0)
    scores = self:forward(x)[{{}, {T0, T0}}]
    
    --print ('inside start text > 0 ')
    --print ('scores------',scores:max(3))
    first_t = T0 + 1
  else
    if verbose > 0 then
      print('Seeding with uniform probabilities')
    end
    local w = self.net:get(1).weight
    scores = w.new(1, 1, self.vocab_size):fill(1)
    first_t = 1
  end
  --print (scores)
  scores = scores:double()
  --m_log_softmax = nn.LogSoftMax()
  --scores_softmax=  m_log_softmax:forward(scores[1][1])
  
  m=nn.SoftMax()
  scores = m:forward(scores[1])
  scores = (scores[1]):log()
  
  --print (scores:size())
  --print (scores-scores_softmax)
  
  --self:resetStates()
  --print (scores[1])
  
  local new_hidden_state ={}
  local timesteps
    
  -- create a new hidden state tbl 
  
  for rnn_count in ipairs(self.rnns) do
    new_hidden_state [rnn_count] ={}
    --print ('---inside new---')
    --print (rnn)
    rnn = self.rnns[rnn_count]
    timesteps = rnn.output:size(2)    
    new_hidden_state[rnn_count]['h0'] = rnn.output[1][timesteps]:view(1,-1):clone()
    new_hidden_state[rnn_count]['c0'] = rnn.cell[1][timesteps]:view(1,-1):clone()
  
  end
  
  return scores, new_hidden_state
end

