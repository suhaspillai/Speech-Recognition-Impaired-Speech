require 'torch-rnn.LanguageModel'
require 'audio'
require 'nn'
require 'cudnn'
require 'cunn'
require 'BatchBRNNReLU'
local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')
require 'Mapper'
require 'SequenceError'

local CTC_clm = torch.class('CTC_CLM_NN_lang_multithread')

function CTC_clm:__init(dictdata) 
  self.map_obj = Mapper(dictdata)
  self.opt = {}
  self.opt['gpu']  =  0
  self.opt ['verbose'] = 0
  self.opt['length'] = 2000
  self.opt['checkpoint'] = '/home/sbp3624/CTCSpeechRecognition/torch-rnn/cv_torgo_no_drop/checkpoint_1900.t7'
  self.opt['gpu_backend'] = "cuda"
  self.opt['temperature'] = 1
  self.opt['start_text'] = ""
  self.opt['sample'] = 0
  self.lm_model = torch.load('/home/sbp3624/CTCSpeechRecognition/torch-rnn/cv_torgo_no_drop/checkpoint_1900.t7')
  self.lm_model = self.lm_model.model
  self.lm_model:evaluate()
end


-- calculate probability using language model
 function CTC_clm:get_clm_prob(prefix, hidden_state_cache)
  local inp_seq_len
  local prob, prev_h_states , curr_h_states
  local input_seq
  local temp_check_seq 
  if prefix:len()==0 then
    input_seq = prefix..' '  
  else
    input_seq = prefix
  end
  
  input_seq = input_seq:upper()
  inp_seq_len = input_seq:len()
  temp_check_seq = string.sub(input_seq,1,inp_seq_len-1)
 
    if hidden_state_cache[temp_check_seq]==nil then
      self.opt['start_text'] = input_seq
      self.opt['length'] = inp_seq_len  -- send the entire input sequene
      prev_h_states = hidden_state_cache[temp_check_seq]
      prob, curr_h_states = self.lm_model:sample_for_ctc(self.opt, prev_h_states)
    else
      self.opt['start_text'] = string.sub(input_seq,inp_seq_len,inp_seq_len)
      self.opt['length'] = 1 -- because that is the only character given as an input
      prev_h_states = hidden_state_cache[temp_check_seq]
      prob, curr_h_states = self.lm_model:sample_for_ctc(self.opt,prev_h_states)
    end
     
    if hidden_state_cache[input_seq] == nil then
      hidden_state_cache[input_seq] = curr_h_states
    end
    return prob, hidden_state_cache
end



function CTC_clm:getvalue(tbl, prefix) 

  if tbl[prefix]==nil then
      tbl[prefix] = {-math.huge,-math.huge, 0} 
      return tbl[prefix]
  else
     return tbl[prefix]  
  end
end



function CTC_clm:initialize_prefix(prefix) 
  
  return {0,0,prefix:len()}  -- prob_notbalnk , prob_blank , len(prefix)
end



function CTC_clm:cal_combine_prob(tot_prob) 
  local prob = 0
  prob = math.exp(tot_prob.a) + math.exp(tot_prob.b) + math.exp(tot_prob.c)
  if prob == 0 then
    return (-1/0)
  else
    return math.log(prob)
  end
 
end


function CTC_clm:find_most_probable_seq(H_next, beam_val, beta) 
  local tbl_sort = {}
  
  for prefix, values in pairs(H_next) do
    local p_nb = values[1]
    local p_b = values[2]
    --local str_len = values[3]
    local str_len = values[3]
    local tot_prob ={}
    tot_prob['a'] = p_nb
    tot_prob['b'] = p_b
    tot_prob['c'] =-1/0
    tbl_sort[prefix] = self:cal_combine_prob(tot_prob) + beta * str_len--math.log(p_nb * p_b * math.pow(str_len,beta))
  end
 
  --sort the values
  local sortedKeys = self:getKeysSortedByValue(tbl_sort, function(a, b) return a > b end)
  
  --get top combinations
  local tbl_beam = {}
  local count = 1
  for prefix,values in ipairs(sortedKeys) do

    if count>beam_val then 
      break
    else

      tbl_beam[values] = H_next[values]
      count = count + 1
    end
    
  end
  
  return tbl_beam
end

function CTC_clm:getKeysSortedByValue(tbl, sortFunction)
  local keys = {}
  for key in pairs(tbl) do
    table.insert(keys, key)
  end

  table.sort(keys, function(a, b)
    return sortFunction(tbl[a], tbl[b])
  end)

  return keys
end


function CTC_clm:print_tbl(H_prev)
  for i,v in pairs(H_prev) do
    print (i,v)
  end
end

-- decode using beam search algorithm and character language model and CTC.
-- [[Based on Lexicon-Free Conversational Speech Recognition with Neural Networks --]]
function CTC_clm:decode_beam_search(probs, alpha, beta, beam_val) 

  local N = probs:size(1)
  local T = probs:size(2)
  local H_prev = {}
  local initial_prefix = ''   -- This is an empty string 

  local inf_var = -1/0
  H_prev[initial_prefix] =  {inf_var,0,0}
  local prob_tbl={a=1,b=1,c=1}
  local H_old={}
  local prev_hidden_cache={}
  local char_lm_prob
  for t = 1, T do
    
    local H_next = {}  -- This is for new combinations which will be formed.
    
    for prefix, values  in pairs(H_prev) do
      local prev_p_nb = values[1] -- probability of not blank
      local prev_p_b = values[2]  -- probability of blank
      local prev_str_len = values[3]  -- length of the prefix
      local prob_ctc = probs[1][t]  -- cuz empty string is  a blank.
      local valsP = self:getvalue(H_next,prefix)  
      prob_tbl.a = prob_ctc + prev_p_nb
      prob_tbl.b = prob_ctc + prev_p_b --prev_prob_tot
      prob_tbl.c = valsP[2]--H_next[prefix][2]--inf_var 
      valsP[2] = self:cal_combine_prob(prob_tbl)  
      valsP[3] = prev_str_len
    
      if prev_str_len>0 then

        local repeat_char = prefix:sub(prev_str_len,prev_str_len)
        local index = self.map_obj.alphabet2token[repeat_char]
        prob_tbl.a = probs[index+1][t] + prev_p_nb --modified to get proper index
        prob_tbl.b = valsP[1]--inf_var
        prob_tbl.c = inf_var--H_next[prefix][1]--inf_var
        valsP[1] = self:cal_combine_prob(prob_tbl)  
      end
      -- Now we need to iterate over the all the charaters 
   
    char_lm_prob,prev_hidden_cache = self:get_clm_prob(prefix, prev_hidden_cache) 

      for index_char = 1, #self.map_obj.token2alphabet do   --this will go only till 28
        local new_char = self.map_obj.token2alphabet[index_char]
        -- get index of charcter language model
        local rnn_index_char = self.lm_model.token_to_idx[new_char:upper()]
        local clm_prob = alpha * char_lm_prob[rnn_index_char]
        local new_prefix = prefix .. new_char 
        local valsN = self:getvalue(H_next,new_prefix)
        valsN[3] = prev_str_len + 1
       
        if new_char ~= prefix:sub(prev_str_len,prev_str_len) and prev_str_len>0 or prev_str_len ==0 then
          prob_tbl.a = probs[index_char+1][t] + clm_prob +  prev_p_nb
          prob_tbl.b = probs[index_char+1][t] + clm_prob +  prev_p_b
          prob_tbl.c = valsN[1]--H_next[new_prefix][1] --inf_var
          valsN[1] = self:cal_combine_prob(prob_tbl)

        else
          prob_tbl.a = probs[index_char+1][t] + clm_prob + prev_p_b
          prob_tbl.b = valsN[1]--inf_var
          prob_tbl.c = inf_var
          valsN[1] = self:cal_combine_prob(prob_tbl) 
        end
        if H_prev[new_prefix]==nil then
          local v_old = self:getvalue(H_old,new_prefix)
          prob_tbl.a = v_old[1] + probs[1][t]
          prob_tbl.b = v_old[2] + probs[1][t]
          prob_tbl.c = valsN[2] 
          valsN[2] = self:cal_combine_prob(prob_tbl) 
         
         -- for non-blank 

         prob_tbl.a =  v_old[1] + probs[index_char+1][t]
         prob_tbl.b = valsN[1] 
         prob_tbl.c = inf_var  
         valsN[1] = self:cal_combine_prob(prob_tbl)
        end
      end
    end
    H_old = H_next
    H_prev = self:find_most_probable_seq(H_next, beam_val, beta)
    
    
    -- Now remove those that are not in H_prev after getting the most likely sentences
    for i,v in pairs(prev_hidden_cache) do
      
      if H_prev[i]==nil and H_prev[i:sub(1,i:len()-1)] then
        prev_hidden_cache[i] = nil --removing the entry from the table.
      end
    end
  end  
  
  local scores = -math.huge
  local tbl={}
  local final_sent
  for i,v in pairs(H_prev) do
    tbl.a=v[1]
    tbl.b = v[2]
    tbl.c = v[3]
    local temp_scores = self:cal_combine_prob(tbl)
    if scores< temp_scores then
        final_sent = i
        scores = temp_scores
    end
  end
  return final_sent
end

