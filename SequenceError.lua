local SequenceError = torch.class("SequenceError")

-- Calculates a sequence error rate (aka Levenshtein edit distance)
function SequenceError:sequenceErrorRate(target_sent, prediction_sent)

    --print (target)
    --print (prediction)
    --print ('-----------')
    local target = target_sent:split(' ')
    local prediction = prediction_sent:split(' ')
    --print (type(target))
    --print (type(prediction))
    local d = torch.Tensor(#target + 1, #prediction + 1):zero()
    for i = 1, #target + 1 do
        for j = 1, #prediction + 1 do
            if (i == 1) then
                d[1][j] = j - 1
            elseif (j == 1) then
                d[i][1] = i - 1
            end
        end
    end

    for i = 2, #target + 1 do
        for j = 2, #prediction + 1 do      
            if (target[i - 1] == prediction[j - 1]) then
            --if (string.sub(target,i-1,i-1)== string.sub(prediction,j-1,j-1)) then
              -- print (string.sub(target,i-1,i-1))
              -- print(string.sub(prediction,j-1,j-1))
              -- print('-------------')
                d[i][j] = d[i - 1][j - 1]
            else
                local substitution = d[i - 1][j - 1] + 1
                local insertion = d[i][j - 1] + 1
                local deletion = d[i - 1][j] + 1
                d[i][j] = torch.min(torch.Tensor({ substitution, insertion, deletion }))
            end
        end
    end
    local errorRate = d[#target + 1][#prediction + 1] / #target
    return errorRate
end


function SequenceError:sequenceErrorRate_WER(target, prediction)
  
      --print (target)
      --print (prediction)
      --print ('------------WER----------------')
      local d = torch.Tensor(#target + 1, #prediction + 1):zero()
      for i = 1, #target + 1 do
          for j = 1, #prediction + 1 do
              if (i == 1) then
                  d[1][j] = j - 1
              elseif (j == 1) then
                  d[i][1] = i - 1
              end
          end
      end

      for i = 2, #target + 1 do
          for j = 2, #prediction + 1 do
              if (target[i - 1] == prediction[j - 1]) then
              --if (string.sub(target,i-1,i-1)== string.sub(prediction,j-1,j-1)) then
                -- print (string.sub(target,i-1,i-1))
                -- print(string.sub(prediction,j-1,j-1))
                -- print('-------------')
                  d[i][j] = d[i - 1][j - 1]
              else
                  local substitution = d[i - 1][j - 1] + 1
                  local insertion = d[i][j - 1] + 1
                  local deletion = d[i - 1][j] + 1
                  d[i][j] = torch.min(torch.Tensor({ substitution, insertion, deletion }))
              end
          end
      end
      local errorRate = d[#target + 1][#prediction + 1] / #target
      return errorRate
end




function SequenceError:calculateCER(targetTranscript, predictTranscript)
    return self:sequenceErrorRate(targetTranscript, predictTranscript)
end

function SequenceError:calculateWER(targetTranscript, predictTranscript)
    --print ('-----------------------')
    --print (targetTranscript)
    --print (predictTranscript)
    -- convert to words before calculation
  --[[
  local targetWords = {}
  
    for word in targetTranscript:gmatch("%S+") do table.insert(targetWords, word) end
    local predictedWords = {}
    for word in predictTranscript:gmatch("%S+") do table.insert(predictedWords, word) end
    return self:sequenceErrorRate_WER(targetWords, predictedWords)
    --]]
    --local predictWords = predictTranscript:split('|')
    
    
    local err=0.0  
    if predictTranscript==nil or predictTranscript=='' then
      err = 1.0
    else
      local targetWords = targetTranscript:split('|')
      local predictWords = predictTranscript:split('|')
      --print (targetWords)
      --print (predictWords)
      --print ('---------------------------------')
      err = self:sequenceErrorRate_WER(targetWords,predictWords)
    end
    return err
end
