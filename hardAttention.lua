hardAttention = {}



function hardAttention.makeAttentionDecisions(i, inputTensor, surprisalValue, attInputTensor)
   if attInputTensor == nil then
      attInputTensor = inputTensor
   end
 
   local attendedInputTensor = torch.CudaTensor(params.batch_size):zero()
      if use_attention_network then 
                local results = attentionNetworks[i]:forward({attInputTensor, reader_c[i-1], surprisalValue})
                 if USE_BASELINE_NETWORK then
                     attention_scores[i] , baseline_scores[i] = unpack(results)
                 else
                     attention_scores[i] = results
                 end
      else
          if params.EXTERNAL_ATTENTION_SOURCE == 'fixed' then 
              attention_scores[i]:fill(FIXED_ATTENTION)
          elseif params.EXTERNAL_ATTENTION_SOURCE == 'WLEN' then
              for l=1, params.batch_size do
                   wlen = string.len(readDict.chars[inputTensor[l]])
                   if wlen > 3 then
                      attention_scores[i][l] = 1
                   else
                      attention_scores[i][l] = 0
                   end
              end
          elseif params.EXTERNAL_ATTENTION_SOURCE == 'NUMERICAL_FILE' then
              for l=1, params.batch_size do
                   value = numericalValues.numericalValuesImporter.values[l][i]
                   if NUMERICAL_VALUES_COLUMN == 3 then
                      if value > 0 then
                         attention_scores[i][l] = 1
                      elseif value > -1 then
                         attention_scores[i][l] = 0
                      else
                         attention_scores[i][l] = 0.7
                      end
                   elseif NUMERICAL_VALUES_COLUMN == 4 then
                      if value > 5.531714 then
                         attention_scores[i][l] = 1
                      elseif value > -1 then
                         attention_scores[i][l] = 0
                      else
                         attention_scores[i][l] = 0.62
                      end
                   elseif NUMERICAL_VALUES_COLUMN == 5 then
                      if value > 3 then
                         attention_scores[i][l] = 1
                      elseif value > 2 then
                         attention_scores[i][l] = 0.62
                      else
                         attention_scores[i][l] = 0
                      end
                   elseif NUMERICAL_VALUES_COLUMN == 9 then --surprisal
                      if value > 4.25  then
                         attention_scores[i][l] = 1
                      elseif value > -1 then
                         attention_scores[i][l] = 0
                      else
                         attention_scores[i][l] = 0.62
                      end
 
                   else
                      print(NUMERICAL_VALUES_COLUMN)
                      crash()
                   end
              end
          else
              print(params.EXTERNAL_ATTENTION_SOURCE)
              crash()
          end 
      end
   for item=1, params.batch_size do
      local dice = torch.uniform()
      if dice > attention_scores[i][item][1] then
         attention_decisions[i][item] = 0
         probabilityOfChoices[item] = probabilityOfChoices[item] * (1-attention_scores[i][item][1])
      else
         attention_decisions[i][item] = 1
         attendedInputTensor[item] = inputTensor[item]
         probabilityOfChoices[item] = probabilityOfChoices[item] * attention_scores[i][item][1]
      end
      totalAttentions[item] = totalAttentions[item] + attention_decisions[i][item]
   end
   return attendedInputTensor, probability
end

