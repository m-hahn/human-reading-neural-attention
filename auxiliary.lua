auxiliary = {__name = "auxiliary"}

auxiliary.QUESTION_FIRST = true



print(auxiliary)


------------------------------------
------------------------------------

function auxiliary.shallowCopyTable(table)
    local result = {}
    for key,value in pairs(table) do
      result[key] = value
    end
    return result
end


-- copy paste from https://coronalabs.com/blog/2014/09/30/tutorial-how-to-shuffle-table-items/
function auxiliary.shuffleTableInPlace(table)
    assert(table, "shuffleTable() expected a table, got nil" )
    local iterations = #table
    local j
    
    for i = iterations, 2, -1 do
        j = math.random(i)
        table[i], table[j] = table[j], table[i]
    end
end

function auxiliary.reverseTable(table,length)
   local result = {}
   for i=1,length do
     result[i] = table[length-i+1]
   end
   return result

end

function auxiliary.buildClones(seq_length,RNNs,core_network)
   for i=1,seq_length do
        print(i)
        RNNs[i] = clone_network(core_network)
   end
end

-- in place
function auxiliary.trimTable(table,length)
    local maxLength = #table
    for i = length+1,maxLength do
        table[i] = nil
    end
end


-- returns new table
function auxiliary.shortenTable(table,length)
  assert(#table >= length)
  local shortenedTable = {}
  for i=1,length do
    shortenedTable[i] = table[i]
  end
  return shortenedTable
end

function auxiliary.printMemory(value)
   print("MEMORY   "..value.."   "..(collectgarbage("count")/1024))
end


function auxiliary.write(value)
   io.write(value.."\t")
end

function auxiliary.prepareMomentum(paramdx)
  paramdx:mul(params.lr_momentum / (1-params.lr_momentum))
end

function auxiliary.clipGradients(paramdx)
   paramdx:clamp(-5,5)
end

function auxiliary.updateParametersWithMomentum(paramx,paramdx, learningRate)
      paramdx:mul((1-params.lr_momentum))

      learningRate = ((learningRate ~=nil) and learningRate) or params.lr
      
      if true then
        paramx:add(-1 * learningRate, paramdx)
      else
        paramx:add(paramdx:mul(- 1 * learningRate))
        paramdx:mul(1 / (- 1 * learningRate)) 
      end
end



function auxiliary.normalizeGradients(paramdx)
   local norm_dw = paramdx:norm()
      if norm_dw > params.max_grad_norm then
          local shrink_factor = params.max_grad_norm / norm_dw
          paramdx:mul(shrink_factor)
      end
end



function auxiliary.toUnaryTables(input)
  local output = {}
  for key,value in pairs(input) do
    output[key] = {value}
  end
  return output
end

function auxiliary.deepPrint(tableOfTensors, tensorToStringFunction)
  for key,value in pairs(tableOfTensors) do
     print(key.." : "..(tensorToStringFunction(value)))
  end
end

function getFromData(data, index, token)
              if #(data[index]) >= token then
                  return data[index][token]
              else
                  return 1
              end
end

function buildInputTensorsForSubcorpus(data, startIndex, endIndex, getFromFunction, maxLength, tensorType)
    local inputTensors = {}
    if tensorType == nil then
      tensorType = torch.CudaTensor --torch.LongTensor
    end
    for token=0, maxLength  do
      local inputTensor = tensorType(params.batch_size)
      -- the batch elements
      for index=startIndex,endIndex do
           if token==0 then
              inputTensor[index-startIndex+1] = 1
           else
              inputTensor[index-startIndex+1] = getFromFunction(data,index,token)
           end
      end
      inputTensors[token] = inputTensor
    end
    return inputTensors
end

function buildInputTensors(data, startIndex, endIndex)
    return buildInputTensorsForSubcorpus(data, startIndex, endIndex, getFromData, params.seq_length, torch.CudaTensor)
end



function auxiliary.buildSeparateInputTensorsQA(data, startIndex, endIndex, vectorOfLengths, maximalLengthOccurringInInput,maximalLengthOccurringInInputQuestion)

        if maximalLengthOccurringInInput ~= nil then
           maximalLengthOccurringInInput[1] = 0
        end

        if maximalLengthOccurringInInputQuestion ~= nil then
           maximalLengthOccurringInInputQuestion[1] = 0
        end

    assert(vectorOfLengths==nil,"not implemented here")

    local maxLengthText = params.seq_length
    local maxLengthQuestion = 50

    -- text = data[index].text/answer/question
    local inputTensorsText = {}
    local inputTensorsQuestion = {}

    for i = 0,maxLengthText do
        inputTensorsText[i] = torch.CudaTensor(params.batch_size):zero()
    end
    for i = 0,maxLengthQuestion do
        inputTensorsQuestion[i] = torch.CudaTensor(params.batch_size):zero()
    end


    -- question and text
   
    for i = startIndex,endIndex do
        text = data[i].text
        for j=1,#text do
                 if j > maxLengthText then
                    break
                 end
                 inputTensorsText[j][i] = text[j]
        end
        if maximalLengthOccurringInInput ~= nil then
           maximalLengthOccurringInInput[1] = math.min(maxLengthText, math.max(maximalLengthOccurringInInput[1], #text))
        end


        question = data[i].question
        for j=1,#question do
                 if j > maxLengthQuestion then
                    break
                 end
                 inputTensorsQuestion[j][i] = question[j]
        end
        if maximalLengthOccurringInInputQuestion ~= nil then
           maximalLengthOccurringInInputQuestion[1] = math.min(maxLengthQuestion, math.max(maximalLengthOccurringInInputQuestion[1], #question))
        end
    end

    return inputTensorsText, inputTensorsQuestion
end




--//function auxiliary.buildInputTensorsQA(data, startIndex, endIndex, vectorOfLengths, maximalLengthOccurringInInput)
--//
--//        if maximalLengthOccurringInInput ~= nil then
--//           maximalLengthOccurringInInput[1] = 0
--//        end
--//
--//
--//    local maxLength = params.seq_length
--//    -- text = data[index].text/answer/question
--//    local inputTensors = {}
--//    for i = 0,maxLength do
--//        inputTensors[i] = torch.CudaTensor(params.batch_size):zero()
--//    end
--//
--//    -- question and text
--//   
--//    for i = startIndex,endIndex do
--//        question = data[i].question
--//        text = data[i].text
--//        if auxiliary.QUESTION_FIRST then 
--//              for j=1,#question do
--//                 if j > maxLength then
--//                    break
--//                 end
--//                 inputTensors[j][i] = question[j]
--//              end
--//              if #question+1 <= maxLength then
--//                 inputTensors[#question+1][i] = params.vocab_size-1 -- 9512 --some special character
--//              end
--//              for j=1,#text do
--//                 if j + #question +1 > maxLength then
--//                    break
--//                 end
--//                 inputTensors[j + #question+1][i] = text[j]
--//              end
--//        else
--//              local lengthOfTextSegment = math.max(0,math.min(#text, maxLength - #question - 1))
--//              local lengthOfQuestionSegment = math.max(0,math.min(#question, maxLength - lengthOfTextSegment -1))
--//              inputTensors[lengthOfTextSegment+1][i] = params.vocab_size-1
--//              for j=1,lengthOfQuestionSegment do
--//                 inputTensors[lengthOfTextSegment+1+j][i] = question[j]
--//              end
--//
--//              for j=1,lengthOfTextSegment do
--//                 inputTensors[j][i] = text[j]
--//              end
--//        end
--//        if vectorOfLengths ~= nil then
--//           vectorOfLengths[i] = math.min(maxLength, #question+1+#text)
--//        end
--//        if maximalLengthOccurringInInput ~= nil then
--//           maximalLengthOccurringInInput[1] = math.min(maxLength, math.max(maximalLengthOccurringInInput[1], #question+1+#text))
--//        end
--//    end
--//
--//-- when we use Sequencer instead of hard-coded unrolling, we need to make this shorter
--//if neatQA.fullModel ~= nil then
--//    for i = maximalLengthOccurringInInput[1]+1,maxLength do
--//        inputTensors[i] = nil
--//    end
--//end
--//    return inputTensors
--//end




---------------------------------------
---------------------------------------

function perturbInputTensor(inputTensor)
   local inputTensor = inputTensor:clone()
   for item=1, params.batch_size do
     if torch.uniform() > 0.9 then
       inputTensor[item] = math.floor(torch.uniform() * (params.vocab_size-1) + 1)
     end
   end
   return inputTensor
end

require('hardAttention')


---------------------------------------
---------------------------------------


function retrieveSurprisalValue(readerSurpValues, inputTensor)
    local surprisals = torch.CudaTensor(params.batch_size,1)
    for i=1, params.batch_size do
       surprisals[i][1] = readerSurpValues[i][inputTensor[i]]
    end
    return surprisals
end





function checkBackprop(data)
   params.max_grad_norm = 100000000 --to prevent renormalization of the gradients

   local loss, _ = fp(data, 1, params.batch_size)

   --paramxRQ:zero()
   local H = 0.005
   for i = 1, paramxRT:size(1) do
       print("------------------------------A "..i)
--      for j = 1,paramdxA[i]:dim() do

          paramxRT[i] = paramxRT[i] + H

          local lossNew, _ = fp(data, 1, params.batch_size)
          local deriv = - torch.sum(lossNew - loss) * 1/H
          print(torch.sum(lossNew - loss))
   
          print((deriv - paramdxRT[i]).." :: "..deriv.."  "..paramdxRT[i])
          paramxRT[i] = paramxRT[i] - H



          paramxRT[i] = paramxRT[i] + H

          local lossNew, _ = fp(data, 1, params.batch_size)
          local deriv = - torch.sum(lossNew - loss) * 1/H
          print(torch.sum(lossNew - loss))
   
          print((deriv - paramdxRT[i]).." :: "..deriv.."  "..paramdxRT[i])
          paramxRT[i] = paramxRT[i] - H
   end


end



function buildGradientsOfProbOutputs(dsAThird, corpus, startIndex, endIndex, tokenIndex)
      for index=startIndex,endIndex do
           if tokenIndex==0 then
               dsAThird[index - startIndex + 1][1] = 1
           else
               dsAThird[index - startIndex + 1][getFromData(corpus,index,tokenIndex)] = 1
           end
      end
end


--[[ taken from Jianpeng's code at https://github.com/cheng6076/SNLI-attention/blob/e296ecd12d57529bcc5590e9c35b4bd7978157d5/util/misc.lua ]]
function auxiliary:clone_list(tensor_list, zero_too)
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

--[[ taken from Jianpeng's code at https://github.com/cheng6076/SNLI-attention/blob/e296ecd12d57529bcc5590e9c35b4bd7978157d5/util/misc.lua ]]
function auxiliary:narrow_list(tensor_list, first, last, zero_too)
    local out = {}
    first = first or 1
    last = last or #tensor_list
    for i = first, last do
        if zero_too then
            table.insert(out, tensor_list[i]:clone():zero())
        else 
            table.insert(out, tensor_list[i])
        end
    end
    return out
end


