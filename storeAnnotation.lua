storeAnnotation = {}


function removeAnnotation(annotationDir)
  for file in lfs.dir(annotationDir) do
    os.remove(annotationDir.."/"..file)
  end
end


STORE_ATTENTION_ANNOTATION = false
if DOING_EVALUATION_OUTPUT and (params.TASK == "autoencoding" or params.TASK == "combined" or params.TASK == "combined-soft" or params.TASK == "neat-qa") then
  STORE_ATTENTION_ANNOTATION = true
end
ATTENTION_ANNOTATION_DIR = DATASET_DIR.."/annotation/att-"..arg[15].."/"
if STORE_ATTENTION_ANNOTATION then
   if DO_TRAINING then
      crash()
   end
   lfs.mkdir(ATTENTION_ANNOTATION_DIR)
   print("STORING ATTENTION ANNOTATION IN "..ATTENTION_ANNOTATION_DIR)
   removeAnnotation(ATTENTION_ANNOTATION_DIR)
end

WRITE_SURPRISAL_SCORES = false
if DOING_EVALUATION_OUTPUT and (params.TASK == "langmod" or params.TASK == "combined" or params.TASK == "combined-soft") then
  WRITE_SURPRISAL_SCORES = true
end
SURPRISAL_ANNOTATION_DIR = DATASET_DIR.."/annotation/surp-"..arg[15].."/"
if WRITE_SURPRISAL_SCORES then
   if DO_TRAINING then
      crash()
   end
    lfs.mkdir(SURPRISAL_ANNOTATION_DIR)
    print("WRITING SURPRISAL SCORES IN "..SURPRISAL_ANNOTATION_DIR)
   removeAnnotation(SURPRISAL_ANNOTATION_DIR)
end

MAKE_SKIPPING_STATISTICS = false
if MAKE_SKIPPING_STATISTICS then
   if DO_TRAINING then
      crash()
   end
    total_encountered_statistics = {}
    has_seen_in_skipping_statistics = {}
    print("WRITING SKIPPING SCORES")
end






function printSkippingStatistics(corpus)
   local skippingStatisticsTable = {}
   for i=1,#has_seen_in_skipping_statistics do
     if has_seen_in_skipping_statistics[i] ~= nil then
        if total_encountered_statistics[has_seen_in_skipping_statistics[i]][1] > 100 then
           table.insert(skippingStatisticsTable,{readDict.chars[i], ((100.0 * total_encountered_statistics[has_seen_in_skipping_statistics[i]][2]) / total_encountered_statistics[has_seen_in_skipping_statistics[i]][1])})
        end
     end
   end
   table.sort(skippingStatisticsTable,  function(a1, a2)
                                                  return (a1[2] < a2[2])
                                              end
             )
   for i=1, #skippingStatisticsTable do
        print(skippingStatisticsTable[i][1].."  "..skippingStatisticsTable[i][2])
   end
end


 
function updateSkippingStatistics(corpus)
  for i=1, params.seq_length do
    for item = 1, params.batch_size do
      local token = getFromData(corpus,item,i)
      if has_seen_in_skipping_statistics[token] == nil then
           has_seen_in_skipping_statistics[token] = (#total_encountered_statistics)+1
           table.insert(total_encountered_statistics, {1,0})
      else
           total_encountered_statistics[has_seen_in_skipping_statistics[token]][1] = total_encountered_statistics[has_seen_in_skipping_statistics[token]][1] +1
      end
      if attention_decisions[i][item] == 0 then
           total_encountered_statistics[has_seen_in_skipping_statistics[token]][2] = total_encountered_statistics[has_seen_in_skipping_statistics[token]][2] + 1
      end
    end
  end
end

function storeAttentionAnnotationQA(corpus)
   assert(params.TASK == "neat-qa")
   for batchIndex = 1, params.batch_size do
       local fileNameForBatchItem = readChunks.files[readChunks.corpusReading.currentFile-params.batch_size+batchIndex]
       print(fileNameForBatchItem)
       if(fileNameForBatchItem == nil) then
         print("Warning: Batch item "..batchIndex.." does not belong to a file.")
         goto continue
       end
       local outputFileNameForBatchItem = ATTENTION_ANNOTATION_DIR.."/"..fileNameForBatchItem..".att.anno"
       local fileOut = io.open(outputFileNameForBatchItem, "a")
       print("11214  "..outputFileNameForBatchItem)
       fileOut:write("++\n")
       for j=1, neatQA.maximalLengthOccurringInInput[1] do
          local inputWordNumber = neatQA.inputTensors[j][batchIndex]
          if inputWordNumber == 0 then
             goto continueInner
          end
          local toPrint = j.."  "..inputWordNumber.."  "..readDict.chars[inputWordNumber].."  "..attention_decisions[j][batchIndex][1].."  "..attention_scores[j][batchIndex][1].."\n"
          fileOut:write(toPrint)
          ::continueInner::
       end
       fileOut:close()
       ::continue::
   end
end



function storeAttentionAnnotation(corpus)
   assert(params.TASK ~= "neat-qa")
   for batchIndex = 1, params.batch_size do
       local fileOut = io.open(ATTENTION_ANNOTATION_DIR.."/"..readChunks.files[readChunks.corpusReading.locationsOfLastBatchChunks[batchIndex].file]..".att.anno", "a")
       print(ATTENTION_ANNOTATION_DIR.."/"..readChunks.files[readChunks.corpusReading.locationsOfLastBatchChunks[batchIndex].file]..".att.anno")
       fileOut:write("++\n")
       for j=1, params.seq_length do
          local toPrint
          if params.TASK == "combined" then
              toPrint = (readChunks.corpusReading.locationsOfLastBatchChunks[batchIndex].offset + j - 1).."  "..getFromData(corpus,batchIndex,j).."  "..attention_decisions[j][batchIndex].."  "..attention_scores[j][batchIndex][1].."\n"
          elseif params.TASK == "combined-soft" then
              toPrint = (readChunks.corpusReading.locationsOfLastBatchChunks[batchIndex].offset + j - 1).."  "..getFromData(corpus,batchIndex,j).."  ".."1".."  "..attention_scores[j][batchIndex][1].."\n"
          else
              crash()
          end
          fileOut:write(toPrint)
       end
       fileOut:close()
   end
end



function storeSurprisalScores(corpus)
   for batchIndex = 1, params.batch_size do
       if(readChunks.files[readChunks.corpusReading.locationsOfLastBatchChunks[batchIndex].file] == nil) then
         print("Warning: Batch item does not belong to a file.")
         goto continue
       end
       local fileOut = io.open(SURPRISAL_ANNOTATION_DIR.."/"..readChunks.files[readChunks.corpusReading.locationsOfLastBatchChunks[batchIndex].file]..".surp.anno", "a")
       if torch.uniform() > 0.99 then
            print(SURPRISAL_ANNOTATION_DIR.."/"..readChunks.files[readChunks.corpusReading.locationsOfLastBatchChunks[batchIndex].file]..".surp.anno")
       end



       fileOut:write("++\n")
       for j=1, params.seq_length do
          local surprisal
          if params.TASK == 'combined' or params.TASK == "combined-soft" then
             if j==1 then
               surprisal = 100
             else
                surprisal = reader_output[j-1][batchIndex][getFromData(corpus,batchIndex,j)]
             end
          elseif params.TASK == 'langmod' then
             surprisal = actor_output[j][batchIndex][getFromData(corpus,batchIndex,j)]
          else
             crash()
          end

          fileOut:write((readChunks.corpusReading.locationsOfLastBatchChunks[batchIndex].offset + j - 1).."  "..getFromData(corpus,batchIndex,j).."  "..(surprisal).."\n")
       end
       fileOut:close()
       ::continue::
   end
end
