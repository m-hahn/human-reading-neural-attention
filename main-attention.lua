

BASE_DIRECTORY = "models/"


BlockGradientLayer = require('nn.BlockGradientLayer')
require('globalForExpOutput')
--require('phono')
--require('linearization')
require('rnn')
require('bidirBaseline')
require('cunn')
require('nngraph')
require('base')
require('setParameters')
require('readDict')
require('datasets')
require 'lfs'
require('storeAnnotation')



assert(not ((not DOING_EVALUATION_OUTPUT) and (not DO_TRAINING)))

require('readChunks')
require('numericalValues')
require('readFiles')
require('lstm')
require('attention')
require('auxiliary')
require('autoencoding')
require('combined')
--require('langmod')



assert(params.TASK == 'combined' or params.TASK == 'langmod')


local function setup()
      return autoencoding.setupAutoencoding()
end



function reset_ds()
    model.dsR[1] = torch.zeros(params.batch_size,params.rnn_size):cuda() 
    model.dsR[2] = torch.zeros(params.batch_size,params.rnn_size):cuda() 
    model.dsR[3] = torch.zeros(params.batch_size,params.vocab_size):cuda()


    model.dsA[1] = torch.zeros(params.batch_size,params.rnn_size):cuda() 
    model.dsA[2] = torch.zeros(params.batch_size,params.rnn_size):cuda() 
    model.dsA[3] = torch.zeros(params.batch_size,params.vocab_size):cuda() 
end


local function fp(corpus, startIndex, endIndex)
      return autoencoding.fpAutoencoding(corpus, startIndex, endIndex)
end



local function bp(corpus, startIndex, endIndex)
      if USE_BASELINE_NETWORK then
         return combined.bpCombined(corpus, startIndex, endIndex)
      else
         return combined.bpCombinedNoBaselineNetwork(corpus, startIndex, endIndex)
      end
end

require('getParams')
require('nn.UniformLayer')
require('nn.ScalarMult')
require('nn.BlockGradientLayer')


local function tryReadParam(func)
  if not pcall(func) then
  print("ERROR ")
  print(func)
  end
end


local function main()
--  g_init_gpu({params.gpu_number})

  PRINTING_PERIOD = 51
  if DOING_EVALUATION_OUTPUT then
     PRINTING_PERIOD = 1
  end
  
  readDict.readDictionary()

  if params.TASK == 'neat-qa' then
     readDict.createNumbersToEntityIDsIndex()
  end

  print("Network parameters:")
  print(params)

  print("setup")
  setup()
  print("setup done")

  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")
  local numberOfWords = 0
  local counter = 0

  tryReadParam(getLearningRateFromFile)
  tryReadParam(getAttentionFromFile)
  tryReadParam(getAttentionLearningRateFromFile)
  if params.TASK == 'combined-soft' or params.TASK == "combined-q" then
      tryReadParam(getTotalAttentionWeightFromFile)
  end

  print(FIXED_ATTENTION)
  print(params.lr)
  print(params.lr_att)

  for epoch = 1,EPOCHS_NUMBER do
     if (not DO_TRAINING) and epoch > 1 then
       print("BREAK. Not doing training, so only doing the first epoch.")
       break
     end

     readChunks.resetFileIterator()
     epochCounter = epoch

     while readChunks.hasNextFile() do

        if ( PERCENTAGE_OF_DATA < (100.0 * (readChunks.corpusReading.currentFile+0.0) / #readChunks.files)) then
          print("Breaking at specified percentage of data")
          break
        end
 
        for l = 1, params.batch_size do
             readChunks.corpus[l] = readChunks.readNextChunkForBatchItem(l)
             if params.INCLUDE_NUMERICAL_VALUES then
               numericalValues.getNumericalValuesForBatchItem(l)
             end
        end
      


      local perp, actor_output = fp(readChunks.corpus, 1, params.batch_size)
      numberOfWords = numberOfWords + params.batch_size * params.seq_length

      if MAKE_SKIPPING_STATISTICS then
           updateSkippingStatistics(readChunks.corpus)
      end

      if STORE_ATTENTION_ANNOTATION then
          storeAttentionAnnotation(readChunks.corpus)
      end

      if WRITE_SURPRISAL_SCORES then
          storeSurprisalScores(readChunks.corpus)
      end


      if DOING_EVALUATION_OUTPUT then
          for  l=1, params.batch_size do
              if readChunks.corpus[l][1] == 2 and readChunks.corpus[l][2] == 2 and readChunks.corpus[l][3] == 2 then
                 print(readChunks.corpus[l])
              else
                 evaluationAccumulators.reconstruction_loglikelihood = evaluationAccumulators.reconstruction_loglikelihood + perp[l]
                 evaluationAccumulators.lm_loglikelihood = evaluationAccumulators.lm_loglikelihood + nll_reader[l]
                 evaluationAccumulators.numberOfTokens = evaluationAccumulators.numberOfTokens + params.seq_length
              end
          end
      end

      counter = counter + 1

      if  counter % 100 == 0 and TASK == 'qa' then
         qaCorrect = 0
         qaIncorrect = 0
      end


      if counter % PRINTING_PERIOD == 0 then
         print('WORDS '..numberOfWords..'  EPOCH '..epoch..'  '..(100.0 * (readChunks.corpusReading.currentFile+0.0) / #readChunks.files))
         print("WORDS/SEC   "..numberOfWords / torch.toc(start_time))
         local since_beginning = g_d(torch.toc(beginning_time) / 60)


         combined.printStuffForCombined(perp, actor_output, since_beginning, epoch, numberOfWords)

         tryReadParam(getLearningRateFromFile)
         tryReadParam(getAttentionFromFile)
         tryReadParam(getAttentionLearningRateFromFile)

         print(FIXED_ATTENTION)
         print(params.lr)
         print(params.lr_att)
      end

      if DO_TRAINING then
         bp(readChunks.corpus, 1, params.batch_size)
      end
      if counter % 33 == 0 then
        cutorch.synchronize()
        collectgarbage()
      end

      if DO_TRAINING and counter % PRINT_MODEL_PERIOD == 0 then
        print("WRITING MODEL...")
        local modelsArray
        local uR, udR = readerRNNs[1]:parameters()
        local uA, udA = actorRNNs[1]:parameters()
        local uRA, udRA = attentionNetworks[1]:parameters()
        modelsArray = {params,(numberOfWords/params.seq_length),uR, udR, uA, udA, uRA, udRA, reader_c[0], reader_h[0]}
        if USE_BIDIR_BASELINE and bidir_baseline ~= nil then
           local uB, udB = bidir_baseline:parameters()
           table.insert(modelsArray, uB)
           table.insert(modelsArray, udB)
        end
        if modelsArray ~= nil then
           torch.save(BASE_DIRECTORY..'/model-'..experimentNameOut, modelsArray, "binary")
        end
      end


      if DOING_EVALUATION_OUTPUT then
            print(evaluationAccumulators.reconstruction_loglikelihood.."&\n"..evaluationAccumulators.lm_loglikelihood.."\n") 
      end
    end
  end
  print("Training is over.")

    if DOING_EVALUATION_OUTPUT then
             PERP_ANNOTATION_FILE = DATASET_DIR.."/annotation/perp-"..experimentNameOut..".txt"

             local fileOut = io.open(PERP_ANNOTATION_FILE, "w")
             print(PERP_ANNOTATION_FILE)
             tokenCountForLM = 49/50 * evaluationAccumulators.numberOfTokens
             fileOut:write("REC "..(evaluationAccumulators.reconstruction_loglikelihood) .."\n".."LM  "..(evaluationAccumulators.lm_loglikelihood).."\n".."REC "..(evaluationAccumulators.reconstruction_loglikelihood/evaluationAccumulators.numberOfTokens) .."\n".."LM  "..(evaluationAccumulators.lm_loglikelihood/tokenCountForLM) .."\n".."REC "..math.exp(evaluationAccumulators.reconstruction_loglikelihood/evaluationAccumulators.numberOfTokens) .."\n".."LM  "..math.exp(evaluationAccumulators.lm_loglikelihood/tokenCountForLM).."\n") 
             fileOut:close()
      end

  fileStats:close()
end

if (not OVERWRITE_MODEL) and  (lfs.attributes(BASE_DIRECTORY..'/model-'..experimentNameOut) ~= nil) then
   print("MODEL EXISTS, ABORTING")
else
  if (lfs.attributes(BASE_DIRECTORY..'/model-'..experimentNameOut) ~= nil) and OVERWRITE_MODEL then
     print("OVERWRITE MODEL")
  end
   main()
end
