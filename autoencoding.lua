autoencoding = {}
autoencoding.__name = "autoencoding"

autoencoding.USE_PRETRAINED_EMBEDDINGS = false
assert(not autoencoding.USE_PRETRAINED_EMBEDDINGS)

print(autoencoding)





function autoencoding.create_network(withOutput, doZeroMaskingOnLookupTable, inputDropout)
  local x                = nn.Identity()()
  local prev_c           = nn.Identity()()
  local prev_h           = nn.Identity()()
  local i

  if doZeroMaskingOnLookupTable then
    i = nn.LookupTableMaskZero(params.vocab_size,params.embeddings_dimensionality)(x)
  else
    i = nn.LookupTable(params.vocab_size,params.embeddings_dimensionality)(x)
  end

  if inputDropout == true then
    i = nn.Dropout(0.2)(i)
  end

  local next_s           = {}
  local next_c, next_h = lstm.lstm(i, prev_c, prev_h, params.embeddings_dimensionality)
  local module
  if withOutput  then
        local h2y              = nn.Linear(params.rnn_size, params.vocab_size)(next_c)
        local output = nn.MulConstant(-1)(nn.LogSoftMax()(h2y))
      module = nn.gModule({x, prev_c, prev_h},
                                      {next_c, next_h, output})
  else
      module = nn.gModule({x, prev_c, prev_h},
                                      {next_c, next_h})
  end
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

function autoencoding.setupAutoencoding()
  print("Creating a RNN LSTM network.")


  -- initialize data structures
  model.sR = {}
  model.dsR = {}
  model.dsA = {}
  model.start_sR = {}
  for j = 0, params.seq_length do
    model.sR[j] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.dsR[1] = transfer_data(torch.zeros(params.rnn_size))
  model.dsR[2] = transfer_data(torch.zeros(params.rnn_size))

  model.dsA[1] = transfer_data(torch.zeros(params.rnn_size))
  model.dsA[2] = transfer_data(torch.zeros(params.rnn_size))
 model.dsA[3] = transfer_data(torch.zeros(params.rnn_size)) -- NOTE actually will later have different size



  reader_c ={}
  reader_h = {}

  actor_c ={[0] = torch.CudaTensor(params.rnn_size)}
  actor_h = {[0] = torch.CudaTensor(params.rnn_size)}

  reader_c[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()
  reader_h[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()

  if params.TASK == 'combined' then
     reader_output = {}
     surprisal_values = {[1] = transfer_data(torch.zeros(params.batch_size,1))}
  end


  attention_decisions = {}
  attention_scores = {}
  baseline_scores = {}
  for i=1, params.seq_length do
     attention_decisions[i] = torch.CudaTensor(params.batch_size)
     attention_scores[i] = torch.CudaTensor(params.batch_size,1)
     baseline_scores[i] = torch.CudaTensor(params.batch_size,1)
  end

  probabilityOfChoices = torch.FloatTensor(params.batch_size)
  totalAttentions = torch.FloatTensor(params.batch_size)
   nll = torch.FloatTensor(params.batch_size)

   if params.TASK == 'combined' then
      nll_reader = torch.FloatTensor(params.batch_size)
   end

  attention_inputTensors = {}
  if USE_PREDICTION_FOR_ATTENTION then
     for i=1, params.seq_length do
        attention_inputTensors[i] = torch.CudaTensor(params.batch_size)
     end
  end




  ones = transfer_data(torch.ones(params.batch_size))
  rewardBaseline = 0

   variance_average = 100
   recurrent_variance_average = 100
   if not LOAD then
     -- READER
     local reader_core_network
     reader_core_network = autoencoding.create_network(true,false,true)

     paramxR, paramdxR = reader_core_network:getParameters()
     readerRNNs = {}
     for i=1,params.seq_length do
        readerRNNs[i] = clone_network(reader_core_network)
     end

     -- ACTOR
     local actor_core_network = autoencoding.create_network(true)
     paramxA, paramdxA = actor_core_network:getParameters()
     actorRNNs = {}
     for i=1,params.seq_length do
        actorRNNs[i] = clone_network(actor_core_network)
     end

     -- ATTENTION
     local attentionNetwork = attention.createAttentionNetwork()
     paramxRA, paramdxRA = attentionNetwork:getParameters()
     attentionNetworks = {}
     for i=1,params.seq_length do
        attentionNetworks[i] = clone_network(attentionNetwork)
     end
  elseif true then

     print("LOADING MODEL AT "..BASE_DIRECTORY.."/model-"..fileToBeLoaded)

     local params2, sentencesRead, SparamxR, SparamdxR, SparamxA, SparamdxA, SparamxRA, SparamdxRA, readerCStart, readerHStart, SparamxB, SparamdxB = unpack(torch.load(BASE_DIRECTORY.."/model-"..fileToBeLoaded, "binary"))

    if SparamxB == nil and USE_BIDIR_BASELINE and DO_TRAINING and IS_CONTINUING_ATTENTION then
        print("no baseline in saved file")
        assert(false)
    end

    print(params2)


     local reader_core_network
     reader_core_network = autoencoding.create_network(true,false,true)

     -- LOAD PARAMETERS
     reader_network_params, reader_network_gradparams = reader_core_network:parameters()
     for j=1, #SparamxR do
           reader_network_params[j]:set(SparamxR[j])
           reader_network_gradparams[j]:set(SparamxR[j])
     end
     paramxR, paramdxR = reader_core_network:getParameters()
     reader_network_params, reader_network_gradparams = reader_core_network:parameters()

     -- CLONE
     readerRNNs = {}

     for i=1,params.seq_length do
        readerRNNs[i] = clone_network(reader_core_network)
     end

     -- ACTOR
     local actor_core_network = autoencoding.create_network(true)
     actor_network_params, actor_network_gradparams = actor_core_network:parameters()
     for j=1, #SparamxA do
           actor_network_params[j]:set(SparamxA[j])
           actor_network_gradparams[j]:set(SparamdxA[j])
     end

     paramxA, paramdxA = actor_core_network:getParameters()

     actorRNNs = {}

     for i=1,params.seq_length do
        actorRNNs[i] = clone_network(actor_core_network)
     end

     -- ATTENTION

     local attentionNetwork = attention.createAttentionNetwork()
     att_network_params, network_gradparams = attentionNetwork:parameters()
     if params.ATTENTION_WITH_EMBEDDINGS then
        if not IS_CONTINUING_ATTENTION then
           att_network_params[1]:set(reader_network_params[1])
           print("Using embeddings from the reader")
        else
           print("Not using embeddings from the reader because continuing attention")
        end
     end

     if USE_BIDIR_BASELINE and DO_TRAINING then
          setupBidirBaseline(reader_network_params, SparamxB, SparamdxB)
     end

     if IS_CONTINUING_ATTENTION then
         network_params, network_gradparams = attentionNetwork:parameters()
         for j=1, #SparamxRA do
            network_params[j]:set(SparamxRA[j])
            network_gradparams[j]:set(SparamdxRA[j])
         end
         print("Got attention network from file")
     else
         print("NOTE am not using the attention network from the file")
     end

     paramxRA, paramdxRA = attentionNetwork:getParameters()
     attentionNetworks = {}
     for i=1,params.seq_length do
        attentionNetworks[i] = clone_network(attentionNetwork)
     end


     paramdxRA:zero()

     print("Sequences read by model "..sentencesRead)

     reader_c[0] = readerCStart
     reader_h[0] = readerHStart
   end
end



function autoencoding.fpAutoencoding(corpus, startIndex, endIndex)
  probabilityOfChoices:fill(1)
  totalAttentions:fill(params.ATTENTION_VALUES_BASELINE) 
  local inputTensors = buildInputTensors(corpus, startIndex, endIndex)

  if params.TASK == 'combined' then
     nll_reader:zero()
     reader_output = {}
  end

  for i=1, params.seq_length do
     local inputTensor = inputTensors[i]

     -- make attention decisions
      if i>1 then
         surprisal_values[i] = retrieveSurprisalValue(reader_output[i-1], inputTensors[i])
      end
      if (not USE_PREDICTION_FOR_ATTENTION) and attention_inputTensors[i] ~= nil then
        crash()
      elseif PREDICTION_FOR_ATTENTION and attention_inputTensors[i] == nil then
        crash()
      end
      local attendedInputTensor, probability = hardAttention.makeAttentionDecisions(i, inputTensor, surprisal_values[i], attention_inputTensors[i])
      reader_c[i], reader_h[i], reader_output[i] = unpack(readerRNNs[i]:forward({attendedInputTensor, reader_c[i-1], reader_h[i-1]}))
      if i < params.seq_length then

         for item=1, params.batch_size do
            local lm_loss_for_item =  reader_output[i][item][getFromData(corpus,startIndex+ item - 1,i+1)] 
            nll_reader[item] = nll_reader[item] + lm_loss_for_item
         end
      end
  end

  actor_c[0] = reader_c[params.seq_length] 
  actor_h[0] = reader_h[params.seq_length] 


  nll:zero()
  actor_output = {}
  
  local inputTensor
  for i=1, params.seq_length do
     inputTensor = inputTensors[i-1]
     actor_c[i], actor_h[i], actor_output[i] = unpack(actorRNNs[i]:forward({inputTensor, actor_c[i-1], actor_h[i-1]}))
     for item=1, params.batch_size do
        local rec_loss_for_item =  actor_output[i][item][getFromData(corpus,startIndex+ item - 1,i)] 
        nll[item] = nll[item] + rec_loss_for_item 
     end
  end
  return nll, actor_output
end



function autoencoding.bpAutoencoding(corpus, startIndex, endIndex)

  paramdxR:zero()
  paramdxA:zero()

  -- MOMENTUM
  paramdxRA:mul(params.lr_momentum / (1-params.lr_momentum))
  reset_ds()
  buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, params.seq_length)
  local inputTensors = buildInputTensors(corpus, startIndex, endIndex)
  if params.lr > 0 or train_autoencoding then
      for i = params.seq_length, 1, -1 do
          inputTensor = inputTensors[i-1]
          local prior_c = actor_c[i-1]
          local prior_h = actor_h[i-1]
          local derr = transfer_data(torch.ones(1))
          local tmp = actorRNNs[i]:backward({inputTensor, prior_c, prior_h},
                                       model.dsA)
          model.dsA[1]:copy(tmp[2])
          model.dsA[2]:copy(tmp[3])
          model.dsA[3]:zero()

          buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, i-1)
          cutorch.synchronize()
      end

      model.dsR[1]:copy(model.dsA[1])
      model.dsR[2]:copy(model.dsA[2])

      -- do it for reader network
      for i = params.seq_length, 1, -1 do
          inputTensor= torch.cmul(inputTensors[i], attention_decisions[i])
          local prior_c = reader_c[i-1]
          local prior_h = reader_h[i-1]
          local derr = transfer_data(torch.ones(1))
          local tmp = readerRNNs[i]:backward({inputTensor, prior_c, prior_h},
                                        model.dsR)
          model.dsR[1]:copy(tmp[2])
          model.dsR[2]:copy(tmp[3])
          cutorch.synchronize()
      end

      model.norm_dwR = paramdxR:norm()
      if model.norm_dwR > params.max_grad_norm then
          local shrink_factor = params.max_grad_norm / model.norm_dwR
          paramdxR:mul(shrink_factor)
      end

      model.norm_dwA = paramdxA:norm()
      if model.norm_dwA > params.max_grad_norm then
          local shrink_factor = params.max_grad_norm / model.norm_dwA
          paramdxA:mul(shrink_factor)
      end

      momentum = 0.8

      paramxR:add(paramdxR:mul(-params.lr))
      paramxA:add(paramdxA:mul(-params.lr))

  end

  if train_attention_network then
     local reward = torch.add(nll, params.TOTAL_ATTENTIONS_WEIGHT,totalAttentions) -- gives the reward for each batch item
     local rewardDifference = reward:cuda():add(-rewardBaseline, ones)
     rewardBaseline = 0.8 * rewardBaseline + 0.2 * torch.sum(reward) * 1/params.batch_size
     rewardDifference:mul(REWARD_DIFFERENCE_SCALING)
     for i = params.seq_length, 1, -1 do
        local whatToMultiplyToTheFinalDerivative = torch.CudaTensor(params.batch_size)
        local attentionEntropyFactor =  torch.CudaTensor(params.batch_size)
        for j=1,params.batch_size do
          attentionEntropyFactor[j] = params.ENTROPY_PENALTY * (math.log(attention_scores[i][j][1]) - math.log(1 - attention_scores[i][j][1]))
           if attention_decisions[i][j] == 0 then
               whatToMultiplyToTheFinalDerivative[j] = -1 / (1 - attention_scores[i][j][1])
           else
               whatToMultiplyToTheFinalDerivative[j] = 1 / (attention_scores[i][j][1])
           end
        end
        local factorsForTheDerivatives =  rewardDifference:clone():cmul(whatToMultiplyToTheFinalDerivative)
        factorsForTheDerivatives:add(attentionEntropyFactor)
        local tmp = attentionNetworks[i]:backward({inputTensors[i], reader_c[i-1]},factorsForTheDerivatives)
     end
     local norm_dwRA = paramdxRA:norm()
     if norm_dwRA > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / norm_dwRA
        paramdxRA:mul(shrink_factor)
     end
     assert(norm_dwRA == norm_dwRA)

     -- MOMENTUM
     paramdxRA:mul((1-params.lr_momentum))
     paramxRA:add(paramdxRA:mul(- 1 * params.lr_att))
     paramdxRA:mul(1 / (- 1 * params.lr_att)) -- is this really better than cloning before multiplying?
  end
end




