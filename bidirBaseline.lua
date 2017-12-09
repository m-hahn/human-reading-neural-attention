bidirBaseline = {}
bidirBaseline.__name = "bidirBaseline.lua"
print(bidirBaseline)



local function buildForwardModule()

  local model = nn.Sequential()

model:add(nn.SplitTable(1,1))
  local parallel1 = nn.ParallelTable()
  parallel1:add(nn.BlockGradientLayer(params.batch_size))
  parallel1:add(nn.BlockGradientLayer(params.batch_size))
  model:add(parallel1)

EMB_SIZE = 5

  if false then -- use own embeddings
     local parallel = nn.ParallelTable()
     parallel:add(nn.LookupTable(params.vocab_size,EMB_SIZE))
     parallel:add(nn.Reshape(1))
     model:add(parallel)
  else -- use reader embeddings
    local parallel = nn.ParallelTable()
    parallel:add(nn.LookupTable(params.vocab_size,params.embeddings_dimensionality))
    parallel:add(nn.Reshape(1))
    model:add(parallel)

    local parallel3 = nn.ParallelTable()
    parallel3:add(nn.BlockGradientLayer(params.batch_size,params.embeddings_dimensionality))
    parallel3:add(nn.Identity())
    model:add(parallel3)

    local parallel2 = nn.ParallelTable()
    parallel2:add(nn.Linear(params.embeddings_dimensionality, EMB_SIZE))
    parallel2:add(nn.Identity())
    model:add(parallel2)
  end


  model:add(nn.JoinTable(1,1))

  model:add(nn.LSTM(EMB_SIZE+1, params.baseline_rnn_size))

return model

end


local function buildBackwardModule()
  local model = nn.Sequential()
  model:add(nn.SplitTable(1,1))
  model:add(nn.SelectTable(1))

  model:add(nn.BlockGradientLayer(params.batch_size))

     EMB_SIZE = params.embeddings_dimensionality

     model:add(nn.LookupTable(params.vocab_size, params.embeddings_dimensionality))
  -- prevents word embeddings from being trained by the baseline
     model:add(nn.BlockGradientLayer(params.batch_size, params.embeddings_dimensionality))



  model:add(nn.LSTM(EMB_SIZE, params.baseline_rnn_size))
  return model
end

local function buildMergeModule()


  local model = nn.Sequential()

  local parallel = nn.ParallelTable()
  parallel:add(nn.View(params.baseline_rnn_size))
  parallel:add(nn.View(params.baseline_rnn_size))
  model:add(parallel)


  model:add(nn.JoinTable(1,1))
  model:add(nn.Linear(2*params.baseline_rnn_size, params.baseline_rnn_size))
  model:add(nn.Sigmoid())
  model:add(nn.Linear(params.baseline_rnn_size, 1))
  return model
end


function createBidirBaseline()
    -- forward module
    local forwardModule = buildForwardModule()
    local backwardModule = buildBackwardModule()
    local mergeModule = buildMergeModule()

    local sequencer = nn.BiSequencer(forwardModule, backwardModule, mergeModule)

    return sequencer:cuda()
end

-- takes the reader params to take over the word embeddings
function setupBidirBaseline(reader_network_params, SparamxB, SparamdxB)
        bidirBaseline.criterion_gradients = {}
        for i=1, params.seq_length do
            bidirBaseline.criterion_gradients[i] = torch.CudaTensor(params.batch_size,1)
        end


        bidir_baseline = createBidirBaseline()


        local bidir_baseline_params_table, bidir_baseline_gradparams_table = bidir_baseline:parameters()

     if SparamxB ~= nil then
        print("Getting baseline network from file")
        if true then
          for j=1, #SparamxB do
            if j ~= 19 then
               bidir_baseline_params_table[j]:set(SparamxB[j])
               bidir_baseline_gradparams_table[j]:set(SparamdxB[j])
            end
          end
        end
     bidir_baseline_params_table[19]:mul(0):add(SparamxB[19])
     end

        if not true then
          for j=1, #bidir_baseline_params_table do
            local weights = bidir_baseline_params_table[j]
            if weights:size(1) == params.vocab_size and weights:size(2) == params.embeddings_dimensionality then
              if true then          
                  weights:mul(0):add(reader_network_params[1]) 
              else
                 weights:set(reader_network_params[1])
              end
            end
          end
        end

       bidir_baseline_params, bidir_baseline_gradparams = bidir_baseline:getParameters()



       bidir_baseline_params_table, bidir_baseline_gradparams_table = nil

        baseline_criterion = nn.MSECriterion():cuda()
        
        joinTable = {}
        for j=1, params.seq_length do
            table.insert(joinTable,nn.JoinTable(2):cuda())
        end
end

