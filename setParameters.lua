
-- 1 PROCESS ID
-- 2 doing evaluation output?
-- 3 LOAD
-- 4 batch size
-- 5 seq length
-- 6 rnn size
-- 7 vocab size
-- 8 total attentions weight
-- 9 use attention network?
-- arg[10] is learning rate
-- embeddings_dimensionality = arg[11]
-- arg[12] is lr_att
-- arg[13] is minus ATTENTION_VALUES_BASELINE
-- arg[14] is whether it is reloaded from a previous reloaded version (in which case the attention network will be taken over)
-- arg[15] the file to be loaded (tacitly assuming that dimensions will match)
-- arg[16] is a suffix to the files that are written
-- arg[17] the task
-- arg[18] whether it should do training at all
-- arg[19] the number of the corpus
-- arg[20] ATTENTION_WITH_EMBEDDINGS
-- arg[21] ENTROPY WEIGHT
-- arg[22] ABLATION
-- arg[23] overwrite model? (true or false)
-- arg[24] external attention source


--------------------------
--------------------------

  USE_PREDICTION_FOR_ATTENTION = false
  USE_BIDIR_BASELINE = true 

USE_BASELINE_NETWORK = false


PRINT_MODEL_PERIOD = 500 

--------------------------
--------------------------

NLL_TO_CHANGE_ATTENTION = 0.00000001

meanNLL = 10000
meanTotalAtt = 0

--------------------------
--------------------------

corpus_name = nil

if arg[3] == 'false' then
   arg[3] = false
end
LOAD = arg[3] and true

--------------------------
--------------------------

REWARD_DIFFERENCE_SCALING = 1

FIXED_ATTENTION = 1.0
BASE_ATTENTION = 0.6


function makeBoolean(string)
   if string == "false" then
     return false
   elseif string == "true" then
     return true
   else
     print(string)
     crash()
   end
end

print(arg)

DOING_EVALUATION_OUTPUT = makeBoolean(arg[2])

OVERWRITE_MODEL = makeBoolean(arg[23])


if arg[24] == nil then
  print("WARNING arg[24] is nil")
  arg[24] = 'fixed'
elseif string.sub(arg[24], 1, 14) == "NUMERICAL_FILE" then
    NUMERICAL_VALUES_COLUMN = string.sub(arg[24], 15) + 0.0
    arg[24] = "NUMERICAL_FILE"
    print("Numerical Values Column: "..NUMERICAL_VALUES_COLUMN) 
elseif arg[24] ~= "WLEN" and arg[24] ~= "fixed" then
  print(arg[24])
  crash()
end

params = {process_id = arg[1]+0,
                batch_size=arg[4]+0,
                seq_length=arg[5]+0,
                rnn_size=arg[6]+0,
                baseline_rnn_size=20,
                init_weight=0.05,
                lr=((arg[10]+0) + 0.0),
                vocab_size=arg[7]+0,
                max_grad_norm=5,
                lr_att =(arg[12]+0.0),
                lr_momentum = 0.9,
                embeddings_dimensionality = arg[11] + 0,
                ATTENTION_VALUES_BASELINE = - (arg[13] + 0.0),
                TOTAL_ATTENTIONS_WEIGHT = arg[8]+0,
                EXTERNAL_ATTENTION_SOURCE = arg[24],
                gpu_number = 1,
                TASK = arg[17],
                ATTENTION_WITH_EMBEDDINGS = makeBoolean(arg[20]),
                INCLUDE_NUMERICAL_VALUES = (arg[24] == "NUMERICAL_FILE"),
                ablation = arg[22],
                ENTROPY_PENALTY = arg[21] + 0.0}



evaluationAccumulators = {reconstruction_loglikelihood = 0,
                                lm_loglikelihood = 0,
                                numberOfTokens = 0}



use_attention_network = nil
train_attention_network = nil
train_autoencoding = nil

if arg[9] == 'false' then
   arg[9] = false
end
if arg[9] then
  use_attention_network = true
  train_attention_network = true
  train_autoencoding = false
else
  use_attention_network = false
  train_attention_network = false
  train_autoencoding = true
end   
if train_attention_network and (not use_attention_network) then
   crash()
end

if params.TASK == 'qa' then
   qaIncorrect = 0
   qaCorrect = 0
end

if arg[18] == 'false' then
   DO_TRAINING = false
elseif arg[18] == 'true' then
   DO_TRAINING = true
else
   crash()
end

print("DO TRAINING?: "..tostring(DO_TRAINING))

if arg[14] == 'false' then
   arg[14] = false
end
IS_CONTINUING_ATTENTION = arg[14]


fileToBeLoaded = arg[15]

suffixForSaving = arg[16]



experimentName = "pg-test-"..params.TASK.."-"..params.seq_length.."-"..params.rnn_size.."-"..params.lr.."-"..params.embeddings_dimensionality
experimentNameOut = experimentName

if LOAD then
  experimentNameOut = experimentNameOut.."-R-"..params.TOTAL_ATTENTIONS_WEIGHT
end

if IS_CONTINUING_ATTENTION then
   experimentName = experimentNameOut
   experimentNameOut = experimentNameOut.."-R2"
end

if suffixForSaving ~= nil then
   experimentNameOut = experimentNameOut..suffixForSaving
end

fileStats = io.open(experimentNameOut..'-stats', 'w')


print("Printing stuff to "..experimentNameOut)



function transfer_data(x)
  return x:cuda()
end

state_train, state_valid, state_test  = nil
model = {}
paramx, paramdx = nil

------------------------

DATASET = arg[19] + 0


-- QA PARAMETERS
MAX_LENGTH_Q_FOR_QA = nil
MAX_LENGTH_T_FOR_QA = nil
NUMBER_OF_ANSWER_OPTIONS = nil


EPOCHS_NUMBER =  1

PERCENTAGE_OF_DATA = 100



