attention = {}

require('nn.BernoulliLayer')


attention.ABLATE_INPUT = false
attention.ABLATE_STATE = false
attention.ABLATE_SURPRISAL = false


if string.match(params.ablation, 'i') then
  attention.ABLATE_INPUT = true
end
if string.match(params.ablation, 'r') then
  attention.ABLATE_STATE = true
end
if string.match(params.ablation, 's') then
  attention.ABLATE_SURPRISAL = true
end


print("ABLATION INP STATE SURP")
print(attention.ABLATE_INPUT)
print(attention.ABLATE_STATE)
print(attention.ABLATE_SURPRISAL)



function attention.createAttentionNetwork()
   assert(params.TASK == 'combined')
   if USE_BASELINE_NETWORK then
      return attention.createAttentionNetworkEmbeddingsSurprisalWithBaseline()
   end
   local x = nn.Identity()()
   local xemb = nn.BlockGradientLayer(params.batch_size, params.embeddings_dimensionality)(nn.LookupTable(params.vocab_size,params.embeddings_dimensionality)(x))
   local y = nn.Identity()()
   local surprisal = nn.Identity()()

   -- ABLATION OF INPUT
   if attention.ABLATE_INPUT then
     xemb = nn.MulConstant(0)(xemb)
   end

   local x2h = nn.Linear(params.embeddings_dimensionality, params.rnn_size)(xemb)
   local y2h = nn.Linear(params.rnn_size, params.rnn_size)(y)

   -- ABLATION OF STATE
   if attention.ABLATE_STATE then
      y2h = nn.MulConstant(0)(y2h)
   end

   local z2h = nn.Linear(1, params.rnn_size)(surprisal)

   -- ABLATION OF SURPRISAL
   if attention.ABLATE_SURPRISAL then
      z2h = nn.MulConstant(0)(z2h)
   end

   local hidden = nn.Sigmoid()(nn.CAddTable()({x2h, y2h, z2h}))
   local attention = (nn.Sigmoid()(nn.Linear(params.rnn_size, 1)(hidden)))
   local module = nn.gModule({x, y, surprisal},
                                      {attention})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end



