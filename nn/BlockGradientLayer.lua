require 'nn'

local BlockGradientLayer, _ = torch.class('nn.BlockGradientLayer', 'nn.Module')

-- note does NOT try to infer minibatch size
function BlockGradientLayer:__init(...)
   self.gradInput = torch.Tensor(...):zero()
end


function BlockGradientLayer:updateOutput(input)
   self.output = input
   return self.output
end

-- NOTE assumes input is a tensor, not a table
function BlockGradientLayer:updateGradInput(input, gradOutput)
   return self.gradInput
end

function BlockGradientLayer:accGradParameters(input, gradOutput)
end

