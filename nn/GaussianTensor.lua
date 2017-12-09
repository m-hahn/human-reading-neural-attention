require 'nn'

local GaussianTensor, Parent = torch.class('nn.GaussianTensor', 'nn.Module')

function GaussianTensor:__init(outputSize)
   Parent.__init(self)
   self.outputSize = outputSize
   self.output = torch.Tensor(outputSize)
end

function GaussianTensor:updateOutput(input)
   self.output = torch.randn(self.outputSize)
   return self.output
end

function GaussianTensor:updateGradInput(input, gradOutput)
   return {}
end


