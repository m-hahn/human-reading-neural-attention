require 'nn'

local UniformLayer, Parent = torch.class('nn.UniformLayer', 'nn.Module')

function UniformLayer:__init(...)
   Parent.__init(self)
   self.output = torch.Tensor(...):zero()
end

function UniformLayer:updateOutput(input)
   self.output:uniform(-0.5,0.5)
   return self.output
end

function UniformLayer:updateGradInput(input, gradOutput)
    return nil
end
       
