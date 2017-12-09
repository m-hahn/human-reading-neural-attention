require 'nn'

local ScalarMult, Parent = torch.class('nn.ScalarMult', 'nn.Module')

function ScalarMult:__init(...)
   Parent.__init(self)
   self.output = torch.Tensor(...):zero()

   self.m = nn.Sequential()
   self.m:add(nn.CMulTable())
end

-- first: N times 1
-- second: N times M
function ScalarMult:updateOutput(input)
   self.output = self.m:forward({input[1]:view(-1,1):expandAs(input[2]), input[2]})
   return self.output
end

function ScalarMult:updateGradInput(input, gradOutput)
    gradients = self.m:backward({input[1]:view(-1,1):expandAs(input[2]), input[2]}, gradOutput)
    gradients[1] = gradients[1]:sum(2):view(-1,1)
    return gradients
end



















