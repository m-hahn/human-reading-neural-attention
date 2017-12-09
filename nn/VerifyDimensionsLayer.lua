require 'nn'

local VerifyDimensionsLayer, _ = torch.class('nn.VerifyDimensionsLayer', 'nn.Module')

function VerifyDimensionsLayer:__init(dimensions)
   self.dimensions = dimensions
end 

function VerifyDimensionsLayer:updateOutput(input)
   self.output = input
   return self.output
end


function VerifyDimensionsLayer:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

function VerifyDimensionsLayer:clearState()
   -- don't call set because it might reset referenced tensors
   local function clear(f)
      if self[f] then
         if torch.isTensor(self[f]) then
            self[f] = self[f].new()
         elseif type(self[f]) == 'table' then
            self[f] = {}
         else
            self[f] = nil
         end
      end
   end
   clear('output')
   clear('gradInput')
   return self
end
