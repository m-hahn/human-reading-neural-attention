require 'nn'

local PrintLayer, _ = torch.class('nn.PrintLayer', 'nn.Module')

-- note does NOT try to infer minibatch size
function PrintLayer:__init(id,probability,fullPrint)
--   self.gradInput = torch.Tensor()
   self.id = id
   self.probability = probability or 1.0 
   self.fullPrint = fullPrint or false
end


function PrintLayer:updateOutput(input)
   self.output = input
   if torch.uniform() < self.probability then
     print("<<< FORWARD PRINT-LAYER  "..self.id)
     if torch.type(self.output) == 'table' then
       print(self.output)
     else --if torch.type(self.output) == 'torch.CudaTensor' then
       if self.fullPrint then
          print(self.output)
       end

       print(torch.type(self.output))
       print(self.output:size())
       print("NORM  "..self.output:norm())
  --   else
    --   print(self.output)
     end
     print(">>> FORWARD PRINT-LAYER  "..self.id)
   end
   return self.output
end

-- NOTE assumes input is a tensor, not a table
function PrintLayer:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
if torch.uniform() < self.probability then

   print("<<< BACKWARD PRINT-LAYER  "..self.id)
   if torch.type(self.gradInput) == 'table' then
     print(self.gradInput)
   else
       if self.fullPrint then
          print(self.gradInput)
       end

     print(torch.type(self.gradInput:size()))
     print(self.gradInput:size())
     print("NORM  "..self.gradInput:norm())
   end
   print(">>> BACKWARD PRINT-LAYER  "..self.id)
end



   return self.gradInput
end


function PrintLayer:clearState()
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

