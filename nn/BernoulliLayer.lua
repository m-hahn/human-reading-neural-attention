require 'nn'

local BernoulliLayer, Parent = torch.class('nn.BernoulliLayer', 'nn.Module')

function BernoulliLayer:__init(...)
   Parent.__init(self)

   self.gradInput = torch.CudaTensor(...):zero()
   self.output = {torch.CudaTensor(...),torch.CudaTensor(...)}


end

-- input: tensor of attention scores
-- output: table consisting of (1) table of filter values (1 or 0), (2) probabilities for the choices, (3) numbers of attended items per item (0 <= ... <= 1) , (4) avg entropy
-- hard-coded that: the first dimension is the batch size, and the second dimension has size 1
function BernoulliLayer:updateOutput(input)
   --print(input[1])
   
   -- for every element 0 or 1 depending on whether it will be attended to
--   local filterValues = input:clone()
  -- self.output = filterValues
   self.output[1]:copy(input)
   self.output[1]:apply(function(attention)
         local experiment = torch.uniform()
         --print(experiment.."  "..attention)
         if experiment < attention then
           return 1
         else
           return 0
         end
      end)

   torch.cmul(self.output[2],input,torch.add(self.output[1],-0.5):mul(2)):add(1):add(-1,self.output[1])


   -- divide by number of attended items

   --The probability of the choice made in the sampling of the filterValues is given by:
   --
   --       (1-filterValues) + 2 * (filterValues - 0.5) * input
   --
   -- which results in
   -- (1-1) + 2 * (1-0.5) * attention = attention if the word is attended to
   -- (1-0) + 2 * (0-0.5) * attention = 1 - attention if it is not attended to
--[[   local probabilities = torch.cmul(input, torch.add(filterValues, -0.5):mul(2)):add(torch.mul(filterValues, -1):add(1))]]


   return self.output
end

-- the gradient of the probability (the second output) with regard to the attention values

function BernoulliLayer:updateGradInput(input, gradOutput)
    self.gradInput:copy(self.output[1])
    self.gradInput:apply(function(x)
       if x == 1 then
          return(1)
       elseif x == 0 then
          return(-1)
       else
          assert(false,x.."")
       end
      end)
    self.gradInput:cmul(gradOutput[2])
    return self.gradInput
end


