getParams = {}

TEMP = {}

function getParamFromFile(filename)
   if pcall(function ()    io.input(filename)
     TEMP.param = nil
            local t = io.read("*all")
            io.input():close()
            for line in string.gmatch(t, "[^\n]+") do
              if line:len() > 0 then
                TEMP.param= line+0.0
              end
            end
      end) then
      print(filename.."  "..TEMP.param)
      return TEMP.param
   else
      print("ERROR "..filename)
      return nil
   end
end

function getAttentionFromFile()
     local filename = "attention-"..arg[1]
     local result = getParamFromFile(filename)
     if result ~= nil then
       FIXED_ATTENTION = result
     end
end

function getLearningRateFromFile()
     local filename = "lr-"..arg[1]
     local result = getParamFromFile(filename)
     if result ~= nil then
       params.lr = result
     end
end

function getAttentionLearningRateFromFile()
     local filename = "lr-att-"..arg[1]
     local result = getParamFromFile(filename)
     if result ~= nil then
       params.lr_att = result
     end
end

function getTotalAttentionWeightFromFile()
     local filename = "total-att-weight-"..arg[1]
     local result = getParamFromFile(filename)
     if result ~= nil then
       params.TOTAL_ATTENTIONS_WEIGHT = result
     end
end
