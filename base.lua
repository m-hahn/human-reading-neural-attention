--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the Apache 2 license found in the
--  LICENSE file in the root directory of this source tree. 
--
-- Adapted from from https://github.com/wojzaremba/lstm/blob/master/base.lua




function clone_network(net)
  local params, gradParams = net:parameters()
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)

    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    collectgarbage()

  mem:close()
  return clone
end



function g_d(f)
  return string.format("%d", torch.round(f))
end
