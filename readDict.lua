readDict = {}


readDict.chars = {}

function readDict.readDictionary()
   io.input(readDict.dictLocation)
   t = io.read("*all")
   for line in string.gmatch(t, "[^\n]+") do
     local isInSecond = false
     local character
       for x in string.gmatch(line, "[^ ]+") do
          character = x
          if isInSecond == true then
            break
          end
          isInSecond = true
       end
       table.insert(readDict.chars, character)
   end
   io.input():close()
   assert(params.vocab_size <= #readDict.chars, tostring(#readDict.chars))
end


-- essentially, the entityNumber/entityID should be used in the softmax
function readDict.createNumbersToEntityIDsIndex()
    readDict.numbersToEntityIDs = {}
    readDict.entityIDToNumberIndex = {}
    readDict.maximumOccurringEntity = 0
    for num, char in pairs(readDict.chars) do
        if char:sub(1,7) == "@entity" then
            entityNumber = char:sub(8)+1 --better add 1 here because the entities start at 0
            readDict.numbersToEntityIDs[num+0] = entityNumber
            readDict.entityIDToNumberIndex[entityNumber] = num
            readDict.maximumOccurringEntity = math.max(readDict.maximumOccurringEntity, entityNumber)
        end
    end
    assert(NUMBER_OF_ANSWER_OPTIONS ~= nil)
    assert(readDict.maximumOccurringEntity ~= nil)

    if(NUMBER_OF_ANSWER_OPTIONS<readDict.maximumOccurringEntity) then
       print("--")
       print( (NUMBER_OF_ANSWER_OPTIONS))
       print( readDict.maximumOccurringEntity)
       crash()
    end
end



function readDict.word2Num(word)
  for i=1,#readDict.chars do
   if readDict.chars[i] == word then
     return i
   end
  end
  return 0
end
