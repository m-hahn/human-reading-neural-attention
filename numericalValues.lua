numericalValues = {}

numericalValues.numericalValuesImporter = {currentNumericalValuesContents = nil, values = {}}


function numericalValues.readNumericalValuesFile(fileName)
   local values = {}
   print(numericalAnnotationDir..fileName)
   io.input(numericalAnnotationDir..fileName)
   t = io.read("*all")
   local counter = 0
   for line in string.gmatch(t, "[^\n]+") do
     counter = 0
     local lineList = {}
     for token in string.gmatch(line, "[^ ]+") do
        counter = counter+1
        table.insert(lineList, token)
     end
     if #lineList >= 3 then
        values[lineList[1]+0] = lineList[NUMERICAL_VALUES_COLUMN]+0.0
     end
   end
   io.input():close()
   return values
end

function numericalValues.getNumericalValuesForBatchItem(batchIndex)
    local numericalValuesCurrent = {}
    local start = readChunks.corpusReading.locationsOfLastBatchChunks[batchIndex].offset
    if start == 1 then
        numericalValues.numericalValuesImporter.currentNumericalValuesContents = numericalValues.readNumericalValuesFile(readChunks.files[readChunks.corpusReading.currentFile])
    end
    for i=start, start+readChunks.corpusReading.locationsOfLastBatchChunks[batchIndex].length-1 do
       local value = numericalValues.numericalValuesImporter.currentNumericalValuesContents[i]
       if value == nil then
            value = -1
       end
       table.insert(numericalValuesCurrent, value)
    end
    numericalValues.numericalValuesImporter.values[batchIndex] = numericalValuesCurrent
end

