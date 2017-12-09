readChunks = {}


function readChunks.createListOfFiles()
    local files = {}
    io.input(LIST_OF_FILES_TO_READ) 
    t = io.read("*all")
    for line in string.gmatch(t, "[^\n]+") do
       table.insert(files, line)
    end
    io.input():close()
    return files
end



readChunks.files = readChunks.createListOfFiles()

readChunks.corpus = {}



readChunks.corpusReading = {lastPosition = 1, currentFileContents = {}, currentFile = 0, locationsOfLastBatchChunks = {}}


function readChunks.resetFileIterator()
    readChunks.corpusReading = {lastPosition = 1, currentFileContents = {}, currentFile = 0, locationsOfLastBatchChunks = {}}
end

function readChunks.hasNextFile()
    return readChunks.corpusReading.currentFile < #readChunks.files
end

-- NOTE TODO this will crash when some of the last files are shorter than params.seq_length
function readChunks.readNextChunk()
   local dataPointFromFile = {}
   -- read a new file if necessary
   while readChunks.corpusReading.lastPosition + params.seq_length >= #readChunks.corpusReading.currentFileContents do
       readChunks.corpusReading.lastPosition = 0
       readChunks.corpusReading.currentFile = readChunks.corpusReading.currentFile + 1
       readChunks.corpusReading.currentFileContents = readFiles.readAFile(readChunks.files[readChunks.corpusReading.currentFile])

       if DOING_EVALUATION_OUTPUT then
          for u=1, params.seq_length do
             table.insert(readChunks.corpusReading.currentFileContents,1)
          end
       end
   end
   local startPosition = readChunks.corpusReading.lastPosition+1
   for i=readChunks.corpusReading.lastPosition+1, readChunks.corpusReading.lastPosition +params.seq_length do
       table.insert(dataPointFromFile, readChunks.corpusReading.currentFileContents[i])
   end
   readChunks.corpusReading.lastPosition = readChunks.corpusReading.lastPosition +params.seq_length
   return dataPointFromFile, readChunks.corpusReading.currentFile, startPosition, params.seq_length
end


function readChunks.readNextChunkForBatchItem(batchIndex)
    local dataPointFromFile, fileNumber, startPosition, lengthOfChunk = readChunks.readNextChunk()
    readChunks.corpusReading.locationsOfLastBatchChunks[batchIndex] = {file = fileNumber, offset = startPosition, length = lengthOfChunk}
    return dataPointFromFile
end

