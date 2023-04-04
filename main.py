# Image compression
#
# You'll need Python 3 and must install the 'numpy' package (just for its arrays).
#
# The code also uses the 'netpbm' package, which is included in this directory.
#
# You can also display a PNM image using the netpbm library as, for example:
#
#   python3 netpbm.py images/cortex.pnm
#
# NOTES:
#
#   - Use struct.pack( '>h', val ) to convert signed short int 'val' to a two-byte bytearray
#
#   - Use struct.pack( '>H', val ) to convert unsigned short int 'val' to a two-byte bytearray
#
#   - Use struct.unpack( '>H', twoBytes )[0] to convert a two-byte bytearray to an unsigned short int.  Note the [0].
#
#   - Use struct.unpack( '>' + 'H' * count, manyBytes ) to convert a bytearray of 2 * 'count' bytes to a tuple of 'count' unsigned short ints


import sys, os, math, time, struct, netpbm
import numpy as np


# Text at the beginning of the compressed file, to identify it

headerText = b'my compressed image - v1.0'



# Compress an image


def compress( inputFile, outputFile ):

  # Read the input file into a numpy array of 8-bit values
  #
  # The img.shape is a 3-type with rows,columns,channels, where
  # channels is the number of component in each pixel.  The img.dtype
  # is 'uint8', meaning that each component is an 8-bit unsigned
  # integer.

  img = netpbm.imread( inputFile ).astype('uint8')

  # Note that single-channel images will have a 'shape' with only two
  # components: the y dimensions and the x dimension.  So you will
  # have to detect whether the 'shape' has two or three components and
  # set the number of channels accordingly.  Furthermore,
  # single-channel images must be indexed as img[y,x] instead of
  # img[y,x,k].  You'll need two pieces of similar code: one piece for
  # the single-channel case and one piece for the multi-channel case.

  # Compress the image

  startTime = time.time()

  outputBytes = bytearray()

  # ---------------- [YOUR CODE HERE] ----------------
  #
  # REPLACE THE CODE BELOW WITH YOUR OWN CODE TO FILL THE 'outputBytes' ARRAY.

  #for y in range(img.shape[0]):
    #for x in range(img.shape[1]):
      #for c in range(img.shape[2]):
        #outputBytes.append( img[y,x,c] )
  #use byte strings because otherwise python applies some utf formatting to strings
  #use dictionary for key value pair
  #take difference mod 256 between the bytes? like this (curr + 256 - prev) % 256
  #in decoder (prev + diff) % 256
  #use struct.pack and unpack to convert ints to bytes
  #do red channel, then green channel, then blue channel (or just make sure to do them one at a time)
  compress_dict = dict()
  for i in range(0,256):
    index = struct.pack('>H', i)
    #print (index)
    compress_dict[index] = i;
  #sys.exit(1)

  #print (compress_dict)
  
  #generation of difference encoding
  diff = []
  if (len(img.shape) == 2): #if image is single channel
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if ((y == 0)):
                diff.append(img[y,x])
            else:
                diff.append((img[y,x] + 256 - img[y-1,x]) % 256)
  elif (len(img.shape) > 2): #if image is multi-channel
    for y in range(img.shape[0]): #one colour channel at a time should make difference encoding more useful
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                if ((y == 0)):
                    diff.append(img[y,x,c])
                else:
                    diff.append((img[y,x,c] + 256 - img[y-1,x,c]) % 256)

  #print(diff)
  #LZW encoding algorithm
  """
            X = symbol sequence <x1, x2, ... xn>

            D = dictionary of symbol sequences, initially
            containing all one-symbol sequences

            s = empty symbol sequence <>

            for x in x1, x2, ..., xn
                if s+x is in D
                    s = s+x
                else
                    output the dictionary index of s
                    add s+x to D
                    s = <x>

            output dictionary index of s
  """
  #X is our diff encoding of the image
  #D is compress_dict
  #create s now
  s = bytes() #using byte array instead of strings
  next_index = 256 #next byte string not in the dictionary will be encoded with 256
  for i in range(len(diff)):
    diff_as_bytes = struct.pack('>H', diff[i])
    #print("s, x, s+x: ")
    #print(s)
    #print(diff_as_bytes)
    s_plus_x = s + diff_as_bytes
    #print(s_plus_x)
    #print("\n")
    if s_plus_x in compress_dict:
        s = s_plus_x
    else:
        index_as_bytes = struct.pack('>H', compress_dict[s])
        outputBytes += index_as_bytes #output index of s
        if (next_index < 2**16): #if we still have room for length 16 bit numbers
            compress_dict[s_plus_x] = next_index #add s+x to D
            next_index = next_index + 1
        s = struct.pack('>H', diff[i]) #s after is equal to x
  
  index_as_bytes = struct.pack('>H', compress_dict[s])
  outputBytes += index_as_bytes #output index of final s
  
  
  # ---------------- [END OF YOUR CODE] ----------------

  endTime = time.time()

  # Output the bytes
  #
  # Include the 'headerText' to identify the type of file.  Include
  # the rows, columns, channels so that the image shape can be
  # reconstructed.

  outputFile.write( headerText + b'\n' )
  if (len(img.shape) == 2):
    outputFile.write( bytes( '%d %d %d\n' % (img.shape[0], img.shape[1], 1), encoding='utf8' ) )
  else:
    outputFile.write( bytes( '%d %d %d\n' % (img.shape[0], img.shape[1], img.shape[2]), encoding='utf8' ) )
  outputFile.write( outputBytes )

  # Print information about the compression
  if (len(img.shape) == 2):
    inSize  = img.shape[0] * img.shape[1]
  else:
    inSize  = img.shape[0] * img.shape[1] * img.shape[2]
  outSize = len(outputBytes)

  sys.stderr.write( 'Input size:         %d bytes\n' % inSize )
  sys.stderr.write( 'Output size:        %d bytes\n' % outSize )
  sys.stderr.write( 'Compression factor: %.2f\n' % (inSize/float(outSize)) )
  sys.stderr.write( 'Compression time:   %.2f seconds\n' % (endTime - startTime) )
  


# Uncompress an image

def uncompress( inputFile, outputFile ):

  # Check that it's a known file

  if inputFile.readline() != headerText + b'\n':
    sys.stderr.write( "Input is not in the '%s' format.\n" % headerText )
    sys.exit(1)
    
  # Read the rows, columns, and channels.  

  rows, columns, numChannels = [ int(x) for x in inputFile.readline().split() ]

  # Read the raw bytes.

  inputBytes = bytearray(inputFile.read())

  startTime = time.time()

  # ---------------- [YOUR CODE HERE] ----------------
  #
  # REPLACE THIS WITH YOUR OWN CODE TO CONVERT THE 'inputBytes' ARRAY INTO AN IMAGE IN 'img'.
  if numChannels > 1:
    img = np.empty( [rows,columns,numChannels], dtype=np.uint8 )
  else:
    img = np.empty( [rows,columns], dtype=np.uint8 )
    
  flattened_image = np.empty(rows * columns * numChannels, dtype=np.uint8)

  uncompress_arr = [] #decoder side "dictionary" can use array since we receive indices
  for i in range(0,256):
    temp_list = []
    temp_list.append(i)
    uncompress_arr.append(temp_list)
    
  flat_index = 0;
  
  #LZW Decoding Algorithm
  """
            I = index sequence <i1, i2, ... ik>

            D = dictionary of symbol sequences, initially
            containing all one-symbol sequences

            s = D[i1]   # s stores the sequence of the
                    # most recently-received index
            output s

            for i in i2, i3, ..., ik

                # t stores the sequence to output

                if i is in D
                    t = D[i]
                    add s + first_symbol(t) to D
                else
                    t = s + first_symbol(s)
                    add t to D

                output t
                s = t
  """
  #in decoder (prev + diff) % 256
  t = []
  s = []
  s = uncompress_arr[struct.unpack( '>H', inputBytes[0:2] )[0]].copy()
  flattened_image[0] = s[0]
  flat_index = 1
  #debug = 0
  for i in range(2, len(inputBytes), 2):
    inputShort = struct.unpack( '>H', inputBytes[i:i+2] )[0]
    if (inputShort < len(uncompress_arr)):
        t = uncompress_arr[inputShort].copy()
        #print("\nlength of dict so far, index, and len(t): ")
        #print(len(uncompress_arr))
        #print(inputShort)
        #print(len(t))
        s.append(t[0])
        uncompress_arr.append(s.copy())
    else:
        s.append(s[0])
        t = s.copy()
        uncompress_arr.append(t.copy())
    for val in t:
        #if flat_index >= rows * columns * numChannels:
            #debug = 1
            #break
        flattened_image[flat_index] = val
        flat_index += 1
    s = t.copy()

  i = 0
  if numChannels > 1:
      for y in range(rows):
        for x in range(columns):
            for c in range(numChannels):
                if ((y == 0)):
                    img[y,x,c] = flattened_image[i]
                else:
                    img[y,x,c] = (int(flattened_image[i]) + int(img[y-1,x,c])) % 256 #typecast to int to get rid of runtime warning for uint8 overflow
                i += 1
  else:
      for y in range(rows):
        for x in range(columns):
                if ((y == 0)):
                    img[y,x] = flattened_image[i]
                else:
                    img[y,x] = (int(flattened_image[i]) + int(img[y-1,x])) % 256 #typecast to int to get rid of runtime warning for uint8 overflow
                i += 1
            #decoder uses just an array not a dictionary (because we receive the indices)

  # ---------------- [END OF YOUR CODE] ----------------

  endTime = time.time()
  sys.stderr.write( 'Uncompression time %.2f seconds\n' % (endTime - startTime) )

  # Output the image

  netpbm.imsave( outputFile, img )
  

  
# The command line is 
#
#   main.py {flag} {input image filename} {output image filename}
#
# where {flag} is one of 'c' or 'u' for compress or uncompress and
# either filename can be '-' for standard input or standard output.


if len(sys.argv) < 4:
  sys.stderr.write( 'Usage: main.py c|u {input image filename} {output image filename}\n' )
  sys.exit(1)

# Get input file
 
if sys.argv[2] == '-':
  inputFile = sys.stdin
else:
  try:
    inputFile = open( sys.argv[2], 'rb' )
  except:
    sys.stderr.write( "Could not open input file '%s'.\n" % sys.argv[2] )
    sys.exit(1)

# Get output file

if sys.argv[3] == '-':
  outputFile = sys.stdout
else:
  try:
    outputFile = open( sys.argv[3], 'wb' )
  except:
    sys.stderr.write( "Could not open output file '%s'.\n" % sys.argv[3] )
    sys.exit(1)

# Run the algorithm

if sys.argv[1] == 'c':
  compress( inputFile, outputFile )
elif sys.argv[1] == 'u':
  uncompress( inputFile, outputFile )
else:
  sys.stderr.write( 'Usage: main.py c|u {input image filename} {output image filename}\n' )
  sys.exit(1)
