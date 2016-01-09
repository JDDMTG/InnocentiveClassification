import math

def findEdge(array2dim):
  rows,cols = array2dim.shape
  for r in xrange(rows):
    for c in xrange(cols):
      if (math.isinf(array2dim[r][c]) or math.isnan(array2dim[r][c])):
        print "Row,Col = " + str(r) + "," + str(c)
        print array2dim[r][c]
  return False

def outputFile
