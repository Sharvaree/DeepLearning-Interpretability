import numpy as np

def ReadEmbeddingsFile():
  EmbedDict = {}
  f = open('BertEmbeddings.txt','r')
  txt = f.read()
  lines = txt.splitlines()
  for line in lines:
    words = line.split()
    EmbedArr = np.array([float(x) for x in(words[1:])])
    EmbedDict[words[0]] = EmbedArr
  return EmbedDict
  
if __name__ == '__main__':
  EmbedDict = ReadEmbeddingsFile()
  # print(EmbedDict["uncle"].shape)