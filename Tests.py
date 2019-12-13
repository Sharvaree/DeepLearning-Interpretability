import numpy as np
from scipy import spatial
import matplotlib
import matplotlib.pyplot as plt

def GetNeutralWordDistance(Embedding,WrdPairs):
    dist = 0
    for (u,v) in WrdPairs:
        dist += spatial.distance.cosine(Embedding,u-v)
    dist /= len(WrdPairs)
    return dist

def PlotEmbeddingDistances(NeutralIds,WrdPairs,Embeddings,ReEmbeddings):
    Before = []
    After = []
    for i in NeutralIds:
        Before.append(GetNeutralWordDistance(Embeddings[i],WrdPairs))
        After.append(GetNeutralWordDistance(ReEmbeddings[i],WrdPairs))
    x = np.arrange(len(NeutralIds))
    width = 0.35
    fig,ax = plt.subplots()
    rects1 = ax.bar(x - width/2, Before, width, label='Before')
    rects2 = ax.bar(x + width/2, After, width, label='Debiased')

    ax.set_ylabel('Mean Distance on Gender Axis')
    ax.set_title('Neutral Word Distance on Gender Axis Before and After Debias')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
