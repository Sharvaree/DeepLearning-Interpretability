import numpy as np
from scipy import spatial
import matplotlib
import matplotlib.pyplot as plt

def GetNeutralWordDistance(Embedding,WrdPairs):
    dist = 0
    for (u,v) in WrdPairs:
        #dist += np.dot(Embedding,u-v)
        dist += spatial.distance.cosine(Embedding,u-v)
        break
    #dist /= len(WrdPairs)
    print(dist)
    return dist

def PlotEmbeddingDistances(Neutrals,WrdPairs,Embeddings,ReEmbeddings):
    Before = []
    After = []
    for i in Neutrals:
        Before.append(GetNeutralWordDistance(Embeddings[i],WrdPairs))
        After.append(GetNeutralWordDistance(ReEmbeddings[i],WrdPairs))
    x = np.arange(len(Neutrals))
    width = 0.35
    fig,ax = plt.subplots()
    rects1 = ax.bar(x - width/2, Before, width, label='Before')
    rects2 = ax.bar(x + width/2, After, width, label='Debiased')

    ax.set_ylabel('Mean Distance on Gender Axis')
    ax.set_title('Neutral Word Distance on Gender Axis Before and After Debias')
    ax.set_xticks(x)
    ax.set_xticklabels(Neutrals)
    ax.legend()

    fig.tight_layout()

    plt.show()
