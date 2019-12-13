import numpy as np
from scipy import spatial
import matplotlib
import matplotlib.pyplot as plt

def GetNeutralWordDistance(Embedding,WrdPairs):
    dist = 0
    for (u,v) in WrdPairs:
        #dist += np.dot(Embedding,u-v)
        dist += abs(1 - spatial.distance.cosine(Embedding,u-v))
    dist /= len(WrdPairs)
    return dist+0.001

def GetGenderScore(Embedding,WrdPairs,idx = 0):
    dist = 0
    for p in WrdPairs:
        #dist += np.dot(Embedding,u-v)
        dist += abs(spatial.distance.cosine(Embedding,p[idx]))
    dist /= len(WrdPairs)
    return dist

def PlotGenderSimilarity(Neutrals,WrdPairs,DebiasPairs,Embeddings,ReEmbeddings):
    BMale = []
    BFemale = []
    AMale = []
    AFemale = []
    for i in Neutrals:
        BMale.append(GetGenderScore(Embeddings[i],WrdPairs,0))
        BFemale.append(GetGenderScore(Embeddings[i],WrdPairs,1))
        AMale.append(GetGenderScore(ReEmbeddings[i],DebiasPairs,0))
        AFemale.append(GetGenderScore(ReEmbeddings[i],DebiasPairs,1))
    x = np.arange(len(Neutrals))
    width = 0.1
    fig,ax = plt.subplots()
    rects1a = ax.bar(x - 2*width, BMale, width, label='Before Male')
    rects1b = ax.bar(x - width, BFemale, width, label='Before Female')
    rects2a = ax.bar(x + width, AMale, width, label='Debiased Male')
    rects2b = ax.bar(x + 2*width, AFemale, width, label='Debiased Female')

    ax.set_ylabel('Distance to Male/Female Gender')
    ax.set_title('Distance of Words from the Male/Female Words')
    ax.set_xticks(x)
    ax.set_xticklabels(Neutrals)
    ax.legend()

    fig.tight_layout()

    plt.show()

def PlotEmbeddingDistances(Neutrals,WrdPairs,DebiasPairs,Embeddings,ReEmbeddings):
    Before = []
    After = []
    for i in Neutrals:
        Before.append(GetNeutralWordDistance(Embeddings[i],WrdPairs))
        After.append(GetNeutralWordDistance(ReEmbeddings[i],DebiasPairs))
    x = np.arange(len(Neutrals))
    width = 0.35
    fig,ax = plt.subplots()
    rects1 = ax.bar(x - width/2, Before, width, label='Before')
    rects2 = ax.bar(x + width/2, After, width, label='Debiased')

    ax.set_ylabel('Similarity to Gender Axis')
    ax.set_title('Neutral Word Projection on Gender Axis Before and After Debias')
    ax.set_xticks(x)
    ax.set_xticklabels(Neutrals)
    ax.legend()

    fig.tight_layout()

    plt.show()
