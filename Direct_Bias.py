from scipy import spatial
import numpy as np




## S function- thresholded cosine function
def cosine_distance_pairs(a,b,x,y, delta):
    '''
    This function is the pairs(a,b) and (x,y)
    with a threshold of delta
    '''
    print(np.linalg.norm(x-y))
    if np.linalg.norm(x-y) <= delta:
        return spatial.distance.cosine(a-b, x-y)
    else:
        return 0



## DIRECT BIAS
## DB function
def Direct_Bias(num_samples, gender_direction, samples, c ):
    bias=0
    for i in range(num_samples):
        bias+= pow(abs(spatial.distance.cosine(samples[i,:],gender_direction)),c)
        #print(bias)

    return bias



## INDIRECT BIAS
## beta function
def beta(w,v, g):
    return (np.dot(w,v) - (np.dot(np.dot(w,g), np.dot(v,g))/ (np.linalg.norm(w, ord=2)*np.linalg.norm(v, ord=2))))/np.dot(w,v)
