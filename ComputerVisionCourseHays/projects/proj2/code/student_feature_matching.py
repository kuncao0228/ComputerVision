import numpy as np


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

            

    
    confidences = []
    matches = []
    matchElementArray= []

    for i in range(0, len(features1)):
        dA = 99999
        dB = 99999
        index_f1 = -1
        index_f2 = -1
        confidence = 0
        for j in range(0, len(features2)):
            dist = np.linalg.norm(np.asarray(features1[i])-np.asarray(features2[j]))
            if(dist < dA):
                dB = dA
                dA = dist
                
                index_f1 = i
                index_f2 = j
                confidence = 1-float(dA/dB)
            elif (dist < dB):
                dB = dist
                confidence = 1-float(dA/dB)
                
        if not (np.all(features1[i]== 0)) and not (np.all(features2[j]==0)):

            match_element = list([index_f1, index_f2, confidence])
            matchElementArray.append(match_element)

            

    matchElementArray = sorted(matchElementArray, key=lambda x:x[2], reverse=True)

    
    for index in matchElementArray:
        indexElement = list([index[0], index[1]])
        conf = index[2]
        matches.append(indexElement)
        confidences.append(conf)
    


        




    matches = np.asarray(matches)
    
    print(matches.shape)
# =============================================================================
#     print (confidences)
# =============================================================================
            
    return matches, confidences
