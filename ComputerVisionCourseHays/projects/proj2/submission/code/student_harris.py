import cv2
import numpy as np
import matplotlib.pyplot as plt


def getNonMaxSupression(harrisResult, feature_width):
    harrisResult = np.asarray(harrisResult)
    threshold = np.mean(harrisResult)
    windowDimension = 30
    slide = 8
    hashSet = set()
    for x in range(windowDimension,len(harrisResult[0])-windowDimension, slide):
        for y in range(windowDimension, len(harrisResult)-windowDimension, slide):
            subMatrix = harrisResult[y:y+windowDimension, x:x+windowDimension]
            indexTuple = np.unravel_index(subMatrix.argmax(), subMatrix.shape)
        
        

            mainMatrixTuple = (int(indexTuple[0]) + y ,int(indexTuple[1]) + x,subMatrix.max())
            if (subMatrix.max() > threshold):
                hashSet.add(mainMatrixTuple)
        
        
    
    print('Finished get NonMaxSupression')

            

    return hashSet

def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    print('Processing Image')
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################
    y_derivative, x_derivative = np.gradient(image)
    
    
    dx_squared = np.square(x_derivative)
    dy_squared = np.square(y_derivative)
    dxdy_product = x_derivative * y_derivative
    
    
    cutoff_frequency = 3
    filter = cv2.getGaussianKernel(ksize=cutoff_frequency*4,
                               sigma=cutoff_frequency)
# =============================================================================
#     filter = np.dot(filter, filter.T)
# =============================================================================


    g_dxsquared = cv2.filter2D(dx_squared,-1,filter)
    g_dysquared = cv2.filter2D(dy_squared,-1,filter)
    g_dxdy_product = cv2.filter2D(dxdy_product,-1,filter)
    
    
    harrisResult = (g_dxsquared*g_dysquared - np.square(g_dxdy_product)) - (.04 * np.square(g_dxsquared + g_dysquared ))
     
    print('Starting getNon Max Supression')
    hashSet = getNonMaxSupression(harrisResult, feature_width)
    
    
    



    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################

    #Sort by harris response returns by greatest to least
    sortedHarrisSet = sorted(hashSet, key=lambda i:i[2], reverse=True)
    sortedHarrisRadiusSet = []
    sortedHarrisRadiusSet.append(sortedHarrisSet[0])

    for x in range(1, len(sortedHarrisSet)):
        supRadius = np.sqrt(np.square(sortedHarrisSet[x][0]-sortedHarrisSet[x-1][0]) \
                            + np.square(sortedHarrisSet[x][1]-sortedHarrisSet[x-1][1])) \
                            * 1.1 * sortedHarrisSet[x][2]
        element = (sortedHarrisSet[x][0],sortedHarrisSet[x][1], supRadius)
        sortedHarrisRadiusSet.append(element)
    
    sortedHarrisRadiusSet = sorted(sortedHarrisRadiusSet, key=lambda x:x[2], reverse=False)
        

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    
    
    x=[]
    y=[]
    maxN = 0
            
    for s in sortedHarrisRadiusSet:
        if(maxN < 1800):
            y.append(s[0])
            x.append(s[1])
            maxN +=1
        else:
            break
        
    x = np.asarray(x)
    y = np.asarray(y)
    

    print(len(x))
    print(len(y))
    return x,y, confidences, scales, orientations


