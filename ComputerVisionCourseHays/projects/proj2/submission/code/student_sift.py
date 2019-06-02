import numpy as np
import cv2
from student_harris import get_interest_points

def createMagnitudeMatrix(dy,dx):
    magnitude = np.zeros(np.asarray(dy).shape)
    for i in range(0, magnitude.shape[0]):
        for j in range(0, magnitude.shape[1]):
            magnitude[i][j] = np.sqrt(np.square(dy[i][j]) + np.square(dx[i][j]))
    return magnitude

def createDirMatrix(dy,dx):
    direction = np.zeros(np.asarray(dy).shape)
    for i in range(0, direction.shape[0]):
        for j in range(0, direction.shape[1]):
            direction[i][j] = np.degrees(np.arctan2(dy[i][j],dx[i][j]))
    return direction
                
def getCenteredFeatures(direc, mag, feature_width,x,y):
    center_distance = feature_width/2
    direction_features = []
    magnitude_features = []
    for i in range(0, len(x)):
        dirSubMatrix = direc[int(y[i]-center_distance): int(y[i] + center_distance), \
                                 int(x[i] - center_distance): int(x[i] + center_distance)].copy()
        magSubMatrix = mag[int(y[i]-center_distance): int(y[i] + center_distance), \
                               int(x[i] - center_distance): int(x[i] + center_distance)].copy()
        
        
        direction_features.append(dirSubMatrix)
        magnitude_features.append(magSubMatrix)
# =============================================================================
#     direction_features = np.asarray(direction_features)
#     magnitude_features = np.asarray(magnitude_features)
# =============================================================================

    return direction_features, magnitude_features

def getHistFeature(direction_feature,magnitude_feature, feature_width):
    smallMatrixWindow = 4
    feature_dimension = int(np.square(feature_width/4) * 8)
    histogramFeature = []
    for i in range(0, len(direction_feature), smallMatrixWindow):
        for j in range(0, len(direction_feature[0]), smallMatrixWindow):
            smallDir = direction_feature[i:i+smallMatrixWindow, j:j+smallMatrixWindow].copy()
            smallMag = magnitude_feature[i:i+smallMatrixWindow, j:j+smallMatrixWindow].copy()
            
            histogramFeature += getHistogram(smallDir, smallMag)

    histogramFeature = histogramFeature/np.linalg.norm(histogramFeature)
    if(len(histogramFeature) is 0 or len(histogramFeature) != feature_dimension):
# =============================================================================
#         print(len(histogramFeature))
# =============================================================================
        return np.zeros(feature_dimension)

    return np.asarray(histogramFeature)
            

def getHistogram(direc, mag):
    magArray = []
    dirArray = []
    for i in range(0, len(direc)):
        for j in range( 0, len( direc[0])):
            magArray.append(mag[i][j])
            dirArray.append(direc[i][j])
            
    histogram = np.histogram(dirArray, bins=[-180,-140,-100,-60,-20,20,60,100,140], weights=magArray)
# =============================================================================
#     print('getHistogram' +str(len(histogram[0])))
# =============================================================================

    return list(histogram[0])
            
    

            
            
    
    
def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'

    print('Number of features ' + str(len(x)))
    
    dy, dx = np.gradient(image)
    
    magnitudeMatrix = createMagnitudeMatrix(dy,dx)
    directionMatrix = createDirMatrix(dy,dx)
    
    direction_features, magnitude_features = \
    getCenteredFeatures(directionMatrix, magnitudeMatrix, feature_width, x,y)
    
    
    print(len(direction_features))
    print(len(magnitude_features))
    
    fv = []
    for i in range (0, len(x)):

        features = getHistFeature(direction_features[i],magnitude_features[i], feature_width)
# =============================================================================
#         print (features.shape)
# =============================================================================
        fv.append(features)


    


    print(np.asarray(fv).shape)
    fv = np.asarray(fv)
    return fv
