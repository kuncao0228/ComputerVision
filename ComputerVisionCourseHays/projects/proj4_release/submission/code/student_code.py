import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from IPython.core.debugger import set_trace


def get_tiny_images(image_paths):
    feats = []
    for i in range(0,len(image_paths)):
        image = load_image_gray(image_paths[i])

  
        ret = getMiniImage(image)
        feats.append(ret)

   
      
    feats = np.asarray(feats)
    print(feats.shape)
    return feats


def getMiniImage(image):

    rowInc = int(image.shape[0]/16)
    colInc = int(image.shape[1]/16)
    
    ret = []
    
    
    for x in range(0, image.shape[0], rowInc):

        for y in range(0, image.shape[1], colInc):
            
            mini_image = image[x:rowInc + x, y:colInc + y]
            
            rowCenter = int(mini_image.shape[0]/2)
            colCenter = int(mini_image.shape[1]/2)
            
            if(mini_image.shape[0] >= rowCenter and mini_image.shape[1] >= colCenter):
                
                if(len(ret) < 256):
                    ret.append(mini_image[rowCenter, colCenter])
    ret = ret-np.mean(ret)
    ret = ret/np.std(ret)
    return ret
            

def build_vocabulary(image_paths, vocab_size):
# =============================================================================
#   """
#   This function will sample SIFT descriptors from the training images,
#   cluster them with kmeans, and then return the cluster centers.
# 
#   Useful functions:
#   -   Use load_image(path) to load RGB images and load_image_gray(path) to load
#           grayscale images
#   -   frames, descriptors = vlfeat.sift.dsift(img)
#         http://www.vlfeat.org/matlab/vl_dsift.html
#           -  frames is a N x 2 matrix of locations, which can be thrown away
#           here (but possibly used for extra credit in get_bags_of_sifts if
#           you're making a "spatial pyramid").
#           -  descriptors is a N x 128 matrix of SIFT features
#         Note: there are step, bin size, and smoothing parameters you can
#         manipulate for dsift(). We recommend debugging with the 'fast'
#         parameter. This approximate version of SIFT is about 20 times faster to
#         compute. Also, be sure not to use the default value of step size. It
#         will be very slow and you'll see relatively little performance gain
#         from extremely dense sampling. You are welcome to use your own SIFT
#         feature code! It will probably be slower, though.
#   -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
#           http://www.vlfeat.org/matlab/vl_kmeans.html
#             -  X is a N x d numpy array of sampled SIFT features, where N is
#                the number of features sampled. N should be pretty large!
#             -  K is the number of clusters desired (vocab_size)
#                cluster_centers is a K x d matrix of cluster centers. This is
#                your vocabulary.
# 
#   Args:
#   -   image_paths: list of image paths.
#   -   vocab_size: size of vocabulary
# 
#   Returns:
#   -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
#       cluster center / visual word
#   """
#   # Load images from the training set. To save computation time, you don't
#   # necessarily need to sample from all images, although it would be better
#   # to do so. You can randomly sample the descriptors from each image to save
#   # memory and speed up the clustering. Or you can simply call vl_dsift with
#   # a large step size here, but a smaller step size in get_bags_of_sifts.
#   #
#   # For each loaded image, get some SIFT features. You don't have to get as
#   # many SIFT features as you will in get_bags_of_sift, because you're only
#   # trying to get a representative sample here.
#   #
#   # Once you have tens of thousands of SIFT features from many training
#   # images, cluster them with kmeans. The resulting centroids are now your
#   # visual word vocabulary.
# =============================================================================

    dim = 128      # length of the SIFT descriptors that you are going to compute.
    
 
    vocab = np.zeros((vocab_size,dim))
    descrip_array = np.empty((0,dim))
    for i in range(0,len(image_paths)):

        image = load_image_gray(image_paths[i])
        _, descriptors = vlfeat.sift.dsift(image, fast=True, step=8)
        descrip_array = np.append(descrip_array, descriptors, axis=0)
    print(descrip_array.shape)
    print('Optimizing KMeans')
        

    
# =============================================================================
#     K = np.empty((0,dim))
#     for i in range(len(indexArray)):
#         print(descrip_array[indexArray[i]].shape)
#         K = np.append(K, descrip_array[indexArray[i]].reshape(1,dim), axis = 0)
# =============================================================================
        
        

    
    
# =============================================================================
#     print(descrip_array.shape)
#     print(descrip_array[indexArray].shape)
# =============================================================================
    vocab = vlfeat.kmeans.kmeans(descrip_array, vocab_size)
    print('K Means Finished')
    


    return vocab

def get_bags_of_sifts(image_paths, vocab_filename):
# =============================================================================
#   """
#   This feature representation is described in the handout, lecture
#   materials, and Szeliski chapter 14.
#   You will want to construct SIFT features here in the same way you
#   did in build_vocabulary() (except for possibly changing the sampling
#   rate) and then assign each local feature to its nearest cluster center
#   and build a histogram indicating how many times each cluster was used.
#   Don't forget to normalize the histogram, or else a larger image with more
#   SIFT features will look very different from a smaller version of the same
#   image.
# 
#   Useful functions:
#   -   Use load_image(path) to load RGB images and load_image_gray(path) to load
#           grayscale images
#   -   frames, descriptors = vlfeat.sift.dsift(img)
#           http://www.vlfeat.org/matlab/vl_dsift.html
#         frames is a M x 2 matrix of locations, which can be thrown away here
#           (but possibly used for extra credit in get_bags_of_sifts if you're
#           making a "spatial pyramid").
#         descriptors is a M x 128 matrix of SIFT features
#           note: there are step, bin size, and smoothing parameters you can
#           manipulate for dsift(). We recommend debugging with the 'fast'
#           parameter. This approximate version of SIFT is about 20 times faster
#           to compute. Also, be sure not to use the default value of step size.
#           It will be very slow and you'll see relatively little performance
#           gain from extremely dense sampling. You are welcome to use your own
#           SIFT feature code! It will probably be slower, though.
#   -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
#           finds the cluster assigments for features in data
#             -  data is a M x d matrix of image features
#             -  vocab is the vocab_size x d matrix of cluster centers
#             (vocabulary)
#             -  assignments is a Mx1 array of assignments of feature vectors to
#             nearest cluster centers, each element is an integer in
#             [0, vocab_size)
# 
#   Args:
#   -   image_paths: paths to N images
#   -   vocab_filename: Path to the precomputed vocabulary.
#           This function assumes that vocab_filename exists and contains an
#           vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
#           or visual word. This ndarray is saved to disk rather than passed in
#           as a parameter to avoid recomputing the vocabulary every run.
# 
#   Returns:
#   -   image_feats: N x d matrix, where d is the dimensionality of the
#           feature representation. In this case, d will equal the number of
#           clusters or equivalently the number of entries in each image's
#           histogram (vocab_size) below.
#   """
# =============================================================================
# load vocabulary
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)
    print(vocab.shape)
    vocab = np.float32(vocab)
    
    
# =============================================================================
#     #Cross Validation
#     number_of_samples = 20
#     random_indexArray = np.random.choice(len(image_paths), number_of_samples, replace = False) 
#          
#     image_paths = image_paths[random_indexArray]
# =============================================================================

   
    
    # dummy features variable
    feats = []  
    
    for i in range(0,len(image_paths)):
        image = load_image_gray(image_paths[i])
        _, descriptors = vlfeat.sift.dsift(image, fast=True, step=8, float_descriptors=True)

        assignments = vlfeat.kmeans.kmeans_quantize(descriptors, vocab)


       
        histogram = [0] * vocab.shape[0]
        for j in range(len(assignments)):
            histogram[assignments[j]] += 1
            
        histogram = np.asarray(histogram)
        histogram = histogram/np.linalg.norm(histogram)
        feats.append(histogram.copy())
            
        
    feats = np.asarray(feats)
    print('Done Finding Features')

    return feats

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
    metric='euclidean'):
# =============================================================================
#     """
#     This function will predict the category for every test image by finding
#     the training image with most similar features. Instead of 1 nearest
#     neighbor, you can vote based on k nearest neighbors which will increase
#     performance (although you need to pick a reasonable value for k).
#     
#     Useful functions:
#         -   D = sklearn_pairwise.pairwise_distances(X, Y)
#             computes the distance matrix D between all pairs of rows in X and Y.
#             -  X is a N x d numpy array of d-dimensional features arranged along
#             N rows
#             -  Y is a M x d numpy array of d-dimensional features arranged along
#             N rows
#             -  D is a N x M numpy array where d(i, j) is the distance between row
#             i of X and row j of Y
# 
#       Args:
#           -   train_image_feats:  N x d numpy array, where d is the dimensionality of
#               the feature representation
#         -   train_labels: N element list, where each entry is a string indicating
#               the ground truth category for each training image
#               -   test_image_feats: M x d numpy array, where d is the dimensionality of the
#               feature representation. You can assume N = M, unless you have changed
#               the starter code
#               -   metric: (optional) metric to be used for nearest neighbor.
#               Can be used to select different distance functions. The default
#               metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
#               well for histograms
# 
#       Returns:
#           -   test_labels: M element list, where each entry is a string indicating the
#           predicted category for each testing image
#           """
# =============================================================================
    test_labels = []
    
# =============================================================================
#     #Cross Validation
#     k_vals = [1,2,3,4,5,6,7,8]
#     crossValPrecisions = []
#     iterations = 100
#     crossValMatrix = []
#     for j in range(len(k_vals)):
#         for i in range(iterations):
#             crossValPrecisions.append(performKNNCrossValidation(train_image_feats, train_labels, test_image_feats, k_vals[j]))
#     
#     
#         row = [k_vals[j], np.mean(crossValPrecisions), np.std(crossValPrecisions)]
#         crossValMatrix.append(row)
#     print(crossValMatrix)
# 
# =============================================================================
    

         
   
    k = 7

    #Non Cross Validation
    Dist = sklearn_pairwise.pairwise_distances(train_image_feats, test_image_feats)    
    for i in range(Dist.shape[0]):

        min_index = Dist[i].argsort()[:k]
        label = getMostOccuringWord(min_index, train_labels)
        
        test_labels.append(label)
    

          



    return  test_labels

def getGMMClusters(image_paths):
    dim= 128
    descrip_array = np.empty((0,dim))
    for i in range(0,int(len(image_paths))):
        print(i)

        image = load_image_gray(image_paths[i])
        _, descriptors = vlfeat.sift.dsift(image, fast=True, step=8)
        descrip_array = np.append(descrip_array, descriptors, axis=0)
        
    clusters_no = 400;
    means, covars, priors, _, _ = vlfeat.gmm.gmm(descrip_array, clusters_no)
    means = means.transpose()
    covars = covars.transpose()
    return means, covars, priors

def getFisherFeats(image_paths, means, covars, priors):
    feats = []
    for i in range(0,int(len(image_paths))):
        print(i)

        image = load_image_gray(image_paths[i])
        _, descriptors = vlfeat.sift.dsift(image, fast=True, step=8)
        feats.append(vlfeat.fisher.fisher(np.float32(descriptors.transpose()), np.float32(means), np.float32(covars), np.float32(priors), fast=True))
    
    feats = np.asarray(feats)
    print(feats.shape)
    return feats
    

def getMostOccuringWord(index, train_labels):
    possible_labels = []
    for i in range(len(index)):
        possible_labels.append(train_labels[index[i]])
    count = 0
    word = ''

    for i in range(len(possible_labels)):
        temp_count = possible_labels.count(possible_labels[i])
        
        if (temp_count > count):
            count = temp_count
            word = possible_labels[i]
            

    return word

def performKNNCrossValidation(train_image_feats, train_labels, test_image_feats, k):
    number_of_samples = 100
    
    # CROSS VALIDATION
    validation_index = np.random.choice(len(train_labels), number_of_samples, replace = False)
    cross_train_index = []
    
    for i in range(0, len(train_labels)):
        if i not in validation_index:
            cross_train_index.append(i)
    
    cross_train_index = np.asarray(cross_train_index)
    
    cross_train_image_feats = []
    cross_train_labels = []
    
    val_train_image_feats = []
    val_actual_labels = []

    for i in range(len(train_labels)):
        if i in cross_train_index:
            cross_train_image_feats.append(train_image_feats[i,:])
            cross_train_labels.append(train_labels[i])
        if i in validation_index:
            val_train_image_feats.append(train_image_feats[i,:])
            val_actual_labels.append(train_labels[i])

            
    cross_train_image_feats = np.asarray(cross_train_image_feats)
    cross_train_labels = np.asarray(cross_train_labels)
    val_train_image_feats = np.asarray(val_train_image_feats)

            
    val_labels = []  
    #Validation Test
    Dist = sklearn_pairwise.pairwise_distances(val_train_image_feats, cross_train_image_feats)


    
    for i in range(Dist.shape[0]):

        min_index = Dist[i].argsort()[:k]
        label = getMostOccuringWord(min_index, cross_train_labels)
        
        val_labels.append(label)  

    count = 0
    for i in range (len(val_labels)):
        if val_labels[i] == val_actual_labels[i]:
            count +=1
    

    
    return float(count/number_of_samples)

def performSVMCrossValidation(train_image_feats, train_labels, test_image_feats, C):
    # CROSS VALIDATION
    number_of_samples = 100
    validation_index = np.random.choice(len(train_labels), number_of_samples, replace = False)
    cross_train_index = []
    
    for i in range(0, len(train_labels)):
        if i not in validation_index:
            cross_train_index.append(i)
    
    cross_train_index = np.asarray(cross_train_index)
    
    cross_train_image_feats = []
    cross_train_labels = []
    
    val_train_image_feats = []
    val_actual_labels = []

    for i in range(len(train_labels)):
        if i in cross_train_index:
            cross_train_image_feats.append(train_image_feats[i,:])
            cross_train_labels.append(train_labels[i])
        if i in validation_index:
            val_train_image_feats.append(train_image_feats[i,:])
            val_actual_labels.append(train_labels[i])

            
    cross_train_image_feats = np.asarray(cross_train_image_feats)
    cross_train_labels = np.asarray(cross_train_labels)
    val_train_image_feats = np.asarray(val_train_image_feats)
    
    categories = list(set(cross_train_labels))

    svms = {cat: LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=5) for cat in categories}
  
    binary_matrix = np.zeros((len(categories), len(cross_train_labels)))

    
    dist_matrix = np.zeros((len(categories), len(val_train_image_feats)))
    


  
    for i in range(len(categories)):
        for j in range(len(cross_train_labels)):
            if(cross_train_labels[j] == categories[i]):

                binary_matrix[i,j] = 1


    
    for i in range(len(svms)):
        svm = svms[categories[i]].fit(cross_train_image_feats, binary_matrix[i])
        for k in range(val_train_image_feats.shape[0]):
            dist_matrix[i,k] = svm.coef_ @ val_train_image_feats[k] + svm.intercept_

         

                
    val_labels_index = np.argmax(dist_matrix, axis = 0)
    
    val_labels = []
    

    for i in range(len(val_labels_index)):
        val_labels.append(categories[val_labels_index[i]])
        
    count = 0
    for i in range (len(val_labels)):
        if val_labels[i] == val_actual_labels[i]:
            count +=1
    

    
    return float(count/number_of_samples)
    
    
        

def svm_classify(train_image_feats, train_labels, test_image_feats):
# =============================================================================
#   """
#   This function will train a linear SVM for every category (i.e. one vs all)
#   and then use the learned linear classifiers to predict the category of
#   every test image. Every test feature will be evaluated with all 15 SVMs
#   and the most confident SVM will "win". Confidence, or distance from the
#   margin, is W*X + B where '*' is the inner product or dot product and W and
#   B are the learned hyperplane parameters.
# 
#   Useful functions:
#   -   sklearn LinearSVC
#         http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
#   -   svm.fit(X, y)
#   -   set(l)
# 
#   Args:
#   -   train_image_feats:  N x d numpy array, where d is the dimensionality of
#           the feature representation
#   -   train_labels: N element list, where each entry is a string indicating the
#           ground truth category for each training image
#   -   test_image_feats: M x d numpy array, where d is the dimensionality of the
#           feature representation. You can assume N = M, unless you have changed
#           the starter code
#   Returns:
#   -   test_labels: M element list, where each entry is a string indicating the
#           predicted category for each testing image
#   """
# =============================================================================
    
# =============================================================================
#     #Cross Validation
#     C_vals = [1,2,3,4,5,6,7,8,9,10]
#     crossValPrecisions = []
#     iterations = 20
#     crossValMatrix = []
#     for j in range(len(C_vals)):
#         for i in range(iterations):
#             crossValPrecisions.append(performSVMCrossValidation(train_image_feats, train_labels, test_image_feats, C_vals[j]))
#     
#     
#         row = [C_vals[j], np.mean(crossValPrecisions), np.std(crossValPrecisions)]
#         crossValMatrix.append(row)
#     print(crossValMatrix)
# =============================================================================

    categories = list(set(train_labels))

    svms = {cat: LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=5) for cat in categories}
    
  
    binary_matrix = np.zeros((len(categories), len(train_labels)))

    
    dist_matrix = np.zeros((len(categories), len(test_image_feats)))
    


  
    for i in range(len(categories)):
        for j in range(len(train_labels)):
            if(train_labels[j] == categories[i]):

                binary_matrix[i,j] = 1


    
    for i in range(len(svms)):
        svm = svms[categories[i]].fit(train_image_feats, binary_matrix[i])
        for k in range(test_image_feats.shape[0]):
            dist_matrix[i,k] = svm.coef_ @ test_image_feats[k] + svm.intercept_

         

                
    test_labels_index = np.argmax(dist_matrix, axis = 0)
    
    test_labels = []
    

    for i in range(len(test_labels_index)):
        test_labels.append(categories[test_labels_index[i]])

    return test_labels
