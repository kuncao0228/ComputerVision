import numpy as np
import cyvlfeat as vlfeat
from utils import *
import os.path as osp
from glob import glob
from random import shuffle
from IPython.core.debugger import set_trace
from sklearn.svm import LinearSVC

import time
import cv2

def generateFlippedPositives(train_path_pos):
    positive_files = glob(osp.join(train_path_pos, '*.jpg'))
    print(train_path_pos)
    count = 0
    for fileName in positive_files:
        count+=1

        

        image = load_image_gray(fileName)  
        flipped_image = image.copy()
        
        for row in range(flipped_image.shape[0]):
            col = 0;
            while col < int(flipped_image.shape[1]/2):
                temp = flipped_image[row, col]
                flipped_image[row, col] = flipped_image[row, flipped_image.shape[1] - col - 1]
                flipped_image[row, flipped_image.shape[1] - col - 1] = temp
                col +=1
        
        path = train_path_pos + '/image_' + str(count) + '.jpg'
        print('Writing to: ' + path)
        print('Done Generating Flipped Images')

        cv2.imwrite(path, flipped_image)
                
        
                
                


                


    
def get_positive_features(train_path_pos, feature_params):
    """
    This function should return all positive training examples (faces) from
    36x36 images in 'train_path_pos'. Each face should be converted into a
    HoG template according to 'feature_params'.

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features

    Args:
    -   train_path_pos: (string) This directory contains 36x36 face images
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slower because the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time
            (although you don't have to make the detector step size equal a
            single HoG cell).

    Returns:
    -   feats: N x D matrix where N is the number of faces and D is the template
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    

    n_cell = np.ceil(win_size/cell_size).astype('int')
    positive_files = glob(osp.join(train_path_pos, '*.jpg'))
    
    
    feats = []

    for fileName in positive_files:


        image = load_image_gray(fileName)  

        hog_feat = vlfeat.hog.hog(image,  cell_size)   
        feature = np.ndarray.flatten(hog_feat)
        feats.append(feature)

    
    feats = np.asarray(feats)
    print('positive feature shape' + str(feats.shape))



    return feats

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    """
    This function should return negative training examples (non-faces) from any
    images in 'non_face_scn_path'. Images should be loaded in grayscale because
    the positive training data is only available in grayscale (use
    load_image_gray()).

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   num_samples: number of negatives to be mined. It is not important for
            the function to find exactly 'num_samples' non-face features. For
            example, you might try to sample some number from each image, but
            some images might be too small to find enough.

    Returns:
    -   N x D matrix where N is the number of non-faces and D is the feature
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    time_start = time.clock()
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))
    n_cell = np.ceil(win_size/cell_size).astype('int')
    
    feats = []

    
    random_index_array = np.random.randint(len(negative_files),size=num_samples)
    
    for index in random_index_array:
        
        image = load_image_gray(negative_files[index])
        if image.shape[0] > 350 and image.shape[1] > 350:
        
        
            min_row = np.random.randint(image.shape[0]-win_size)
            min_col = np.random.randint(image.shape[1]-win_size)
        
            sub_image = image[min_row:min_row+win_size, min_col:min_col+win_size]
            hog_feat = vlfeat.hog.hog(sub_image,  cell_size) 
            feature = np.ndarray.flatten(hog_feat)
            feats.append(feature)

    
    feats = np.asarray(feats)
    print('negative feature shape' + str(feats.shape))

    print('RunTime Took: ' + str(time.clock()-time_start) + ' seconds.')
    return feats

def train_classifier(features_pos, features_neg, C):
    """
    This function trains a linear SVM classifier on the positive and negative
    features obtained from the previous steps. We fit a model to the features
    and return the svm object.

    Args:
    -   features_pos: N X D array. This contains an array of positive features
            extracted from get_positive_feats().
    -   features_neg: M X D array. This contains an array of negative features
            extracted from get_negative_feats().

    Returns:
    -   svm: LinearSVC object. This returns a SVM classifier object trained
            on the positive and negative features.
    """

    train = np.append(features_pos, features_neg, axis=0)
    label = np.append([1] * len(features_pos), [-1]* len(features_neg))

    
    print('Total Positive Size ' + str(features_pos.shape))
    print('Total Negative Size ' + str(features_neg.shape))
    print('Total Train Size ' + str( train.shape))
    
    svm = LinearSVC(random_state=0, loss='hinge', C=C).fit(train, label)



    return svm

def mine_hard_negs(non_face_scn_path, svm, feature_params):
    """
    This function is pretty similar to get_random_negative_features(). The only
    difference is that instead of returning all the extracted features, you only
    return the features with false-positive prediction.

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features
    -   svm.predict(feat): predict features

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   svm: LinearSVC object

    Returns:
    -   N x D matrix where N is the number of non-faces which are
            false-positive and D is the feature dimensionality.
    """

    num_samples = 10000
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))
    n_cell = np.ceil(win_size/cell_size).astype('int')
    
    feats = []
    
    random_index_array = np.random.randint(len(negative_files),size=num_samples)
    
    for index in random_index_array:
        
        image = load_image_gray(negative_files[index])
        
        
        min_row = np.random.randint(image.shape[0]-win_size)
        min_col = np.random.randint(image.shape[1]-win_size)
        
        sub_image = image[min_row:min_row+win_size, min_col:min_col+win_size]
        hog_feat = vlfeat.hog.hog(sub_image,  cell_size) 
        feature = np.ndarray.flatten(hog_feat)
        feats.append(feature)

    
    feats = np.asarray(feats)
    confidences = svm.decision_function(np.vstack(feats))
    
    
    preds = confidences.copy()
    preds[preds >= 0] = 1
    preds[preds <  0] = -1
    fp_feats = []
    for i in range(len(confidences)):
        if preds[i] > 0 and preds[i]!=-1:
            fp_feats.append(feats[i])

    print('Total False Positive Features: ' + str(len(fp_feats)))

    return fp_feats

def run_detector(test_scn_path, svm, feature_params, verbose=False):
        
    scale_iterations = 18
    time_start = time.clock()
    im_filenames = sorted(glob(osp.join(test_scn_path, '*.jpg')))
    
    
    bboxes = np.empty((0, 4))
    confidences = np.empty(0)
    
    image_ids = []

    # number of top detections to feed to NMS
    topk = 500
    threshold = -1.5

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    template_size = int(win_size / cell_size)


    image_scales = []     

    scale = 1
    for i in range(scale_iterations):
    

        image_scales.append(scale)
        scale*=.9
        

    
        

    for idx, im_filename in enumerate(im_filenames):
        print(im_filename)
        im_feats = np.empty((0,cell_size * cell_size * 31))
        cur_bboxes = np.empty((0, 4))
        for scale in image_scales:

# =============================================================================
#             print('Detecting faces in {:s}'.format(im_filename))
# =============================================================================
            im = load_image_gray(im_filename)
            im_id = osp.split(im_filename)[-1]
            im_shape = im.shape
            
            im = cv2.resize(im, (0,0), fx = scale, fy = scale)


            feats, boxes = sampleImageFeatures(win_size, im_shape, im, cell_size, scale)
            feats = np.asarray(feats)


            if (feats.shape[0] !=0 and feats.shape[1] == cell_size * cell_size * 31):
                im_feats = np.append(im_feats,feats, axis = 0)

            
                cur_bboxes = np.append(cur_bboxes, boxes, axis=0)
            


            

            
        if (cur_bboxes.shape[0] !=0 and im_feats.shape[0] != 0):

            im_feats = np.asarray(im_feats)
# =============================================================================
#             print(im_feats.shape)
#             print(cur_bboxes.shape)
# =============================================================================
            cur_confidences = svm.decision_function(im_feats)
            index_threshold = np.nonzero(cur_confidences > threshold)          
            cur_confidences = cur_confidences[index_threshold]
            cur_bboxes = cur_bboxes[index_threshold]



            idsort = np.argsort(-cur_confidences)[:topk]
            cur_bboxes = cur_bboxes[idsort]
            cur_confidences = cur_confidences[idsort]
            
            
# =============================================================================
#             print(cur_confidences.shape)
#             print(cur_bboxes.shape)
# =============================================================================
            
            if cur_bboxes.shape[0] != 0 and cur_bboxes.shape[1] !=0 and\
                    cur_confidences.shape[0] != 0:
                is_valid_bbox = non_max_suppression_bbox(cur_bboxes, cur_confidences,
                                                 im_shape, verbose=verbose)
            

                cur_bboxes = cur_bboxes[is_valid_bbox]
                cur_confidences = cur_confidences[is_valid_bbox]
        
                bboxes = np.vstack((bboxes, cur_bboxes))
                confidences = np.hstack((confidences, cur_confidences))
                image_ids.extend([im_id] * len(cur_confidences))
        

    print('Done')
    print('RunTime Took: ' + str(time.clock()-time_start) + ' seconds.')
    return bboxes, confidences, image_ids



def sampleImageFeatures(win_size, im_shape, im, cell_size, scale):
    feats = []
    boxes = []
    step_size = 9
    
    for i in range(win_size, im_shape[0]-win_size, step_size):
        for j in range(win_size, im_shape[1] - win_size,step_size):
            
            sub_image = im[i-win_size:i, j-win_size:j]
            if sub_image.shape[0] == win_size and sub_image.shape[1] == win_size:
                hog_feat = vlfeat.hog.hog(sub_image,  cell_size) 
                flattened_feat = np.ndarray.flatten(hog_feat)
                boxes.append([np.ceil((j-win_size)/scale), np.ceil((i-win_size)/scale),np.ceil(j/scale),np.ceil(i/scale) ])
                feats.append(flattened_feat)
                
    feats = np.asarray(feats)
    
    boxes = np.asarray(boxes)

    return feats, boxes
    
