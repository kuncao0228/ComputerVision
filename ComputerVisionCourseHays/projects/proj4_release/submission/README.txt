Cross Validation:
1. Uncomment code block with #Cross Validation comment in
nearest_neighbor_classify and svm_classify to enable cross validation
(Uncommented as a default upon submission)


Vocab Files:
Under /code there should exist a set of vocab files.
The size 400 clusters are currently defaulted as vocab.pkl. Other
clusters are denoted as vocab<N>.pkl where N is the size of the clusters.

Extra Credit:
1. K-NN supports multiple values kf K, mainly utilizes getMostOcurringWord()
Simply Change the value of 'k' variable in the method nearest_neighbor_classify()

2. GMM clusters and Fisher features were implemented using getGMMClusters()
and getFeatureFeats()

To run, make sure there exists a .vocab file so build vocabulary does not run

Next in section 2a of jupyter proj4.ipynb, comment out #train_image_feats and test_image_feats
that use sc.get bags of sift

Lastly, add

means, covars, priors = sc.getGMMClusters(train_image_paths)
train_image_feats = sc.getFisherFeats(train_image_paths, means, covars, priors)
test_image_feats = sc.getFisherFeats(test_image_paths, means, covars, priors) 

to the end of section 2a.