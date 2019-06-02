
In Section 1 of the jupyter notebook I commented out
#sc.generateFlippedPositives(train_path_pos) 

uncommenting the function will add additional training images to the
specified directory.

num_negative_examples is currently set to 13000
svm C parameter is set to 1e-1

To satisfy the 10minute requirement, please run
the following blocks in the juypter notebook

1. Section 1 load positive and negative
2. Section 2 Train Classifier with 1e-1
3. Then the first run_detector for Section 5 with
the default implementation, this portion should
take ~515 seconds to run

4. The first code block under Section 6

To increase the runtime of my code,

step size can be increased in the method
sampleImageFeatures() on line 371

Furthermore, the number of scales to be considered
can be decreased by changing scale_iterations in line 261 of
run_detector()