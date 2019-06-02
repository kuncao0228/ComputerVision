import numpy as np

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """
  

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  im_row = len(image)
  im_column = len(image[0])
  im_z = len(image[0][0])
  
  print("column size for image is: " + str(im_column))
  print("row size for image is: " + str(im_row))
  
  f_row = len(filter)
  f_col = len(filter[0])
  
  print("row size for filter is: " + str(f_row))
  print("column size for filter is: " + str(f_col))
  
  
  pad_row = (int)(f_row/2)
  pad_col = (int)(f_col/2)
  
  print("row pad: " + str(pad_row))
  print("col pad: " + str(pad_col))
  
  #Construct Padded Zero 2d Matrix
  padded_shape = (2*pad_row + im_row, 2*pad_col + im_column, im_z)
  padded_image = np.zeros(padded_shape)
  print ("Shape of Zero Matrix is: " + str(np.shape(padded_image)))
  
  #Fill in Padded Zero 2d Matrix with Image
  
  for x in range(pad_row, im_row+pad_row):
      for y in range(pad_col,im_column+pad_col):
          for z in range(0, im_z):              
              padded_image[x][y][z] = image[x - pad_row][y - pad_col][z]

  filtered_image = np.zeros(np.shape(image))
  print(str(np.shape(filtered_image)))
  
  

   

  for x in range(0, len(filtered_image)):
      for y in range(0, len(filtered_image[0])):
          for z in range(0, len(filtered_image[0][0])):
              filtered_image[x][y][z] = np.sum(padded_image
                            [x:x+f_row, y:y+f_col,z]*filter)





  ### END OF STUDENT CODE ####
  ############################
  
  np.clip(filtered_image, 0,1)

  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """



  
  
  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  low_frequencies = my_imfilter(image1,filter)
  high_frequencies = image2 - my_imfilter(image2, filter)
  hybrid_image = low_frequencies + high_frequencies
  
  low_frequencies = low_frequencies.clip(0,1)
  high_frequencies = high_frequencies.clip(0,1)
  hybrid_image = hybrid_image.clip(0,1)


  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
