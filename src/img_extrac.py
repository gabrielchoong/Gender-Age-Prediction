

import os
from skimage import io, color, transform


def preprocess(data: str, img_not_found: int) -> tuple[list, list, list]:
  '''
  Preprocesses the dataset by importing images, resizing them,
      and extracting age and gender information.

      Args:
          data: Path to the directory containing the images.
          img_not_found: Number of images that could not be loaded. Default is 0.

      Returns:
          A tuple containing three lists:
          - List of flattened face images.
          - List of ages extracted from the image filenames.
          - List of genders extracted from the image filenames.

      Raises:
          FileNotFoundError: If an image file is not found in the specified directory.
  '''
  
  Faces, Ages, Gender = [], [], []
  
  for index, filename in enumerate(os.listdir(data)):
    
    # avoid FileNotFoundError when introducing new images
    try:
      img = io.imread(os.path.join(data, filename))
      
    except FileNotFoundError:
      img_not_found += 1
    
    '''
    This extracts the age and gender information for training and testing
    
    Please modify the code below if your age and gender information
    are stored differently.
    '''
    age, genders = os.path.basename(filename).split('_')[:2]
    
    
    Ages.append(age)
    Genders.append(genders)
    
    # original and colored images use more resources
    img_gray = color.rgb2gray(img)
    
    img_rescaled = transform.rescale(img_gray, scale=0.5, anti_aliasing=False)
    
    Faces.append(img_rescaled.flatten())
    
    
    # comment out the following 3 lines when not needed
    # if (index < 3):
    #   io.imshow(img_rescaled)
    #   io.show()
      
  print(f'{len(Faces)} image imported \n{img_not_found} images failed to load')