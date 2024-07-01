
'''
Important! This test has not been implemented correctly

Don't push this test into the project yet
'''


# import the necessary packages
import os
from skimage import io, color, transform
from nose.tools import assert_raises, assert_equal

# define the test function
def test_preprocess():
    # create a temporary directory to store the test images
    temp_dir = os.path.join(os.path.dirname(__file__), 'test_images')
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    
    # create sample images and save them to the temporary directory
    img_files = ['0001_male.jpg', '0002_female.jpg', '0003_male.jpg']
    for img_file in img_files:
        img = io.imread(os.path.join('data', img_file))
        io.imsave(os.path.join(temp_dir, img_file), img)
    
    # define the input arguments for the preprocess function
    input_args = (temp_dir, 0)
    
    # call the preprocess function and store the output
    output = preprocess(*input_args)
    
    # extract the expected output from the input arguments
    expected_faces, expected_ages, expected_genders = input_args[1:]
    
    # assert that the output is a tuple of three lists
    assert isinstance(output, tuple)
    assert len(output) == 3
    
    # assert that the face images are correct
    assert_equal(len(output[0]), len(expected_faces))
    for face, exp_face in zip(output[0], expected_faces):
        assert_equal(face.shape, exp_face.shape)
    
    # assert that the ages are correct
    assert_equal(len(output[1]), len(expected_ages))
    for age, exp_age in zip(output[1], expected_ages):
        assert_equal(age, exp_age)
    
    # assert that the genders are correct
    assert_equal(len(output[2]), len(expected_genders))
    for gender, exp_gender in zip(output[2], expected_genders):
        assert_equal(gender, exp_gender)
    
    # remove the temporary directory
    os.rmdir(temp_dir)

# run the tests
if __name__ == '__main__':
    test_preprocess()