import os
import matplotlib.pyplot as plt

def load_images(folderpath):
    return os.listdir(folderpath)

def count_images(Data):
    return len(Data)

def visualize_images(folderpath, file_list, num_images_to_show=10):
    for i in range(num_images_to_show):
        filepath = os.path.join(folderpath, file_list[i])
        visualize_image(filepath)

def visualize_image(filepath):
    # Read the image file
    image = plt.imread(filepath)

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off the axis labels and ticks
    plt.show()
    plt.pause(0.1)

# returns a List of tuples of age and sex
def gender_and_age(folderpath):
    file_list = os.listdir(folderpath)
    data = []
    for image in file_list:
        parts = image.split('_')
        age = parts[0]
        sex = 'male' if parts[1] == '0' else 'female'
        data.append((age, sex))
    return data


if __name__ == '__main__':
    folderpath = '../public/FaceData'
    Data = load_images(folderpath)
    print(count_images(Data))

    num_images_to_show = 10  # Specify the number of images to show
    visualize_images(folderpath, Data, num_images_to_show)

    age_gender = gender_and_age(folderpath)

    # store in dictionary
    image_info = {filename: info for filename, info in zip(Data, age_gender)}
    print(image_info)