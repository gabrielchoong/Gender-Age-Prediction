import os
import matplotlib.pyplot as plt

def load_images(folderpath):
    file_list = os.listdir(folderpath)
    Data = [image for image in file_list]
    return Data

def count_images(Data):
    return len(Data)

def visualize_images(folderpath, file_list, num_images_to_show):
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


if __name__ == '__main__':
    folderpath = '../FaceData'
    Data = load_images(folderpath)
    print(count_images(Data))

    num_images_to_show = 10  # Specify the number of images to show
    visualize_images(folderpath, Data, num_images_to_show)
