import numpy as np
import tensorflow as tf
import cv2

class ImageDataGenerator:
    def __init__(self, class_list, scale_size=(256, 256), shuffle=True, nb_classes = 2):

        # Init params
        self.n_classes = nb_classes
        self.scale_size = scale_size
        self.pointer = 0
        self.shuffle = shuffle

        self.read_class_list(class_list)
        if shuffle:
            self.shuffle_data()


    def read_class_list(self,class_list):
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.labels = []
            kernel = np.array([[-1, 2, -2, 2, -1],
                               [2,  -6, 8, -6, 2],
                               [-2, -8, -12, 8, -2],
                               [2, -6, 8, -6, 2],
                               [-1, 2, -2, 2, -1]], np.float32)/12

            for l in lines:
                items = l.split("_")
                mytestImage = tf.image.decode_jpeg(items[0])
                print mytestImage
                # load images
                loadImage = cv2.imread(items[0])
                # cut images and filter
                height, width, channels = loadImage.shape
                newHeight = height/256
                newWidth = width/256
                # print height, width, newWidth, newHeight
                for col in range(newWidth):
                    for row in range(newHeight):
                        cropped = loadImage[row*256:(row+1)*256, col*256:(col+1)*256]
                        dst = cv2.filter2D(cropped, -1, kernel)
                        # print cropped, image
                        # cv2.imshow("image", dst)
                        # cv2.waitKey(0)
                        self.images.append(dst)
                        self.labels.append(0)

            #store total number of data
            self.data_size = len(self.images)

    def shuffle_data(self):
        """Conjoined shuffling of the list of paths and labels."""
        images = self.images
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.images = []
        self.labels = []
        for i in permutation:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0

        if self.shuffle:
            self.shuffle_data()


    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory
        """
        # get images
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(batch_size):
            images[i] = self.images[self.pointer + i]
            label = self.labels[self.pointer + i]
            one_hot_labels[i][label] = 1

        self.pointer += batch_size

        #return array of images and labels
        return images, one_hot_labels
