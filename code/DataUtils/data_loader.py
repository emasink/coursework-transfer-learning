
from matplotlib import pyplot as plt
import numpy as np
import os
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset

class ImageDownloader():
    def __init__(self):
        IMAGES_DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data', 'sat4', 'sat-4-full.mat')

        data = scipy.io.loadmat(IMAGES_DATA_PATH)
        train_images = data['train_x']
        train_labels = data['train_y']

        test_images = data['test_x']
        test_labels = data['test_y']

        self.x_train = train_images.transpose(3,0,1,2)
        self.t_train = train_labels.transpose()

        self.x_test = test_images.transpose(3,0,1,2)
        self.t_test = test_labels.transpose()


    def get_train_data(self):
        return self.x_train, self.t_train
    
    def get_test_data(self):
        return self.x_test, self.t_test
    
class CustomDataset(Dataset):
    def __init__(self, split="train", transforms=None):
        downloader = ImageDownloader()

        self.images = downloader.x_train if split == "train" else downloader.x_test
        self.labels = downloader.t_train if split == "train" else downloader.t_test
        self.transforms = transforms

    def __len__(self):
        return (len(self.labels))

    def __getitem__(self, i):
        img = self.images[i][:,:,0:3]
        
        if self.transforms is not None:
            img = np.float32(img)
            img = self.transforms(img)
        else:
            img = torch.from_numpy(img)

        img = img.permute(2, 0, 1).to(torch.float32)
        y = self.labels[i]
        y = torch.from_numpy(y)
        return (img, y)

class ImageProcessingUtilities():
    directory_path = ''
    barren_land_counter = 0
    trees_counter = 0
    grassland_counter = 0
    none_counter = 0

    def __init__(self):
        self.directory_path = '/content/data/'

    def create_image_list(self, image_arrays):
        image_list = []
        for image_array in image_arrays:
            image_list.append(Image.fromarray(image_array[:,:,:3]))
        return image_list

    def get_class_name(self, one_hot_class_encoding):
        class_names = ['barren_land', 'trees', 'grassland', 'none']
        index_of_one = np.argmax(one_hot_class_encoding)#.index(1)
        return class_names[index_of_one]

    def create_label_list(self, labels):
        label_list = []
        for label in labels:
            label_list.append(self.get_class_name(label))
        return label_list

    def increment_class_counter(self, class_name):
        match class_name:
            case 'barren_land':
                self.barren_land_counter += 1
                return self.barren_land_counter
            case 'trees':
                self.trees_counter += 1
                return self.trees_counter
            case 'grassland':
                self.grassland_counter += 1
                return self.grassland_counter
            case 'none':
                self.none_counter += 1
                return self.none_counter
            

