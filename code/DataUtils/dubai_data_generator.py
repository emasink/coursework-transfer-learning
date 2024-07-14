import glob
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import dataset_saver as ds
from PIL import Image
from torch.utils.data import Dataset
import dataset_saver
from keras.utils import to_categorical

class DubaiDataGenerator():
    def __init__(self, images_paths, mask_paths):
        self.images = self.get_image_list(images_paths)
        self.masks = self.get_image_list(mask_paths)
        self.ds = dataset_saver.DatasetSaver()

    def generate_and_save_data(self):
        for image, mask in zip(self.images, self.masks):
            image_tiles, mask_tiles = self.get_image_tiles(image, mask, 28, 28)
            class_names = []
            for image_tile, mask_tile in zip(image_tiles, mask_tiles):
                image_tile, class_name = self.classify_tile(image_tile, mask_tile)
                if image_tile is not None and class_name is not None:
                    class_names.append(self.ds.save_image(image_tile, class_name))
            print(set(class_names))

    def get_image_list(self, files_paths):
        images = []
        for file_path in files_paths:
            images.append(Image.open(file_path))
        return images
    
    def get_image_tiles(self, image, mask, patch_width, patch_height):
        img_width, img_height = image.size
        imeges = []
        masks = []
        for i in range(0, img_width, patch_width):
            for j in range(0, img_height, patch_height):
                if not (i+patch_width) >= img_width and not (j+patch_height) >= img_height:
                    box = (i, j, i+patch_width, j+patch_height)
                    imeges.append(image.crop(box))
                    masks.append(mask.crop(box))
        return imeges, masks

    def classify_tile(self, tile, mask):
        unique_pixels, counts = self.get_unique_pixels(mask)
        counts = np.array(counts)
        most_common_pixel_value_index = np.argmax(counts)
        if counts[most_common_pixel_value_index] >= self.get_treshold(tile, 0.7):
            class_name = self.get_class_name(unique_pixels[most_common_pixel_value_index])
            return tile, class_name
        return None, None
    
    def get_treshold(self, tile, threshold):
        w, h = tile.size
        pixel_count = w * h
        return pixel_count * threshold

    def get_unique_pixels(self, mask):
        # Load the image
        image_pixels = []
        w, h = mask.size
        for i in range(w):
            for j in range(h):
                pixel = mask.getpixel((i,j))
                image_pixels.append(pixel)

        pixels = np.array(image_pixels)
        unique_pixels, counts = np.unique(pixels, axis=0, return_counts=True)
        return unique_pixels, counts
    
    def get_class_name(self, pixel):
        pixel = list(pixel)
        match pixel:
            case [60, 16, 152]:
                return 'building'
            case [132, 41, 246]:
                return 'land'
            case [110, 193, 228]:
                return 'road'
            case [254, 221, 58]:
                return 'vegetation'
            case [226, 169, 41]:
                return 'water'
            case [155, 155, 155]:
                return 'unlabeled'
            case _: 
                Exception("Invalid pixel value")

class DubaiDataset(Dataset):
    # glob pattern has to be provided for this dataset

    def __init__(self, data_folder_path:str, transforms=None):
        self.transforms = transforms
        self.glob_pattern = os.path.join(data_folder_path, "*/*.jpg")
        self.images_paths = glob.glob(self.glob_pattern)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = Image.open(image_path)
        image = np.array(image)
        if self.transforms is not None:
            # change typep of array to float32
            # image = np.float32(image)
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image)
        label = image_path.split('/')[-2]
        label = self.get_one_hot_of_class(label)
        return (image, label)

    def get_one_hot_of_class(self, label):
        index_of_class = self.get__class_names().index(label)
        mask = np.array([index_of_class])
        mask_one_hot = to_categorical(mask, num_classes=6)
        return mask_one_hot
    
    def get__class_names(self):
        return ['building', 'land', 'road', 'vegetation', 'water', 'unlabeled']

    def get_image_count_for_class(self, class_name):
        dubai_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'dubai_dataset')
        return len(glob.glob(f'{dubai_path}/{class_name}/*.jpg'))