import os
import shutil

class DatasetSaver:

    def __init__(self):

        self.building_counter = 0
        self.land_counter = 0
        self.road_counter = 0
        self.vegetation_counter = 0
        self.water_counter = 0
        self.unlabeled_counter = 0

        
        self.data_folder_path = os.path.join(os.path.dirname(os.getcwd()), 'duomenys', 'dubai_dataset')

        if os.path.exists(self.data_folder_path):
            shutil.rmtree(self.data_folder_path)

        self.classes = ['building', 'land', 'road', 'vegetation', 'water', 'unlabeled']

        self.create_directories()
   
    def save_image(self, image, class_name):
        image_name = str(self.increment_class_counter(class_name)).zfill(6) + ".jpg"
        image_path = os.path.join(self.data_folder_path, class_name, image_name)
        image.save(image_path, "JPEG")

    def create_directories(self):
        for class_name in self.classes:
            if not os.path.exists(self.data_folder_path):
                os.makedirs(self.data_folder_path)
                
            if not os.path.exists(class_name):
                class_folder_path = os.path.join(self.data_folder_path, class_name)
                os.makedirs(class_folder_path) 


    def increment_class_counter(self, class_name):
        match class_name:
            case 'building':
                self.building_counter += 1
                return self.building_counter
            case 'land':
                self.land_counter += 1
                return self.land_counter
            case 'road':
                self.road_counter += 1
                return self.road_counter
            case 'vegetation':
                self.vegetation_counter += 1
                return self.vegetation_counter
            case 'water':
                self.water_counter += 1
                return self.water_counter
            case 'unlabeled':
                self.unlabeled_counter += 1
                return self.unlabeled_counter
            case _:
                print("Invalid class name")
                return -1