from utils import get_data_dir, get_cpu_cores
from preprocessing import preprocess, crop, get_crop_dimensions
from multiprocessing import Pool
import os
import numpy as np
from collections import namedtuple

# TODO: isn't smart about checking if a patient has
# already been preprocessed etc.

# def crop_and_save(images3D):


# python code to preprocess stage1 images & save on disk
def preprocess_stage1():
    STAGE1_DIR = get_data_dir() + 'stage1/'
    patients = [STAGE1_DIR + patient for patient in os.listdir(STAGE1_DIR)]
    #p = Pool(get_cpu_cores())
    # EDIT patients based on # of patients
    # you want to preprocess
    images3D = map(preprocess, patients)
    images3D = crop(images3D)

    save_dir = 'stage1-preprocessed-large'

    new_dir = get_data_dir() + save_dir
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # write out all of the arrays
    for patient_path, image3D in zip(patients, images3D):
        # import pdb; pdb.set_trace()
        new_path = patient_path.replace('stage1', save_dir) + '.npy'
        np.save(new_path, image3D)

PathImage = namedtuple("PathImage", "path image")

# def preprocess_intermediate(intermediate_dir, save_dir):
#     intermediates = []
#     intermediate_path = get_data_dir() + intermediate_dir
#     for dirpath, subdirs, file_names in os.walk(intermediate_path):
#         for file_name in file_names:
#             if file_name.endswith(".npy"):
#                 path = dirpath + file_name
#                 intermediates.append((path, np.load(dirpath + file_name)))

#     patient_paths, images3D = zip(*intermediates)
#     cropped_images = crop(images3D)

#     # write out all of the arrays
#     for patient_path, cropped_image in zip(patient_paths, cropped_images):
#         # import pdb; pdb.set_trace()
#         new_path = patient_path.replace(intermediate_dir, save_dir) + '.npy'
#         np.save(new_path, cropped_image)

def preprocess_intermediate(intermediate_dir, save_dir):
    intermediates = []
    intermediate_path = get_data_dir() + intermediate_dir
    for dirpath, subdirs, file_names in os.walk(intermediate_path):
        for file_name in file_names:
            if file_name.endswith(".npy"):
                path = dirpath + file_name
                intermediates.append((path, np.load(dirpath + file_name)))

    print "got intermediates"
    patient_paths, images3D = zip(*intermediates)

    median_dim_z, median_dim_y = get_crop_dimensions(images3D)

    for elem in intermediates:
        cropped_image = crop(elem[1], median_dim_z, median_dim_y)
        new_path = elem[0].replace(intermediate_dir, save_dir)
        np.save(new_path, cropped_image)
        print "saved patient"

if __name__ == '__main__':
    preprocess_stage1()
    preprocess_intermediate('stage1-intermediate-large/', 'stage1-preprocessed-large/')
