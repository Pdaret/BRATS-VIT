import os
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def rename_file(train_dataset_path, old_file, new_file):
    old_name = os.path.join(train_dataset_path, old_file)
    new_name = os.path.join(train_dataset_path, new_file)

    # Renaming the file
    try:
        os.rename(old_name, new_name)
        print("File has been re-named successfully!")
    except FileExistsError:
        print("File is already renamed!")

def load_nifti_image(file_path):
    """
    Load a .nii file and return it as a numpy array.
    """
    image = nib.load(file_path).get_fdata()
    return image

def normalize_image(image):
    """
    Normalize the image using MinMaxScaler to scale pixel values between 0 and 1.
    """
    scaler = MinMaxScaler()
    image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    return image

def main():
    # Define paths
    TRAIN_DATASET_PATH = "path_to_your_dataset/"  # Replace with your actual path
    old_file = "BraTS20_Training_355/W39_1998.09.19_Segm.nii"
    new_file = "BraTS20_Training_355/BraTS20_Training_355_seg.nii"

    # Rename the file
    rename_file(TRAIN_DATASET_PATH, old_file, new_file)

    # Load and preprocess the FLAIR image
    flair_image_path = os.path.join(TRAIN_DATASET_PATH, "BraTS20_Training_355/BraTS20_Training_355_flair.nii")
    test_image_flair = load_nifti_image(flair_image_path)

    print("Shape: ", test_image_flair.shape)
    print("Dtype: ", test_image_flair.dtype)
    print("Min: ", test_image_flair.min())
    print("Max: ", test_image_flair.max())

    # Normalize the image
    test_image_flair = normalize_image(test_image_flair)

    print("Min after normalization: ", test_image_flair.min())
    print("Max after normalization: ", test_image_flair.max())

    # If you want to convert the numpy array to a PyTorch tensor
    test_image_flair_tensor = torch.tensor(test_image_flair, dtype=torch.float32)
    print("Tensor shape: ", test_image_flair_tensor.shape)

if __name__ == "__main__":
    main()
