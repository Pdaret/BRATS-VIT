import os
import cv2
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define segmentation classes
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',
    2: 'EDEMA',
    3: 'ENHANCING'
}

# Select Slices and Image Size
VOLUME_SLICES = 100
VOLUME_START_AT = 22
IMG_SIZE = 128

class BrainTumorDataset(Dataset):
    def __init__(self, list_IDs, dataset_path, transform=None):
        """
        Args:
            list_IDs (list): List of IDs for each patient case.
            dataset_path (str): Path to the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.list_IDs = list_IDs
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs) * VOLUME_SLICES

    def __getitem__(self, index):
        # Determine case index and slice index within the volume
        case_index = index // VOLUME_SLICES
        slice_index = index % VOLUME_SLICES

        # Get the case ID
        case_id = self.list_IDs[case_index]

        # Load FLAIR and T1CE images
        case_path = os.path.join(self.dataset_path, case_id)
        flair = nib.load(os.path.join(case_path, f'{case_id}_flair.nii')).get_fdata()
        t1ce = nib.load(os.path.join(case_path, f'{case_id}_t1ce.nii')).get_fdata()
        seg = nib.load(os.path.join(case_path, f'{case_id}_seg.nii')).get_fdata()

        # Resize the images and segmentation masks
        flair_slice = cv2.resize(flair[:, :, slice_index + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        t1ce_slice = cv2.resize(t1ce[:, :, slice_index + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        seg_slice = cv2.resize(seg[:, :, slice_index + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

        # Stack the input channels
        X = np.stack([flair_slice, t1ce_slice], axis=0)  # (2, IMG_SIZE, IMG_SIZE)

        # Convert segmentation labels
        seg_slice[seg_slice == 4] = 3  # Merge label 4 into label 3
        y = torch.tensor(seg_slice, dtype=torch.long)  # Convert to PyTorch tensor

        if self.transform:
            X = self.transform(X)

        # One-hot encode the segmentation mask and resize to match input dimensions
        Y = F.one_hot(y, num_classes=4).permute(2, 0, 1).float()  # (4, IMG_SIZE, IMG_SIZE)

        return torch.tensor(X, dtype=torch.float32), Y

def plot_data_distribution(train_ids, val_ids, test_ids):
    """
    Plot the distribution of the dataset splits.
    """
    plt.bar(["Train", "Valid", "Test"],
            [len(train_ids), len(val_ids), len(test_ids)],
            align='center',
            color=['green', 'red', 'blue'],
            label=["Train", "Valid", "Test"]
           )

    plt.legend()
    plt.ylabel('Number of Studies')
    plt.title('Data Distribution')
    plt.show()

def display_slice_and_segmentation(flair, t1ce, segmentation):
    """
    Display a single slice and its corresponding segmentation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].imshow(flair, cmap='gray')
    axes[0].set_title('Flair')
    axes[0].axis('off')

    axes[1].imshow(t1ce, cmap='gray')
    axes[1].set_title('T1CE')
    axes[1].axis('off')

    axes[2].imshow(segmentation)  # Displaying segmentation
    axes[2].set_title('Segmentation')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    TRAIN_DATASET_PATH = "path_to_your_dataset/"  # Replace with your actual path

    # Assume train_ids, val_ids, and test_ids are already defined
    train_ids = ['BraTS20_Training_355']  # Replace with your actual training IDs
    val_ids = ['BraTS20_Training_356']  # Replace with your actual validation IDs
    test_ids = ['BraTS20_Training_357']  # Replace with your actual test IDs

    # Create dataset instances
    train_dataset = BrainTumorDataset(train_ids, TRAIN_DATASET_PATH)
    valid_dataset = BrainTumorDataset(val_ids, TRAIN_DATASET_PATH)
    test_dataset = BrainTumorDataset(test_ids, TRAIN_DATASET_PATH)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Retrieve a batch from the training loader
    for X_batch, Y_batch in train_loader:
        break

    # Extract Flair, T1CE, and segmentation from the batch
    flair_batch = X_batch[0, 0].numpy()
    t1ce_batch = X_batch[0, 1].numpy()
    segmentation_batch = torch.argmax(Y_batch[0], dim=0).numpy()

    # Display the 50th slice and its segmentation
    slice_index = 60  # Indexing starts from 0
    slice_flair = flair_batch[slice_index]
    slice_t1ce = t1ce_batch[slice_index]
    slice_segmentation = segmentation_batch[slice_index]

    display_slice_and_segmentation(slice_flair, slice_t1ce, slice_segmentation)

if __name__ == "__main__":
    main()
