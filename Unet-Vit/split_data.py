import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def path_list_into_ids(dir_list):
    """
    Extract the directory names from the full path.
    """
    return [os.path.basename(directory) for directory in dir_list]

def split_dataset(train_dataset_path, val_split=0.2, test_split=0.15):
    """
    Split the dataset into training, validation, and test sets.
    """
    # List directories containing studies
    directories = [f.path for f in os.scandir(train_dataset_path) if f.is_dir()]

    # Extract directory names as IDs
    ids = path_list_into_ids(directories)

    # Split into train+test and validation sets
    train_test_ids, val_ids = train_test_split(ids, test_size=val_split, random_state=42)

    # Further split train+test into training and test sets
    train_ids, test_ids = train_test_split(train_test_ids, test_size=test_split, random_state=42)

    return train_ids, val_ids, test_ids

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

def main():
    # Define the path to your dataset
    TRAIN_DATASET_PATH = "path_to_your_dataset/"  # Replace with your actual path

    # Split the dataset
    train_ids, val_ids, test_ids = split_dataset(TRAIN_DATASET_PATH)

    # Print the lengths of each split
    print(f"Train length: {len(train_ids)}")
    print(f"Validation length: {len(val_ids)}")
    print(f"Test length: {len(test_ids)}")

    # Plot the distribution
    plot_data_distribution(train_ids, val_ids, test_ids)

if __name__ == "__main__":
    main()
