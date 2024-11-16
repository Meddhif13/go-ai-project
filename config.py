import os

# Define the path to your dataset
DATASET_PATH = os.getenv('DATASET_PATH', 'data/games.1000000.data')

# Function to verify if the dataset exists
def verify_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at: {DATASET_PATH}")
        return False
    print("Dataset is available")
    return True

# Example usage
if __name__ == "__main__":
    verify_dataset()
