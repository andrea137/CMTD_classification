from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    ref: https://gist.githubusercontent.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d/raw/df4746fa46c3a06f5c041cec18a7eb66fb801197/pytorch_image_folder_with_file_paths.py
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # copy this function from 
        # https://gist.githubusercontent.com/andrea137/0ce7b497f3c10ef1aabb5d4c1ccdcdf0/raw/df4746fa46c3a06f5c041cec18a7eb66fb801197/pytorch_image_folder_with_file_paths.py
        return tuple_with_path