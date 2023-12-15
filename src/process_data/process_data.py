import os
from pathlib import Path
from torch.utils.data import DataLoader
from utils.transform_data import get_resize_transform
from utils.regression_image_folder import RegressionImageFolder


def get_path_to_label_from_path(path: Path) -> dict:
  image_paths = list(path.glob("*/*/*.jpg"))
  path_to_labels = dict()
  for image_path in image_paths:
    # TODO: find label in path
    path_to_labels[image_path] = 10
  return path_to_labels

def load_dataset(data_path: str, transform):
  ''' Expected data path:
    PATH_TO_DATA/ <- overall dataset folder
        train/ <- training images
            0/ <- class name as folder name
                image01.jpeg
                image02.jpeg
                ...
        test/ <- testing images
            0/
                image101.jpeg
                image102.jpeg
                ...
  '''
  # Setup path to data folder
  image_path = Path(data_path)
  train_path = image_path + "/train"
  test_path = image_path + "/test"

  # Write transform for image
  data_transform = transform()

  # TODO: Augment training data further.

  # Use ImageFolder to create dataset(s)
  train_dataset = RegressionImageFolder(
    root=train_path, # target folder of images
    image_scores=get_path_to_label_from_path(train_path), # get score
    transform=data_transform) # transforms to perform on data (images)

  validation_dataset = RegressionImageFolder(
    root=test_path,
    image_scores=get_path_to_label_from_path(test_path), # get score
    transform=data_transform)

  test_dataset = RegressionImageFolder(
    root=test_path,
    image_scores=get_path_to_label_from_path(test_path), # get score
    transform=data_transform)

  print(f"Train data:\n{train_dataset}\nValidation data:{validation_dataset}\nTest data:\n{test_dataset}")
  for dataset in [train_dataset, validation_dataset, test_dataset]:
    img, label = dataset[0][0], dataset[0][1]
    print(f"Image tensor:\n{img}")
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")

  return train_dataset, validation_dataset, test_dataset

def build_dataloader(train_dataset, validation_dataset, test_dataset, batch_size, num_workers):
  print(f"Creating DataLoader's with batch size {batch_size} and {num_workers} workers.")

  # Create DataLoader's
  train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)

  validation_dataloader = DataLoader(validation_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers)

  test_dataloader = DataLoader(test_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers)

  return train_dataloader, validation_dataloader, test_dataloader

if __name__ == "__main__":
  DATA_PATH = Path("../")
  BATCH_SIZE = 32
  NUM_WORKERS = os.cpu_count()
  train_dataset, validation_dataset, test_dataset = load_dataset(DATA_PATH, get_resize_transform)
  # train_dataloader, validation_dataloader, test_dataloader = build_dataloader(train_dataset, validation_dataset, test_dataset, BATCH_SIZE, NUM_WORKERS)
