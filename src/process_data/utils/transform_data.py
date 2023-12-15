from torchvision import transforms

def get_resize_transform():
  # Write transform for image
  data_transform = transforms.Compose([
      # Resize the images to 64x64
      transforms.Resize(size=(64, 64)),
      # Flip the images randomly on the horizontal
      transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
      # Turn the image into a torch.Tensor
      transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
  ])
  return data_transform
