import sys
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision import io
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
import os
torch.manual_seed(0)

# Model
class FCN(nn.Module):
    '''
    # SegNet
    Full convolutional network Class
    Values:
        im_chan: the number of channels of the input image, a scalar
    hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=64):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4),
            self.make_disc_block(hidden_dim * 4, hidden_dim * 8),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding = 1, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN,
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                    (affects activation and batchnorm)
        '''
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Sigmoid()
            )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2,padding = 1, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block,
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                    (affects activation and batchnorm)
        '''
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.Sigmoid()
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        return self.fcn(image)

device = 'cuda'

model_fcn = FCN().to(device)
model_fcn.load_state_dict(torch.load("./model_versions/fcn_ver_5.2.pth"))


def graying_image(image):
    # Convert the PyTorch tensor image to a NumPy array
    image_np = image.permute(1, 2, 0).numpy()
    # Convert the image to grayscale
    if image_np.shape[2] != 1:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        gray_image = torch.from_numpy(gray_image).unsqueeze(0)  # Keep it as a single-channel image

        return gray_image
    return image

def image_preprocessing_pipeline(input_image_path):
    # Load the image
    image = io.read_image(input_image_path)
    # Convert the image to a grayscale image
    gray_image = graying_image(image)
    gray_image = gray_image.float()
    resize = transforms.Resize((1456, 768))
    gray_image = resize(gray_image)
    normalize = transforms.Normalize((0.5,), (0.5,))
    gray_image = normalize(gray_image)
    # Save the processed image
    return gray_image

def image_processing_pipeline(input_image_path, output_image_path):
    # Preprocess the image
    gray_image = image_preprocessing_pipeline(input_image_path)
    # Make the image a batch of one
    gray_image = gray_image.unsqueeze(0).to(device)
    # Make a prediction
    with torch.no_grad():
        pred = model_fcn(gray_image)
    # Convert the prediction to a binary image
    if pred.shape[1] == 1:
            # Convert single-channel prediction to 3 channels (grayscale to RGB)
            pred = torch.cat([pred] * 3, dim=1)
    pred[pred > 0.2] = 1
    pred[pred <= 0.2] = 0
    # Save the prediction
    resize = transforms.Resize((1447, 845))
    pred = resize(pred)
    # save the image
    # get the image name in the input_image_path
    pred_mask_filename = os.path.join(output_image_path, str(input_image_path.split("/")[-1]))
    plt.imsave(pred_mask_filename, np.transpose(pred[0].cpu().numpy(), (1, 2, 0)))


def main(input_image_path, output_image_path):
    try:
        image_processing_pipeline(input_image_path, output_image_path)
        print(f"Image processing completed. Result saved at {output_image_path}")
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_image_path> <output_image_path>")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        main(input_image_path, output_image_path)

