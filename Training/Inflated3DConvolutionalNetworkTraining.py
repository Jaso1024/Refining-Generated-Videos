import sys

sys.path.append('Utils')
sys.path.append('Models')
sys.path.append('Weights')
sys.path.append('Data')

from DataGenerator import DataGenerator
from UnetInpainter import Unet3D
from Inflated3DConvolutionalNetwork import InceptionI3d
from FrechetVideoDistanceUtils import compute_fvd, get_fvd_logits


import torch
import torch.nn as nn
import torch.optim as optim
import os

import os
import numpy as np
import imageio.v3 as iio
import random
import cv2
import torch
import tqdm
import json
import torch.nn.functional as F

VIDEO_EXTENSIONS = [
    'mkv', 'avi', 'mp4'
]

def preprocess(videos, target_resolution=(224,224)):
    # videos in {0, ..., 255} as np.uint8 array
    b, t, h, w, c = videos.shape
    all_frames = videos.flatten(end_dim=1) # (b * t, h, w, c)
    all_frames = all_frames.permute(0, 3, 1, 2).contiguous() # (b * t, c, h, w)
    resized_videos = F.interpolate(all_frames, size=target_resolution,
                                   mode='bilinear', align_corners=False)
    resized_videos = resized_videos.view(b, c, t, *target_resolution)
    output_videos = resized_videos# (b, c, t, *)
    scaled_videos = 2. * output_videos / 255. - 1 # [-1, 1]
    return scaled_videos

class DataGenerator:
    def __init__(
        self, 
        window=16,
        batch_size=2,
        dataset='/UCF101',
        length=1000,
        output_size=128,
        resize_width=128,
        resize_height=128,
        crop_shape = (240, 240)
    ):
        self.dataset = dataset
        self.window = window
        self.batch_size = batch_size
        self.length = length
        self.output_size = output_size
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.crop_shape = crop_shape

        # Precompute the label to integer mapping for the entire dataset
        all_videos = [v for v in os.listdir(f'Data{self.dataset}') if any(v.endswith(ext) for ext in VIDEO_EXTENSIONS)]
        unique_labels = list(set([video.split('_')[1] for video in all_videos]))
        self.label_to_int = {label: i for i, label in enumerate(unique_labels)}

        if os.path.exists('Utils/label_mappings.json'):
            with open('Utils/label_mappings.json', 'r') as file:
                self.label_to_int = json.load(file)
        else:
            # Code to generate label_to_int if the file doesn't exist
            # (e.g., the original code you provided)
            print('Using new class - label mappings, will cause model to have to relearn')
            all_videos = [v for v in os.listdir(f'Data{self.dataset}') if any(v.endswith(ext) for ext in VIDEO_EXTENSIONS)]
            unique_labels = list(set([video.split('_')[1] for video in all_videos]))
            self.label_to_int = {label: i for i, label in enumerate(unique_labels)}


    def __getitem__(self, index):
        if index > self.length:
            raise StopIteration

        batch_videos = []
        video_labels = []

        # Get all video files from the UCF101 dataset directory
        all_videos = [v for v in os.listdir(f'Data{self.dataset}') if any(v.endswith(ext) for ext in VIDEO_EXTENSIONS)]
        
        while len(batch_videos) < self.batch_size:
            video_choice = random.choice(all_videos)
            if video_choice[-3:] in VIDEO_EXTENSIONS:
                batch_videos.append(iio.imiter(f'Data{self.dataset}/{video_choice}'))
                # Extract class label from the video filename
                video_labels.append(self.label_to_int[video_choice.split('_')[1]])

        batch_data = []
        for frame_iterator in batch_videos:
            frame_iterator = list(frame_iterator)
            start_index = random.randint(0, len(frame_iterator) - self.window)
            sampled_frames = frame_iterator[start_index:start_index + self.window]
            sampled_frames = self.crop_frames(sampled_frames)
            sampled_frames = self.resize_frames(sampled_frames)
            batch_data.append(sampled_frames)

        batch_data = np.array(batch_data)
        batch_data = torch.tensor(batch_data, dtype=torch.float32)

        video_labels = torch.tensor(video_labels, dtype=torch.long)

        return batch_data, video_labels

    def resize_frames(self, frames):
        return [cv2.resize(frame, (self.resize_width, self.resize_height)) for frame in frames]

    def crop_frames(self, frames):
        cropped_frames = []
        for frame in frames:
            height, width, _ = frame.shape
            start_x = (width - self.crop_shape[0]) // 2
            start_y = (height - self.crop_shape[1]) // 2
            cropped_frame = frame[start_y:start_y+self.crop_shape[0], start_x:start_x+self.crop_shape[1]]
            cropped_frames.append(cropped_frame)
        return cropped_frames

    def __len__(self):
        pass
# Constants
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 0.00001

# Create synthetic data
datagen = DataGenerator(batch_size=BATCH_SIZE, length=1000)

# Load the I3D model for training
def load_i3d_model(device, num_classes=101):
    i3d = InceptionI3d(num_classes, in_channels=3).to(device)
    return i3d

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = 'cuda' 
i3d_model = load_i3d_model(device)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(i3d_model.parameters(), lr=LEARNING_RATE)

model_dict = i3d_model.state_dict()
pretrained_dict = {k: v for k, v in torch.load('Weights/Inflated3D/best_i3d_model_128_cropped.pt').items() if k in model_dict and model_dict[k].shape == v.shape}
model_dict.update(pretrained_dict)
i3d_model.load_state_dict(model_dict)

best_val_acc = 0.0
last_three_accuracies = [0, 0, 0, 0]  # To keep track of the last three accuracies

for epoch in range(EPOCHS):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    # Training loop
    i3d_model.train()  # Set the model to training mode
    progress_bar = tqdm(enumerate(datagen), total=1000, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    for idx, (batch_data, batch_labels,) in progress_bar:
        # Preprocess data and move to device
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        batch_data = preprocess(batch_data)
 
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = i3d_model(batch_data)
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += batch_labels.size(0)
        correct_train += predicted.eq(batch_labels).sum().item()
        
        # Update the progress bar description
        train_accuracy = 100. * correct_train / total_train
        progress_bar.set_description(f"Epoch {epoch + 1} Loss: {loss.item():.3f} Running Loss: {running_loss / (idx + 1):.3f} Train Acc: {train_accuracy:.2f}%")

    # Check if training accuracy has stagnated
    last_three_accuracies.pop(0)  # Remove the oldest accuracy
    last_three_accuracies.append(train_accuracy)  # Add the latest accuracy


    # Validation loop
    i3d_model.eval()  # Set the model to evaluation mode
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch_data, batch_labels in DataGenerator(batch_size=1, length=200):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            batch_data = preprocess(batch_data)
            
            outputs = i3d_model(batch_data)
            _, predicted = outputs.max(1)
            total_val += batch_labels.size(0)
            correct_val += predicted.eq(batch_labels).sum().item()
    
    val_accuracy = 100. * correct_val / total_val
    print(f"Epoch {epoch + 1} Validation Accuracy: {val_accuracy:.2f}%")

    # Model checkpointing
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(i3d_model.state_dict(), 'Weights/Inflated3D/best_i3d_model_128_cropped.pt')
        print('Higher accuracy than best, saving')

print("Finished Training")
