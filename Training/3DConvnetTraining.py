import sys

sys.path.append('Utils')
sys.path.append('Models')
sys.path.append('Weights')

from Generate3DConvModel import generate_model
from Conv3DNetworkUtils import parse_opts

from DataGenerator import DataGenerator
from FuseFormer import InpaintGenerator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary
import os
import numpy as np
import imageio.v3 as iio
import random
import cv2
import torch
import tqdm
import json
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

# Parameters
num_samples = 320
eval_samples = 12
num_epochs = 200
batch_size = 8
learning_rate = 0.0001

opt = parse_opts()
opt.sample_size = 112
opt.sample_duration = 16
opt.n_classes = 101

VIDEO_EXTENSIONS = [
    'mkv', 'avi', 'mp4'
]

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


# ... [rest of the imports and class definitions remain unchanged]

if __name__ == "__main__":
    datagen = DataGenerator(
        batch_size=batch_size, 
        length=num_samples, 
        window=16, 
    )

    # Create a validation data generator
    val_datagen = DataGenerator(
        batch_size=batch_size, 
        length=eval_samples, 
        window=16, 
    )

    model = generate_model(opt).to('cuda')
    model.load_state_dict(torch.load('Weights/Conv3D/Conv3D_resnext101_UCF101.pth'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_acc = 0.0
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        if epoch % 6 == 0 and epoch != 0:
            model.load_state_dict(torch.load('Weights/Conv3D/Conv3D_resnext101_UCF101.pth'))
        
        model.train()
        epoch_loss = 0
        correct_train = 0
        total_train = 0

        with tqdm(total=num_samples, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for i, (batch_data, batch_labels) in enumerate(datagen):
                batch_data = batch_data.permute(0, 4, 1, 2, 3).to('cuda')

                normalization_vector = torch.tensor([1, 1, 1], dtype=torch.float32).to('cuda')
                normalization_vector = normalization_vector[None, :, None, None, None]  # Expand dimensions to match batch_labels
                batch_data = torch.div(batch_data, normalization_vector)  # Element-wise division


                optimizer.zero_grad()
                output = model(batch_data)
                loss = criterion(output, batch_labels.to('cuda'))
                loss.backward()
                optimizer.step()

                _, predicted = output.max(1)
                total_train += batch_labels.size(0)
                correct_train += predicted.eq(batch_labels.to('cuda')).sum().item()

                epoch_loss += loss.item()
                train_accuracy = 100. * correct_train / total_train
                pbar.set_postfix({'Loss': loss.item(), 'Running Loss': epoch_loss/(i+1), 'Train Acc': train_accuracy})
                pbar.update(1)

            scheduler.step()  # Step the scheduler

            # Validation loop
            model.eval()
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for batch_data, batch_labels in val_datagen:
                    batch_data = batch_data.permute(0, 4, 1, 2, 3).to('cuda')
                    normalization_vector = torch.tensor([1, 1, 1], dtype=torch.float32).to('cuda')
                    normalization_vector = normalization_vector[None, :, None, None, None]  # Expand dimensions to match batch_labels
                    batch_data = torch.div(batch_data, normalization_vector)  # Element-wise division
                    outputs = model(batch_data)
                    _, predicted = outputs.max(1)
                    total_val += batch_labels.size(0)
                    correct_val += predicted.eq(batch_labels.to('cuda')).sum().item()

            val_accuracy = 100. * correct_val / total_val
            print(f"Epoch {epoch + 1} Validation Accuracy: {val_accuracy:.2f}%")

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(model.state_dict(), 'Weights/Conv3D/Conv3D_resnext101_UCF101.pth')
                print(f"New best Validation Accuracy achieved. Saved model weights.")
