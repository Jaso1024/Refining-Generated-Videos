import os
import numpy as np
import imageio.v3 as iio
import random
import cv2
from skimage.feature import canny
import torch


VIDEO_EXTENSIONS = [
    'mkv', 'avi', 'mp4'
]

class DataGenerator:
    def __init__(
        self, 
        window=16,
        batch_size=2,
        dataset='/UCF101',
        dataset2='/2Iters16x16',
        length=1000,
        output_size=64,
        mask_type=None,
        mask_shape=(8, 8, 3),
        num_masks=4,
        resize_width=64,
        resize_height=64,
        section_width=32,
        section_height=32,
        use_sections=False,
        return_name=False,
        use_dual_datasets=False,
        selected_class=None,
        pixelation_factor = 3,
        removal=True,
        crop_shape = (240, 240)
    ):
        self.dataset = dataset
        self.dataset2 = dataset2
        self.window = window
        self.batch_size = batch_size
        self.length = length
        self.output_size = output_size
        self.mask_type = mask_type
        self.mask_shape = mask_shape
        self.num_masks = num_masks
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.section_width = section_width
        self.section_height = section_height
        self.use_sections = use_sections
        self.use_dual_datasets = use_dual_datasets
        self.selected_class = selected_class
        self.pixelation_factor = pixelation_factor
        self.removal = removal
        self.crop_shape = crop_shape

    def __getitem__(self, index):

        if index > self.length:
            raise StopIteration

        batch_videos = []
        video_names = []

        all_videos = [v for v in os.listdir(f'Data{self.dataset}')]
        
        # If dual dataset loading is enabled, load videos from the second dataset as well
        if self.use_dual_datasets and self.dataset2:
            all_videos2 = os.listdir(f'Data{self.dataset2}') 
        
        while len(batch_videos) < self.batch_size:
            video_choice = random.choice(all_videos)
            if video_choice[-3:] in VIDEO_EXTENSIONS:
                batch_videos.append(iio.imiter(f'Data{self.dataset}/{video_choice}'))
                video_names.append(video_choice.split('_')[1])
                if self.removal:
                    all_videos.remove(video_choice)
            
            # If dual dataset loading is enabled, alternate between the two datasets
            if self.use_dual_datasets and self.dataset2 and len(batch_videos) < self.batch_size:
                all_videos2 = [v for v in os.listdir(f'Data{self.dataset2}') if v.replace(' ','').split('_')[1].lower() == video_names[-1].lower()]
                video_choice2 = random.choice(all_videos2)

                if video_choice2[-3:] in VIDEO_EXTENSIONS:
                    batch_videos.append(iio.imiter(f'Data{self.dataset2}/{video_choice2}'))
                    video_names.append(video_choice2.split('_')[1])


        if self.use_sections:
            sectioned_truths = [[] for _ in range((self.resize_width // self.section_width) * (self.resize_height // self.section_height))]
            sectioned_masks = [[] for _ in range((self.resize_width // self.section_width) * (self.resize_height // self.section_height))]
            sectioned_true_truths = [[] for _ in range((self.resize_width // self.section_width) * (self.resize_height // self.section_height))]

        else:
            sectioned_truths = []
            sectioned_masks = []
            sectioned_true_truths = []

        for frame_iterator in batch_videos:
            frame_iterator = list(frame_iterator)
            start_index = random.randint(0, len(frame_iterator) - self.window)
            sampled_frames = frame_iterator[start_index:start_index + self.window]
            sampled_frames = self.crop_frames(sampled_frames)
            sampled_frames = self.resize_frames(sampled_frames)

            if self.use_sections:
                sections = self.split_into_sections(sampled_frames)
                true_truths = sections
                masks, sections = zip([self.create_mask(np.array(section, dtype=np.float32), self.mask_type) for section in sections])
                    
                for i, (truth_section, mask_section, true_truths) in enumerate(zip(sections, masks, true_truths)):
                    sectioned_truths[i].append(truth_section)
                    sectioned_masks[i].append(mask_section)
                    sectioned_true_truths[i].append(true_truths)
            else:
                true_truths = sampled_frames
                masks, sampled_frames = self.create_mask(np.array(sampled_frames, dtype=np.float32), self.mask_type)
                sectioned_truths.append(sampled_frames)
                sectioned_masks.append(masks)
                sectioned_true_truths.append(true_truths)
        try:
            sectioned_truths = np.array(sectioned_truths)
            sectioned_masks = np.array(sectioned_masks)
            sectioned_true_truths = np.array(sectioned_true_truths)

            if self.mask_type is not None:
                if 'mask' in self.mask_type:
                    masked_batches = [truths * masks for truths, masks in zip(sectioned_truths, sectioned_masks)]
                    for masked_batch in masked_batches:
                        masked_batch[masked_batch == 0] = 0 #255 for unet inpainter
                else:
                    masked_batches = sectioned_truths
            else:
                masked_batches = [truths * masks for truths, masks in zip(sectioned_truths, sectioned_masks)]
                for masked_batch in masked_batches:
                    masked_batch[masked_batch == 0] = 0 #255 for unet inpainter

            masked_batches = np.array(masked_batches)
            sectioned_truths = np.array(sectioned_true_truths)

            masked_batches = torch.tensor(masked_batches, dtype=torch.float32)
            sectioned_truths = torch.tensor(sectioned_truths, dtype=torch.float32)
            sectioned_masks = torch.tensor(np.array([1 - masks for masks in sectioned_masks]), dtype=torch.float32)

            return masked_batches, sectioned_truths, sectioned_masks, video_names
        except Exception as e:
            print(e)
            raise IndexError

    def resize_frames(self, frames):
        #frames = [cv2.resize(frame, (32,32)) for frame in frames]
        #frames = [cv2.resize(frame, (64, 64)) for frame in frames]
        return [cv2.resize(frame, (self.resize_width, self.resize_height)) for frame in frames]
        #return [self.resize_with_bars(frame) for frame in frames]

    def resize_with_bars(self, img_array):
        
        # Create a new black image array with the desired size
        new_img_array = np.zeros((240, 432, 3), dtype=np.uint8)
        
        # Calculate the starting x-coordinate to center the image
        start_x = (432 - 240) // 2
        
        # Place the original image array into the center of the new image array
        new_img_array[:, start_x:start_x+240] = img_array
        
        return new_img_array
    
    def crop_frames(self, frames):
        cropped_frames = []
        for frame in frames:
            height, width, _ = frame.shape
            start_x = (width - self.crop_shape[0]) // 2
            start_y = (height - self.crop_shape[1]) // 2
            cropped_frame = frame[start_y:start_y+self.crop_shape[0], start_x:start_x+self.crop_shape[1],]
            cropped_frames.append(cropped_frame)
        return cropped_frames


    def split_into_sections(self, frame):
        sections = []
        for i in range(0, self.resize_width, self.section_width):
            for j in range(0, self.resize_height, self.section_height):
                section = np.array(frame)[:,i:i+self.section_width, j:j+self.section_height]
                sections.append(section)
        return sections
    
    def create_mask(self, truth, mask_type):
        # mask should never be more than 50% of the video
        mask = np.ones_like(truth)
        if mask_type == 'sequence length tensor masking':
            center_x = random.choice(range(truth.shape[-3]//5, 4*(truth.shape[-3]//5)))
            center_y = random.choice(range(truth.shape[-2]//5, 4*(truth.shape[-2]//5)))
            mask[
            :,
            center_x-self.mask_shape[0]//2: center_x+self.mask_shape[0]//2,
            center_y-self.mask_shape[1]//2: center_y+self.mask_shape[1]//2,
            :
            ] = 0
        elif mask_type == 'random masking':
            # randomly mask pixels all over the video tensor
            num_frames, height, width, num_channels = truth.shape
            num_pixels_to_mask = np.prod(truth.shape) // 8  # 12.5% of the video

            # Generate random indices for the frames, rows, columns, and channels
            frame_indices = torch.randint(0, num_frames, (num_pixels_to_mask,))
            row_indices = torch.randint(0, height, (num_pixels_to_mask,))
            col_indices = torch.randint(0, width, (num_pixels_to_mask,))

            # Set the selected pixels to -1
            mask[frame_indices, row_indices, col_indices, :] = 0
        elif mask_type == 'random matrix masking':
                for j in range(truth.shape[0]):
                    center_x = random.choice(range(truth.shape[-3]//5, 4*(truth.shape[-3]//5)))
                    center_y = random.choice(range(truth.shape[-2]//5, 4*(truth.shape[-2]//5)))
                    mask[
                    j,
                    center_x-self.mask_shape[0]//2: center_x+self.mask_shape[0]//2,
                    center_y-self.mask_shape[1]//2: center_y+self.mask_shape[1]//2,
                    :
                    ] = 0
        elif mask_type == 'random tensor masking':
            for _ in range(self.num_masks):
                center_x = random.choice(range(truth.shape[-3]//5, 4*(truth.shape[-3]//5)))
                center_y = random.choice(range(truth.shape[-2]//5, 4*(truth.shape[-2]//5)))
                center_z = random.choice(range(1, truth.shape[-4]-1))
                mask[
                center_z-(self.mask_shape[2]//2): center_z+(self.mask_shape[2]//2),
                center_x-self.mask_shape[0]//2: center_x+self.mask_shape[0]//2,
                center_y-self.mask_shape[1]//2: center_y+self.mask_shape[1]//2,
                :
                ] = 0
        elif mask_type == 'random tensor blurring':
            for _ in range(self.num_masks):
                center_x = random.choice(range(truth.shape[-3]//5, 4*(truth.shape[-3]//5)))
                center_y = random.choice(range(truth.shape[-2]//5, 4*(truth.shape[-2]//5)))
                center_z = random.choice(range(1, truth.shape[-4]-1))

                # Extracting the region to be blurred
                region = truth[
                    center_z-self.mask_shape[2]//2: center_z+self.mask_shape[2]//2,
                    center_x-self.mask_shape[0]//2: center_x+self.mask_shape[0]//2,
                    center_y-self.mask_shape[1]//2: center_y+self.mask_shape[1]//2
                ].copy()

                # Iterating through the z-dimension (3rd dimension) to apply 2D blur slice-by-slice
                for i in range(region.shape[0]):
                    region[i] = cv2.GaussianBlur(region[i], (self.mask_shape[0], self.mask_shape[1]), 0)

                # Assigning the blurred region back to the truth tensor
                truth[
                    center_z-self.mask_shape[2]//2: center_z+self.mask_shape[2]//2,
                    center_x-self.mask_shape[0]//2: center_x+self.mask_shape[0]//2,
                    center_y-self.mask_shape[1]//2: center_y+self.mask_shape[1]//2
                ] = region

                mask[
                    center_z-self.mask_shape[2]//2: center_z+self.mask_shape[2]//2,
                    center_x-self.mask_shape[0]//2: center_x+self.mask_shape[0]//2,
                    center_y-self.mask_shape[1]//2: center_y+self.mask_shape[1]//2
                ] = 0
        elif mask_type == 'random tensor pixelation':
            for _ in range(self.num_masks):
                center_x = random.choice(range(truth.shape[-3]//5, 4*(truth.shape[-3]//5)))
                center_y = random.choice(range(truth.shape[-2]//5, 4*(truth.shape[-2]//5)))
                center_z = random.choice(range(1, truth.shape[-4]-1))
                
                # Determine mask region
                z_start, z_end = center_z-self.mask_shape[2]//2, center_z+self.mask_shape[2]//2
                x_start, x_end = center_x-self.mask_shape[0]//2, center_x+self.mask_shape[0]//2
                y_start, y_end = center_y-self.mask_shape[1]//2, center_y+self.mask_shape[1]//2
                
                # Pixelation using downscaling and upscaling
                pixel_size = self.pixelation_factor  # Adjust this value as per the desired level of pixelation

                for z in range(z_start, z_end):  # Looping over depth can be adjusted based on your needs
                    region = truth[z, x_start:x_end, y_start:y_end, :]

                    # Downscale
                    downscaled = cv2.resize(region, (region.shape[1]//pixel_size, region.shape[0]//pixel_size), interpolation=cv2.INTER_LINEAR)

                    # Upscale
                    upscaled = cv2.resize(downscaled, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)

                    truth[z, x_start:x_end, y_start:y_end, :] = upscaled

                    mask[z, x_start:x_end, y_start:y_end, :] = 0

        elif mask_type == 'optical masking':
            # randomly mask the first frame with a mask matrix, then use optical flow maps to move the mask with the pixels for subsequent frames in the video
            # Note: This is a placeholder implementation. You will need to implement optical flow maps to move the mask with the pixels for subsequent frames.
            mask_x = random.randint(0, truth.shape[-1] - self.mask_shape[0])
            mask_y = random.randint(0, truth.shape[-2] - self.mask_shape[1])
            mask[0, mask_x:mask_x+self.mask_shape[0], mask_y:mask_y+self.mask_shape[1], :] = 0
        elif mask_type is None:
            # randomly select a mask type and use it
            mask_types = ['sequence length tensor masking', 'random masking', 'random matrix masking', 'random tensor masking']
            mask_type = random.choice(mask_types)
            return self.create_mask(truth, mask_type)
        return mask, truth
    
    def remask(self, batch_videos):
        """
        Remasks a batch of videos based on the chosen masking type.

        Parameters:
        - batch_videos: A batch of videos to be remasked.
        - mask_type: The type of mask to be applied.

        Returns:
        - masked_videos: The remasked videos.
        """
        masked_videos = []
        for video in batch_videos:
            _, masked_video = self.create_mask(video, self.mask_type)
            masked_videos.append(masked_video)
        return torch.tensor(masked_videos, dtype=torch.float32)

    def iterative_masking(self, video_tensor, mask_tensor, stride=1):
        """
        Iteratively masks the video tensor by sliding the mask tensor over it.

        Parameters:
        - video_tensor: The video tensor to be masked.
        - mask_tensor: The tensor mask to be applied.
        - stride: The number of pixels the mask moves in each iteration.

        Returns:
        - masked_videos: A list of video tensors, each with the mask applied at a different position.
        """
        # Get the dimensions of the video tensor and the mask tensor
        video_height, video_width = video_tensor.shape[1:3]
        mask_height, mask_width = mask_tensor.shape[1:3]

        # List to store the masked videos
        masked_videos = []

        # Slide the mask over the video tensor
        for i in range(0, video_height - mask_height + 1, stride):
            for j in range(0, video_width - mask_width + 1, stride):
                # Create a copy of the video tensor to apply the current mask
                current_masked_video = video_tensor.copy()
                
                # Apply the mask at the current position
                current_masked_video[:, i:i+mask_height, j:j+mask_width, :] *= mask_tensor
                
                # Append the masked video to the list
                masked_videos.append(current_masked_video)

        return masked_videos




    def __len__(self):
        pass

import matplotlib.pyplot as plt

def visualize_data(masked_batch, truths):
    # Convert the tensors to numpy arrays
    masked_batch = masked_batch.numpy()
    truths = truths.numpy()

    # Get the first video in the batch
    masked_video = masked_batch[0]
    truth_video = truths[0]

    # Get the first frame of the video
    masked_frame = masked_video[0]
    truth_frame = truth_video[0]

    masked_frame[masked_frame < 0] = 0

    # Convert the frames to the range [0, 1]
    masked_frame = (masked_frame - masked_frame.min()) / (masked_frame.max() - masked_frame.min())
    truth_frame = (truth_frame - truth_frame.min()) / (truth_frame.max() - truth_frame.min())

    # Plot the frames
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(masked_frame)
    plt.title('Masked Frame')

    plt.subplot(1, 2, 2)
    plt.imshow(truth_frame)
    plt.title('Truth Frame')

    plt.show()

import time

def test_generation_time(datagen, num_trials=10):
    total_time = 0

    for _ in range(num_trials):
        start_time = time.time()
        datagen.__getitem__(0)
        end_time = time.time()
        total_time += (end_time - start_time)

    average_time = total_time / num_trials
    return average_time

if __name__ == '__main__':
    mask_types = ['sequence length tensor masking', 'random masking', 'random matrix masking', 'random tensor masking']
    
    for mask_type in mask_types:
        #datagen = DataGenerator(mask_type=mask_type)
        #avg_time = test_generation_time(datagen)
        #print(f'Mask Type: {mask_type}, Average Time: {avg_time:.4f} seconds')

        datagen = DataGenerator(mask_type=mask_type, dataset='/SudoData')
        masked_batch, truths, _, _ = datagen.__getitem__(0)
        print(f'Mask Type: {mask_type}')
        visualize_data(masked_batch, truths)
