import sys

sys.path.append('Utils')
sys.path.append('Models')
sys.path.append('Weights')

from DataGenerator import DataGenerator
from AugmentedUnetInpainter import Unet3D

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary


from tqdm import tqdm
import numpy as np

# Parameters
num_samples = 320
eval_samples = 12
num_epochs = 200
batch_size = 1
learning_rate = 0.0001
accumulation_steps = 4  
mask_type = "random tensor blurring"


if __name__ == "__main__":
    datagen = DataGenerator(
        batch_size=batch_size, 
        length=num_samples, 
        window=16, 
        use_sections=False, 
        mask_type=mask_type, 
        mask_shape=(16, 16, 3)
    )

    model = Unet3D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
    ).to('cuda')

    model.load_state_dict(torch.load('Weights/UnetV2BestWeights_sigmoid_pixelation.pth'), strict=False)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters())

    best_val_mse = float('inf')

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        if epoch % 6 == 0 and epoch != 0:
            model.load_state_dict(torch.load('Weights/UnetV2BestWeights_GaussianBlur_128.pth'))
        model.train()
        epoch_loss = 0
        with tqdm(total=num_samples, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for i, (batch_data, batch_labels, inverse_batch_masks, class_names) in enumerate(datagen):
                    batch_data = batch_data.permute(0, 4, 1, 2, 3).to('cuda')/255
                    batch_labels = batch_labels.permute(0, 4, 1, 2, 3).to('cuda')/255
                    inverse_batch_masks = inverse_batch_masks.permute(0, 4, 1, 2, 3).to('cuda')

                    optimizer.zero_grad()
                    output = model(batch_data)

                    loss = criterion(output*255*inverse_batch_masks, batch_labels*255*inverse_batch_masks)*100

                    loss.backward()

                    optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix({'Loss': loss.item(), 'Running Loss': epoch_loss/(i+1)})
                    pbar.update(1)
            
            scheduler.step()  # Step the scheduler
            
            if epoch_loss/num_samples < best_val_mse:
                best_val_mse = epoch_loss/num_samples
                torch.save(model.state_dict(), 'Weights/UnetV2BestWeights_GaussianBlur_128.pth')
                print(f"New best MSE achieved. Saved model weights.")
        


