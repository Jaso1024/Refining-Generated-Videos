import sys

sys.path.append('Utils')
sys.path.append('Models')
sys.path.append('Weights')

from DataGenerator import DataGenerator
from UnetInpainter import Unet3D
from DegrainerUtils import grainify
from FuseFormer import InpaintGenerator

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
mask_type = ""


if __name__ == "__main__":
    datagen = DataGenerator(
        batch_size=batch_size, 
        length=num_samples, 
        window=16, 
        use_sections=False, 
        mask_type=mask_type, 
        mask_shape=(16, 16, 3),
    )

    model = InpaintGenerator(
            init_weights=True,
            output_size = (16, 16),
    ).to('cuda')


    model.load_state_dict(torch.load('Weights/Degrainer/Fusegrainer_64x64_D03d02.pth'))

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters())

    best_val_mse = float('inf')

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        if epoch % 6 == 0 and epoch != 0:
            model.load_state_dict(torch.load('Weights/Degrainer/Fusegrainer_64x64_D03d02.pth'))
        model.train()
        epoch_loss = 0
        with tqdm(total=num_samples, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for i, (_, batch_labels, _, _) in enumerate(datagen):
                    batch_labels = 2 * batch_labels.permute(0, 4, 1, 2, 3).to('cuda')/255 - 1
                    
                    batch_data = grainify(batch_labels)

                    optimizer.zero_grad()
                    
                    output = model(batch_data)

                    loss = criterion((output+1)/2, (batch_labels+1)/2)

                    loss.backward()

                    optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix({'Loss': loss.item(), 'Running Loss': epoch_loss/(i+1)})
                    pbar.update(1)
            
            scheduler.step()  # Step the scheduler
            
            if epoch_loss/num_samples < best_val_mse:
                best_val_mse = epoch_loss/num_samples
                torch.save(model.state_dict(), 'Weights/Degrainer/Fusegrainer_64x64_D03d02.pth')
                print(f"New best MSE achieved. Saved model weights.")
        


