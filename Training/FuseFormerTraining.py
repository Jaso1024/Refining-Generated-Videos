import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from peft import inject_adapter_in_model, LoraConfig, get_peft_model


sys.path.append('Utils')
sys.path.append('Models')
sys.path.append('Weights')
sys.path.append('Data')

from DataGenerator import DataGenerator
from FuseFormer import InpaintGenerator

# Parameters
num_samples = 160
eval_samples = 12
num_epochs = 200
batch_size = 1
learning_rate = 1e-4
accumulation_steps = 4  
mask_type = "random tensor pixelation"
CUDA_LAUNCH_BLOCKING=1
# Define the LoRA configuration
lora_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=32,
    bias="none",
    target_modules=["query_embedding", "key_embedding"],  # specify which modules to target for LoRA
)

if __name__ == "__main__":
    datagen = DataGenerator(
        batch_size=batch_size, 
        length=num_samples,
        window=8, 
        use_sections=False, 
        mask_type=mask_type, 
        mask_shape=(65,65,4),
        pixelation_factor = 5
    )

    # Initialize and inject LoRA into the model
    model = InpaintGenerator().to('cuda')
    model.load_state_dict(torch.load('Weights/FuseFormer/FuseFormer_TensorBlurring_MaskFocused_64x64x4#4.pth', map_location='cuda'))
    #model = get_peft_model(model, lora_config)
    #model.print_trainable_parameters()
    model = model.cuda()
    

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_mse = float('inf')
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=num_samples, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            try:
                for i, (batch_data, batch_labels, inverse_batch_masks, class_names) in enumerate(datagen):
                    batch_data = 2 * batch_data.permute(0, 4, 1, 2, 3).cuda()/255 - 1
                    batch_labels = batch_labels.permute(0, 4, 1, 2, 3).cuda()/255 
                    inverse_batch_masks = inverse_batch_masks.permute(0, 4, 1, 2, 3).cuda()
                    #batch_data[inverse_batch_masks == 1] = 0

                    optimizer.zero_grad()
                    output = model(batch_data)
                    output = (output+1)/2
                    loss = criterion(output.cuda()*inverse_batch_masks, batch_labels.cuda()*inverse_batch_masks)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix({'Loss': loss.item(), 'Running Loss': epoch_loss/(i+1)})
                    pbar.update(1)
            except Exception as e:
                print(e)
                continue
            
        scheduler.step()  # Step the scheduler
            
        if epoch_loss/num_samples < best_val_mse:
            best_val_mse = epoch_loss/num_samples
            torch.save(model.state_dict(), 'Weights/FuseFormer/FuseFormer_TensorBlurring_MaskFocused_64x64x4#4.pth')
            print(f"New best MSE achieved. Saved model weights.")
