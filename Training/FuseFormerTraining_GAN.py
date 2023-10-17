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
from FuseFormer import InpaintGenerator, Discriminator
from AdversarialLoss import AdversarialLoss
# Parameters
num_samples = 160
eval_samples = 12
num_epochs = 200
batch_size = 4
learning_rate = 1e-7
mask_type = "random tensor masking"

GAN_LOSS = 'hinge'
ADVERSARIAL_WEIGHT = .01
HOLE_WEIGHT = .8
VALID_WEIGHT = 1.2

if __name__ == "__main__":
    datagen = DataGenerator(
        batch_size=batch_size, 
        length=num_samples,
        window=8, 
        use_sections=False, 
        mask_type=mask_type, 
        mask_shape=(16,16,4),
        pixelation_factor = 3,
        output_size=64,
        resize_width=64,
        resize_height=64,
    )

    generator = InpaintGenerator(
            init_weights=True,
            channel = 256,
            hidden = 1024,
            stack_num = 16,
            num_head = 8,
            kernel_size = (7, 7),
            padding = (3, 3),
            stride = (3, 3),
            output_size = (16, 16),
            blocks = [],
            dropout = 0.,
            n_vecs = 1
    ).to('cuda')
    generator = generator.cuda()
    
    adversarial_loss = AdversarialLoss(GAN_LOSS)
    adversarial_loss = adversarial_loss.cuda()

    l1_loss = nn.L1Loss()
    l1_loss = l1_loss.cuda()

    discriminator = Discriminator(in_channels=3, use_sigmoid=GAN_LOSS != 'hinge')
    discriminator = discriminator.cuda()

    generator.load_state_dict(torch.load('Weights/FuseFormer/GanTrained/FuseFormer_Generator_64x64x3#4TensorMasking', map_location='cuda'))
    discriminator.load_state_dict(torch.load('Weights/FuseFormer/GanTrained/FuseFormer_Discriminator_64x64x3#4TensorMasking', map_location='cuda'))

    #model_dict = generator.state_dict()
    #pretrained_dict = {k: v for k, v in torch.load('Weights/FuseFormer/PretrainedFuseFormer.pth').items() if k in model_dict and model_dict[k].shape == v.shape}
    #model_dict.update(pretrained_dict)
    #generator.load_state_dict(model_dict)

    generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0,.99))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    generator_scheduler = CosineAnnealingLR(generator_optimizer, T_max=num_epochs)
    discriminator_scheduler = CosineAnnealingLR(discriminator_optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        epoch_loss = 0
        

        with tqdm(total=num_samples, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            
                for i, (batch_data, batch_labels, inverse_batch_masks, class_names) in enumerate(datagen):
                    dis_loss = 0
                    gen_loss = 0

                    generator_optimizer.zero_grad()
                    discriminator_optimizer.zero_grad()
                    
                    batch_data = 2 * batch_data.permute(0, 4, 1, 2, 3).cuda()/255 - 1
                    batch_labels = 2 * batch_labels.permute(0, 4, 1, 2, 3).cuda()/255 - 1 
                    inverse_batch_masks = inverse_batch_masks.permute(0, 4, 1, 2, 3).cuda()

                    batch_data[inverse_batch_masks == 1] = 0

                    fake_image = generator(batch_data)

                    real_vid_feat = discriminator(batch_labels)
                    fake_vid_feat = discriminator(fake_image.detach())

                    dis_real_loss = adversarial_loss(real_vid_feat, True, True)
                    dis_fake_loss = adversarial_loss(fake_vid_feat, False, True)
                    dis_loss += (dis_real_loss + dis_fake_loss) / 2

                    dis_loss.backward()
                    discriminator_optimizer.step()

                    gen_vid_feat = discriminator(fake_image)
                    gan_loss = adversarial_loss(gen_vid_feat, True, False)
                    gan_loss = gan_loss * ADVERSARIAL_WEIGHT
                    gen_loss += gan_loss

                    # generator l1 loss
                    hole_loss = l1_loss(fake_image*(1-inverse_batch_masks), batch_labels*(1-inverse_batch_masks))
                    hole_loss = hole_loss / torch.mean(1-inverse_batch_masks) * HOLE_WEIGHT
                    gen_loss += hole_loss 

                    valid_loss = l1_loss(fake_image*inverse_batch_masks, batch_labels*inverse_batch_masks)
                    valid_loss = valid_loss / ((torch.mean(inverse_batch_masks) * VALID_WEIGHT) + 1e-8)
                    gen_loss += valid_loss 
                    
                    gen_loss.backward()
                    generator_optimizer.step()

                    epoch_loss += gen_loss.item()
                    pbar.set_postfix({'Gen Loss': gen_loss.item(), 'Dis Loss': dis_loss.item(), 'Running Loss': epoch_loss/(i+1)})
                    pbar.update(1)
            
            
                torch.save(generator.state_dict(), 'Weights/FuseFormer/GanTrained/FuseFormer_Generator_64x64x3#4TensorMasking')
                torch.save(discriminator.state_dict(), 'Weights/FuseFormer/GanTrained/FuseFormer_Discriminator_64x64x3#4TensorMasking')
         
