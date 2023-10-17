import sys

sys.path.append('Utils')
sys.path.append('Models')
sys.path.append('Weights')

from DataGenerator import DataGenerator
from UnetInpainter import Unet3D

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
learning_rate = 0.000001
accumulation_steps = 4  
mask_type = None

ucf101_prompts = {
    "Apply Eye Makeup": "A person applying eye makeup.",
    "Apply Lipstick": "A woman putting on lipstick.",
    "Archery": "An archer aiming a bow and arrow.",
    "Baby Crawling": "A baby crawling on the floor.",
    "Balance Beam": "A gymnast performing on a balance beam.",
    "Band Marching": "A marching band parading on a street.",
    "Baseball Pitch": "A baseball player pitching the ball.",
    "Basketball": "A player shooting a basketball into the hoop.",
    "Basketball Dunk": "A basketball player performing a slam dunk.",
    "Bench Press": "An individual lifting weights on a bench press.",
    "Biking": "A person riding a bicycle.",
    "Billiards": "A player taking a shot in billiards.",
    "Blow Dry Hair": "Someone blow-drying their hair.",
    "Blowing Candles": "A person blowing out candles on a birthday cake.",
    "Body Weight Squats": "An individual doing bodyweight squats.",
    "Bowling": "A bowler releasing the ball down the lane.",
    "Boxing Punching Bag": "A boxer practicing punches on a punching bag.",
    "Boxing Speed Bag": "A boxer hitting a speed bag.",
    "Breaststroke": "A swimmer performing the breaststroke.",
    "Brushing Teeth": "A person brushing their teeth.",
    "Clean and Jerk": "A weightlifter performing the clean and jerk.",
    "Cliff Diving": "A diver jumping off a cliff into water.",
    "Cricket Bowling": "A cricket bowler delivering the ball.",
    "Cricket Shot": "A batsman playing a shot in cricket.",
    "Cutting In Kitchen": "A chef cutting vegetables in the kitchen.",
    "Diving": "A diver performing a dive into a pool.",
    "Drumming": "A drummer playing the drums.",
    "Fencing": "Two fencers dueling with their swords.",
    "Field Hockey Penalty": "A field hockey player taking a penalty shot.",
    "Floor Gymnastics": "A gymnast performing a routine on the floor.",
    "Frisbee Catch": "A person catching a flying frisbee.",
    "Front Crawl": "A swimmer doing the front crawl stroke.",
    "Golf Swing": "A golfer taking a swing.",
    "Haircut": "A barber giving a haircut to a customer.",
    "Hammer Throw": "An athlete throwing a hammer in a field event.",
    "Hammering": "A person hammering a nail into wood.",
    "Handstand Pushups": "An individual doing handstand push-ups.",
    "Handstand Walking": "A gymnast walking on their hands.",
    "Head Massage": "A person receiving a head massage.",
    "High Jump": "An athlete performing a high jump.",
    "Horse Race": "Jockeys racing their horses on a track.",
    "Horse Riding": "A person riding a horse in a field.",
    "Hula Hoop": "An individual twirling a hula hoop around their waist.",
    "Ice Dancing": "Ice skaters performing a dance routine on ice.",
    "Javelin Throw": "An athlete throwing a javelin.",
    "Juggling Balls": "A juggler juggling multiple balls.",
    "Jump Rope": "A person skipping with a jump rope.",
    "Jumping Jack": "An individual doing jumping jacks.",
    "Kayaking": "A person paddling in a kayak.",
    "Knitting": "Someone knitting with yarn and needles.",
    "Long Jump": "An athlete performing a long jump.",
    "Lunges": "An individual doing lunges.",
    "Military Parade": "Soldiers marching in a military parade.",
    "Mixing": "A person mixing batter in a bowl.",
    "Mopping Floor": "Someone mopping a floor.",
    "Nun chucks": "A martial artist practicing with nun chucks.",
    "Parallel Bars": "A gymnast performing on parallel bars.",
    "Pizza Tossing": "A chef tossing pizza dough in the air.",
    "Playing Guitar": "A musician strumming a guitar.",
    "Playing Piano": "A pianist playing a piano.",
    "Playing Tabla": "A musician playing the tabla drums.",
    "Playing Violin": "A violinist playing a violin.",
    "Playing Cello": "A cellist playing a cello.",
    "Playing Daf": "A musician playing the daf drum.",
    "Playing Dhol": "A drummer playing the dhol.",
    "Playing Flute": "A flutist playing a flute.",
    "Playing Sitar": "A musician playing a sitar.",
    "Pole Vault": "An athlete performing a pole vault.",
    "Pommel Horse": "A gymnast performing on a pommel horse.",
    "Pull Ups": "An individual doing pull-ups on a bar.",
    "Punch": "A boxer throwing a punch.",
    "Push Ups": "A person doing push-ups.",
    "Rafting": "People rafting down a river.",
    "Rock Climbing Indoor": "A person indoor rock climbing.",
    "Rope Climbing": "An individual climbing a rope.",
    "Rowing": "Athletes rowing a boat.",
    "Salsa Spin": "Dancers performing salsa spins.",
    "Shaving Beard": "A man shaving his beard.",
    "Shotput": "An athlete throwing a shot put.",
    "Skate Boarding": "A skateboarder performing a trick.",
    "Skiing": "A skier going down a snowy slope.",
    "Skijet": "A person riding a skijet on water.",
    "Sky Diving": "A skydiver free-falling from an airplane.",
    "Soccer Juggling": "A soccer player juggling a soccer ball.",
    "Soccer Penalty": "A soccer player taking a penalty kick.",
    "Still Rings": "A gymnast performing on still rings.",
    "Sumo Wrestling": "Sumo wrestlers in a match.",
    "Surfing": "A surfer riding a wave.",
    "Swing": "Children swinging on a playground swing.",
    "Table Tennis Shot": "A player making a shot in table tennis.",
    "Tai Chi": "Individuals practicing Tai Chi.",
    "Tennis Swing": "A tennis player taking a swing.",
    "Throw Discus": "An athlete throwing a discus.",
    "Trampoline Jumping": "A person jumping on a trampoline.",
    "Typing": "Someone typing on a keyboard.",
    "Uneven Bars": "A gymnast performing on uneven bars.",
    "Volleyball Spiking": "A volleyball player spiking the ball.",
    "Walking with dog": "A person taking a walk with their dog.",
    "Wall Pushups": "An individual doing push-ups against a wall.",
    "Writing On Board": "A teacher writing on a chalkboard.",
    "Yo Yo": "A person playing with a yoyo.",
}

ucf101_prompts = {
    key.replace(" ", "").lower(): key
    for key, value in ucf101_prompts.items()
}

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
        dim = 128,
        dim_mults = (1, 2, 4, 8),
        use_bert_text_cond = True,
    ).to('cuda')

    model.load_state_dict(torch.load('Weights/UnetV2BestWeights_TextConditioned.pth'))

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters())

    best_val_mse = 800

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        if epoch % 6 == 0 and epoch != 0:
            model.load_state_dict(torch.load('Weights/UnetV2BestWeights_TextConditioned.pth'))
        model.train()
        epoch_loss = 0
        with tqdm(total=num_samples, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for i, (batch_data, batch_labels, inverse_batch_masks, class_names) in enumerate(datagen):
                    batch_data = batch_data.permute(0, 4, 1, 2, 3).to('cuda')/255
                    batch_labels = batch_labels.permute(0, 4, 1, 2, 3).to('cuda')/255
                    inverse_batch_masks = inverse_batch_masks.permute(0, 4, 1, 2, 3).to('cuda')

                    optimizer.zero_grad()
                    print([ucf101_prompts[class_name.lower()] for class_name in class_names])
                    output = model(x=batch_data, cond=[ucf101_prompts[class_name.lower()] for class_name in class_names])

                    loss = criterion(output*255, batch_labels*255)

                    loss.backward()

                    optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix({'Loss': loss.item(), 'Running Loss': epoch_loss/(i+1)})
                    pbar.update(1)
            
            scheduler.step()  # Step the scheduler
            
            if epoch_loss/num_samples < best_val_mse:
                best_val_mse = epoch_loss/num_samples
                torch.save(model.state_dict(), 'Weights/UnetV2BestWeights_TextConditioned.pth')
                print(f"New best MSE achieved. Saved model weights.")
        


