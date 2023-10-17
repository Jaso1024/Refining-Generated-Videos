import sys

sys.path.append('Utils')
sys.path.append('Models')
sys.path.append('Weights')

from DataGenerator import DataGenerator
from AugmentedUnetInpainter import Unet3D
from FuseFormer import InpaintGenerator

import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np

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

# Load model
model = InpaintGenerator(
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

model.load_state_dict(torch.load('Weights/FuseFormer/GanTrained/FuseFormer_Generator_64x64x3#4TensorMasking'))
model.eval()

MASK_TYPE = "random tensor masking"
NUM_MASKS = 4
RESOLUTION_y = 64
RESOLUTION_x = 64
FORCED_CONSISTENCY = False
MASK_SHAPE = (16,16,3)
RESOLUTION=64
dataset = f'/FuseFormer/TensorMasking/{RESOLUTION_x}x{RESOLUTION_y}/NoForcedConsistency/{1}Iters{MASK_SHAPE[0]}x{MASK_SHAPE[1]}x{MASK_SHAPE[2]}#{NUM_MASKS}'
    

# Create synthetic data
datagen = DataGenerator(
    batch_size=1, 
    length=10, 
    dataset='/UCF101',
    mask_type='random tensor masking', 
    mask_shape=(16,16,3),
    resize_width=64,
    resize_height=64,
)  

# Visualization function
def visualize_videos(original, inpainted, ground_truth):
    num_frames = original.shape[1]
    
    # Create a window for visualization
    cv2.namedWindow('Visualization', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Visualization', 900, 300)  # Adjust window size as needed

    for i in range(num_frames):
        # Concatenate the frames horizontally
        combined_frame = np.hstack([
            original[0, i, :, :, :], 
            inpainted[0, i, :, :, :], 
            ground_truth[0, i, :, :, :]
        ])
        
        # Convert the frame from [0, 1] range to [0, 255] for visualization
        combined_frame = (combined_frame).astype(np.uint8)
        
        # Display the concatenated frame
        cv2.imshow('Visualization', combined_frame)
        
        # Wait for 10ms between frames
        cv2.waitKey(200)

    # Close the OpenCV window
    cv2.destroyAllWindows()

def save_video(frames, filename):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(filename, fourcc, 5.0, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

# Forward pass and visualization
for idx, (batch_data, batch_labels, inverse_batch_masks, class_names) in enumerate(datagen):
    batch_data = 2 * batch_data.permute(0, 4, 1, 2, 3).to('cuda')/255 - 1
    batch_labels = 2 * batch_labels.permute(0, 4, 1, 2, 3).to('cuda')/255 - 1
    inverse_batch_masks = inverse_batch_masks.permute(0, 4, 1, 2, 3).to('cuda')

    batch_data[inverse_batch_masks == 1] = 0

    with torch.no_grad():
        output = model(batch_data)

    batch_data = (batch_data.permute(0, 2, 3, 4, 1)+1)/2
    batch_labels = (batch_labels.permute(0, 2, 3, 4, 1)+1)/2
    inverse_batch_masks = inverse_batch_masks.permute(0, 2, 3, 4, 1)

    batch_data[inverse_batch_masks == 1] = 0
    output = (output.permute(0, 2, 3, 4, 1)+1)/2

    #output[inverse_batch_masks == 0] = batch_labels[inverse_batch_masks == 0]

    visualize_videos(batch_data.cpu().numpy()*255, output.cpu().numpy()*255, batch_labels.cpu().numpy()*255)
    
    # Save the model's output for the first video
    if idx == 0:
        output_frames = [frame for frame in output[0].cpu().numpy()]
        output_frames = [(frame * 255).astype(np.uint8) for frame in output_frames]
        save_video(output_frames, 'model_output.avi')
        break  # Stop after saving the first video's output
