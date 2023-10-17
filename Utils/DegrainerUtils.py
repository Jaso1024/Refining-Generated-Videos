import torch
import random

def grainify(inp, noise_type='gaussian', max_difference=0.03, stdev=0.02, abs_min=0, abs_max=1):
    if noise_type == 'gaussian':
        noise = torch.randn_like(inp) * stdev
        noise = torch.clamp(noise, -max_difference, max_difference)
        noisy_tensor = inp + noise
    
    elif noise_type == 'speckle':
        noise = torch.randn_like(inp) * stdev
        noisy_tensor = inp + inp * noise
    
    elif noise_type == 'poisson':
        noisy_tensor = torch.poisson(inp * 255) / 255

    elif noise_type == 'None':
        noisy_tensor = inp
    
    # Clamp the values to be within the absolute min and max
    noisy_tensor = torch.clamp(noisy_tensor, abs_min, abs_max)
    
    return noisy_tensor

def rgb_to_hsv(rgb):
    r, g, b = rgb[:, :, 0, :, :], rgb[:, :, 1, :, :], rgb[:, :, 2, :, :]
    max_val, _ = torch.max(rgb, dim=2)
    min_val, _ = torch.min(rgb, dim=2)
    diff = max_val - min_val

    h = torch.zeros_like(max_val)
    h[max_val == r] = ((g[max_val == r] - b[max_val == r]) / diff[max_val == r]) % 6
    h[max_val == g] = (b[max_val == g] - r[max_val == g]) / diff[max_val == g] + 2
    h[max_val == b] = (r[max_val == b] - g[max_val == b]) / diff[max_val == b] + 4
    h[min_val == max_val] = 0.0
    h = h / 6.0

    s = torch.zeros_like(max_val)
    s[max_val != 0] = diff[max_val != 0] / max_val[max_val != 0]

    v = max_val

    return torch.stack([h, s, v], dim=2)

def compute_hue_loss(predicted, ground_truth):
    # Convert RGB to HSV
    predicted_hsv = rgb_to_hsv(predicted)
    ground_truth_hsv = rgb_to_hsv(ground_truth)
    
    # Extract Hue component
    predicted_hue = predicted_hsv[:, :, 0, :, :]
    ground_truth_hue = ground_truth_hsv[:, :, 0, :, :]
    
    # Compute Hue Loss using Mean Squared Error
    hue_loss = torch.mean((predicted_hue - ground_truth_hue) ** 2)
    
    return hue_loss



if __name__ == "__main__":
    inp_tensor = torch.tensor([1.0, 1.0, 1.0, 1.0])


    noisy_tensor = grainify(inp_tensor)
    print(noisy_tensor)