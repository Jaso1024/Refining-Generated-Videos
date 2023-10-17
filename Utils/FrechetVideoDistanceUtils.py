import sys

sys.path.append('Utils')
sys.path.append('Models')
sys.path.append('Weights')
sys.path.append('Data')

import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data

from Inflated3DConvolutionalNetwork import InceptionI3d
import os
import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import polynomial_kernel

MAX_BATCH = 2
FVD_SAMPLE_SIZE = 2048
TARGET_RESOLUTION = (224, 224)

def preprocess(videos, target_resolution):
    # videos in {0, ..., 255} as np.uint8 array
    b, t, h, w, c = videos.shape
    all_frames = torch.FloatTensor(videos).flatten(end_dim=1) # (b * t, h, w, c)
    all_frames = all_frames.permute(0, 3, 1, 2).contiguous() # (b * t, c, h, w)
    resized_videos = F.interpolate(all_frames, size=target_resolution,
                                   mode='bilinear', align_corners=False)
    resized_videos = resized_videos.view(b, c, t, *target_resolution)
    output_videos = resized_videos# (b, c, t, *)
    scaled_videos = 2. * output_videos / 255. - 1 # [-1, 1]
    return scaled_videos

def get_fvd_logits(videos, i3d, device):
    videos = preprocess(videos, TARGET_RESOLUTION)
    embeddings = get_logits(i3d, videos, device)
    return embeddings

def load_fvd_model(device):
    i3d = InceptionI3d(400, in_channels=3).to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    i3d_path = os.path.join(current_dir, 'i3d_pretrained_400.pt')
    i3d.load_state_dict(torch.load(i3d_path, map_location=device))
    i3d.eval()
    return i3d


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())

# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1) # unbiased estimate
    m_center = m - torch.mean(m, dim=1, keepdim=True)
    mt = m_center.t()  # if complex: mt = m.t().conj()
    return fact * m_center.matmul(mt).squeeze()


def frechet_distance(x1, x2):
    m, m_w = x1.mean(dim=0), x2.mean(dim=0)
    sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)

    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
    trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    mean = torch.sum((m - m_w) ** 2)
    fd = trace + mean
    return fd


def polynomial_mmd(X, Y):
    m = X.shape[0]
    n = Y.shape[0]
    # compute kernels
    K_XX = polynomial_kernel(X)
    K_YY = polynomial_kernel(Y)
    K_XY = polynomial_kernel(X, Y)
    # compute mmd distance
    K_XX_sum = (K_XX.sum() - np.diagonal(K_XX).sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - np.diagonal(K_YY).sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)
    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum
    return mmd



def get_logits(i3d, videos, device):
    assert videos.shape[0] % MAX_BATCH == 0
    with torch.no_grad():
        logits = []
        for i in range(0, videos.shape[0], MAX_BATCH):
            batch = videos[i:i + MAX_BATCH].to(device)
            logits.append(i3d.extract_features(batch))
        logits = torch.cat(logits, dim=0)
        return logits

def get_preds(i3d, videos, device):
    assert videos.shape[0] % MAX_BATCH == 0
    with torch.no_grad():
            logits = []
            for i in range(0, videos.shape[0], MAX_BATCH):
                batch = videos[i:i + MAX_BATCH].to(device)
                logits.append(i3d(batch))
            logits = torch.cat(logits, dim=0)
            return logits

def compute_fvd(real, samples, i3d, device=torch.device('cpu')):
    # real, samples are (N, T, H, W, C) numpy arrays in np.uint8
    real, samples = preprocess(real, TARGET_RESOLUTION), preprocess(samples, TARGET_RESOLUTION)
    
    first_embed = get_logits(i3d, real, device).reshape(2,1024)
    second_embed = get_logits(i3d, samples, device).reshape(2,1024)


    return frechet_distance(first_embed, second_embed)

import numpy as np

def kvd(real_activations, generated_activations, max_block_size=10):
    """Kernel "classifier" distance for evaluating a generative model.
    ...
    """
    # Ensure the inputs are 2D arrays
    assert len(real_activations.shape) == 2
    assert len(generated_activations.shape) == 2
    assert real_activations.shape[1] == generated_activations.shape[1]

    # Figure out how to split the activations into blocks of approximately
    # equal size, with none larger than max_block_size.
    n_r, n_g = real_activations.shape[0], generated_activations.shape[0]
    n_bigger = max(n_r, n_g)
    n_blocks = int(np.ceil(n_bigger / max_block_size))

    v_r, v_g = n_r // n_blocks, n_g // n_blocks
    n_plusone_r, n_plusone_g = n_r - v_r * n_blocks, n_g - v_g * n_blocks

    sizes_r = np.concatenate([np.full(n_blocks - n_plusone_r, v_r), np.full(n_plusone_r, v_r + 1)])
    sizes_g = np.concatenate([np.full(n_blocks - n_plusone_g, v_g), np.full(n_plusone_g, v_g + 1)])

    inds_r = np.concatenate([[0], np.cumsum(sizes_r)])
    inds_g = np.concatenate([[0], np.cumsum(sizes_g)])

    dim = real_activations.shape[1]

    def polynomial_kernel(x, y):
        return ((np.dot(x, y.T) / dim) + 1) ** 3

    def compute_kid_block(i):
        r_start, r_end = inds_r[i], inds_r[i + 1]
        g_start, g_end = inds_g[i], inds_g[i + 1]

        k_xx = polynomial_kernel(real_activations[r_start:r_end], real_activations[r_start:r_end])
        k_yy = polynomial_kernel(generated_activations[g_start:g_end], generated_activations[g_start:g_end])
        k_xy = polynomial_kernel(real_activations[r_start:r_end], generated_activations[g_start:g_end])

        n = k_xx.shape[0]  # Assuming k_xx and k_yy are square matrices
        return (np.sum(k_xx) + np.sum(k_yy) - 2 * np.sum(k_xy)) / (n * (n - 1))

    ests = np.array([compute_kid_block(i) for i in range(n_blocks)])

    mn = np.mean(ests)
    
    n_blocks_ = n_blocks
    var = np.var(ests, ddof=1)  # using ddof=1 for Bessel's correction

    return mn, np.sqrt(var / n_blocks_)



def compute_preds(real, i3d, device=torch.device('cpu')):
    # real, samples are (N, T, H, W, C) numpy arrays in np.uint8
    real = preprocess(real, TARGET_RESOLUTION)
    
    first_embed = get_preds(i3d, real, device)

    return first_embed
