# liệt kê các path chứa ảnh
import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
# chứa sdg, adam trong network
import torch.optim as optim
import numpy as np
# chứa hàm số để điều khiển data
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset

# làm các tác vụ computer vision trong torch
import torchvision

# models để gọi ra những models đã train trên imagenet rồi, transform để tiền xử lý
from torchvision import models, transforms