from lib import *

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# tiền xử lý bức ảnh

resize = 224
mean = (0.4854, 0.456, 0.406)
std = (0.2854, 0.256, 0.206)