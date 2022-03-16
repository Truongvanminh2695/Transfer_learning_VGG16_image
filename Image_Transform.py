from lib import *

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                 transforms.RandomResizedCrop(resize, scale = (0.5, 1.0)),

                 # xác suất để xoay
                 transforms.RandomHorizontalFlip(),
                # đưa bức ảnh vào trong network thì phải chuyển về tensor
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
              ]),
             'val':transforms.Compose([
                 transforms.Resize(resize),

                 # xác suất để xoay
                 transforms.CenterCrop(resize),
                # đưa bức ảnh vào trong network thì phải chuyển về tensor
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        }
#     tạo hàm gọi data_transform ra, khi taoj objet thì tự động gọi vào hàm call
#     phase là  mode train hoặc val, khi gọi đến class này thì chỉ cần truyền vào img và phase
    def __call__(self, img, phase = 'train'):
        return self.data_transform[phase](img)