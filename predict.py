from lib import *
from classification import load_model,ImageTransform,save_path,resize,mean,std
import io
class_idx = ['ants','bees']

class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index
    # output là lớp cuối của mạng
    def predict_max(self, output):
        # detach để tách ra không đạo hàm nữa
        max_id = np.argmax(output.detach().numpy)
        predicted_label = self.class_index[max_id]
        return predicted_label

predictor = Predictor(class_idx)

# def predict(img, savepath):
#
#     # prepare network
#     use_pretrained =  True
#     net = models.vgg16(pretrained=use_pretrained)
#     net.classifier[6] = nn.Linear(in_features=4096, out_features= 2)
#
#     # .eval để đưa sang dạng để predict
#     net.eval()
#
# #     load model đã train
#     model = load_model(net, savepath)
#     img = Image.open(img)
# #     prepare input img
#     transform = ImageTransform(resize, mean, std)
#     img = transform(img,phase='test')
#
#     # tạo thêm 1 chiều nữa chuyển về (1, chanel, height, w
#     img = img.unsqueeze_(0)
#
# #     predict
#     output = model(img)
#     result = predictor.predict_max(output)
#     return result
img = 'data/hymenoptera_data/train/ants/0013035.jpg'

img = Image.open(img)
transform = ImageTransform(resize, mean, std)
img = transform(img,phase='test')
img = img.unsqueeze_(0)
print(img.shape)
use_pretrained =  True
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(in_features=4096, out_features= 2)
#
#     # .eval để đưa sang dạng để predict
net.eval()
model = load_model(save_path, net)
# print(predict(img, save_path))
out = model(img)
result = predictor.predict_max(out)
print(result)








