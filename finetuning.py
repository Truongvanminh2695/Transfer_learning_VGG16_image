from lib import *
from Image_Transform import ImageTransform


from config import *

# get path of datak

def make_datapath_list(phase = 'train'):
    rootpath = 'data/hymenoptera_data/'
    target_path = osp.join(rootpath+phase+"/**/*.jpg")

    # stack từng path ảnh vào list này
    path_list = []

    # quét tất cả ảnh trong folder có định dạng này
    for path in glob.glob(target_path):
        path_list.append(path)
    return path_list

train_list = make_datapath_list('train')
val_list = make_datapath_list('val')

# kế thừa class Dataset của pytorch chứa các hàm số để thao tác với dataset
class MyDataset(data.Dataset):
    def __init__(self, file_list, transform = None, phase = 'train'):
        # lấy đường dẫn của img
        self.file_list = file_list

        self.transform = transform
        self.phase = phase

#    function return len dataset hàm bắt buộc phải có
    def __len__(self):
        return len(self.file_list)

#     ham bắt buộc phải có, trả cho mk đầu ra để truyền vào model
#     idx: là ảnh thứ bao nhiêu trong dataset
    def __getitem__(self, idx):
        img_path = self.file_list[idx]

#         mở ảnh lên
        img = Image.open(img_path)

        # truyền vào hàm transform đã được define ở trên nên chỉ cần truyền vào img và phase
        img_transformed = self.transform(img, self.phase)

        # get label
        # if self.phase == 'train':
        #     label = img_path[30:34]
        # elif self.phase == 'val':
        #     label = img_path[28:32]
        label = img_path.split('\\')[-2]
        if label =='ants':
            label = 0
        elif label == 'bees':
            label = 1
        return img_transformed, label

# dataset
train_dataset = MyDataset(train_list, transform = ImageTransform(resize, mean, std), phase='train')
val_dataset = MyDataset(val_list, transform = ImageTransform(resize, mean, std), phase='val')

batch_size = 4

# dataloader
# shuffle = True là để xáo trộn data trong mỗi 1 epoch
train_dataloader = DataLoader(train_dataset, batch_size, shuffle= True)

# khi kiếm thử không xáo trộn data shuffle = False
val_dataloader = DataLoader(val_dataset, batch_size, shuffle= False)

dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

# network
use_pretrained = True

# gọi models đã train với imagenet
net = models.vgg16(pretrained= True)
print(net)

# (6): Linear(in_features=4096, out_features=1000, bias=True) sửa index số 6 này có out_features = 2
# đây là transfer learning
net.classifier[6] = nn.Linear(in_features= 4096, out_features= 2 , bias= True)
# print(net)

# setting mode train anh val
# net.train()

# loss
criterion = nn.CrossEntropyLoss()

# optimizer
#chỉ muốn thay đổi params cuối cùng của network và giữ nguyên params của models pretrained
params_to_update = []
update_params_name = ['classifier.6.weight','classifier.6.bias']

for name, param in net.named_parameters():
    if name in update_params_name:
        # nếu name nằm trong update_params_name thì
        # requires_grad = True(lưu được thông số gradiant trong network thì mới update được weight và bias)
        # nếu không lưu thì ảnh không học được gì
        param.requires_grad = True
        # add param này vào thanh phần tử cần được update
        params_to_update.append(param)
    else:
        param.requires_grad = False
# tìm layers update với finetuning
def params_to_update(net):
    params_to_update1 = []
    params_to_update2 = []
    params_to_update3 = []

    update_params_name1 = ["features"]
    update_params_name2 = ['classifier.0.weight', 'classifier.0.bias','classifier.3.weight', 'classifier.3.bias']
    update_params_name3 = ['classifier.6.weight', 'classifier.6.bias']

    for name, param in net.named_parameters():
        if name in update_params_name1:
            param.requires_grad = True
            params_to_update1.append(param)



optimizer = optim.SGD(params=params_to_update, lr=0.01, momentum=0.9)

# training
def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print("epoch {}/{}".format(epoch, num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0
            if (epoch == 0) and (phase == 'train'):
                continue
            #     thư viện tqdm để chạy ra cái pbar cho mình xem
            for inputs, labels in tqdm(dataloader_dict[phase]):
                # đưa params và grad = 0 vì mỗi 1 epoch thì phải đưa về thông tin không có gì để học lại chứ không dùng cái cũ
                optimizer.zero_grad()

                # nếu train thì phase == 'train' là True thì enable grad để update weight
                # nếu val thì sẽ la False sẽ không enable
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    # tính loss so sánh outputs và labels của ảnh
                    loss = criterion(outputs, labels)
                    # outputs này gồm matrix có số hàng = batch_size , số cột bằng số class, axis = 1 là tìm max trong từng hàng 1, mỗi hàng biểu diễn cho 1 bức ảnh đi vào
                    _, preds = torch.max(outputs, 1)
                    # backward để update với training
                    if phase == 'train':
                        loss.backward()
                    #   update params trong optimize
                        optimizer.step()
                    # in thông số loss hiện tại của epoch
                    #  epoch loss = từng loss batch_size cộng dồn lại
                    # loss đang ở dạng tensor muốn lấy được giá trị trong tensor thì dùng item()
                    # input là 1 tensor có 4 chiều (bat_size, chaneel, height, wight)
                    epoch_loss += loss.item()*inputs.size(0)

                    #
                    epoch_corrects += torch.sum(preds == labels.data)
                # lấy trung bình loss của các batch_zise để ra loss của epoch
                epoch_loss = epoch_loss/ len(dataloader_dict[phase].dataset)
                epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
                print ("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

# num_epochs = 2
# train_model(net, dataloader_dict, criterion, optimizer, num_epochs)