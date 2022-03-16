import os
import zipfile
import urllib.request

data_dir = "./datazip"
url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

# join tệp có tên .zip vào đường dẫn tuyệt đối data_dir
save_path = os.path.join(data_dir, "hymenoptera_data.zip")

# check pass đã tồn tại hay chưa nếu có không down nữa
if not os.path.exists(save_path):

    # request để tải dữ liệu xuống và lưu vào đường dẫn
    urllib.request.urlretrieve(url, save_path)
#
    # read by zipfile
    zip = zipfile.ZipFile(save_path)

    # giải nén file zip
    zip.extractall(data_dir)
    zip.close()

    os.remove(save_path)