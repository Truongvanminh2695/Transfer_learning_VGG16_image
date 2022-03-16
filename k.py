import matplotlib.pyplot as plt
from PIL import Image
img = Image.open('dog.jpeg')
img1 = img.convert("RGB")
plt.subplot(1,2,1)
plt.imshow(img)

# plt.subplot(1,2,2)
# plt.imshow(img1)
# plt.show()
