import glob

from PIL import Image
from torchvision import transforms
import cv2
from torchvision.transforms import InterpolationMode

patch_size = 48

for file in glob.glob("*.jpg"):
    img = transforms.ToTensor()(Image.open(file).convert('RGB')) * 255
    img_lr = transforms.Resize(patch_size, InterpolationMode.BICUBIC)(
        transforms.CenterCrop(8 * patch_size)(img))

    img_hr = transforms.CenterCrop(8 * patch_size)(img)

    cv2.imwrite(f'AID_{file.split(".")[0]}_LR.png', img_lr.permute((1, 2, 0)).numpy())
    print(f'AID_{file.split(".")[0]}_LR.png')
    cv2.imwrite(f'AID_{file.split(".")[0]}_HR.png', img_hr.permute((1, 2, 0)).numpy())

