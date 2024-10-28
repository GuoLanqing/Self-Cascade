import numpy as np
import torch, os
import torchvision
from PIL import Image
from .resize import build_resizer
import zipfile


class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, size=(299, 299), random_crop=False, fdir=None):
        self.files = files
        self.fdir = fdir
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = build_resizer(mode)
        if random_crop:
            self.custom_image_tranform = torchvision.transforms.RandomCrop((299, 299))
        else:
            self.custom_image_tranform = lambda x: x
        self._zipfile = None
        self.random_crop = random_crop
        self.index=0
    def _get_zipfile(self):
        assert self.fdir is not None and '.zip' in self.fdir
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.fdir)
        return self._zipfile

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        while True:
            path = str(self.files[i])
            if self.fdir is not None and '.zip' in self.fdir:
                with self._get_zipfile().open(path, 'r') as f:
                    img_np = np.array(Image.open(f).convert('RGB'))
            elif ".npy" in path:
                img_np = np.load(path)
            else:
                img_pil = Image.open(path).convert('RGB')
                img_np = np.array(img_pil)
            # print(f'self.random_crop={self.random_crop}')
            if self.random_crop:
                if (img_np.shape[0] < 299) or (img_np.shape[1] < 299):
                    i += 1
                    print('skip')
                    continue
                # print(f'Before rand cop, img_np.shape={img_np.shape}')
                img_np = np.array(self.custom_image_tranform(Image.fromarray(img_np)))
                # print(f'After rand cop, img_np.shape={img_np.shape}')
            else:
                # apply a custom image transform before resizing the image to 299x299
                img_np = self.custom_image_tranform(img_np)
            # fn_resize expects a np array and returns a np array
            if not self.random_crop:
                img_resized = self.fn_resize(img_np)
            else:
                img_resized = img_np
            # ToTensor() converts to [0,1] only if input in uint8
            if img_resized.dtype == "uint8":
                img_t = self.transforms(np.array(img_resized))*255
            elif img_resized.dtype == "float32":
                img_t = self.transforms(img_resized)
            break
        
        # dir="/apdcephfs_cq3/share_1290939/yingqinghe/results/any-res/debug/fid-imgs-fake"
        # os.makedirs(dir,exist_ok=True)
        # id=len(os.listdir(dir))
        # torchvision.utils.save_image(img_t, os.path.join(dir, f'{(id+1):06d}.jpg'),normalize=True)
        return img_t


EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
              'tif', 'tiff', 'webp', 'npy', 'JPEG', 'JPG', 'PNG'}