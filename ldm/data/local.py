import glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from functools import partial
from PIL import Image
from torch.utils.data import Dataset

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light


class Local(Dataset):
    def __init__(self, source_mather, limit=None, size=None,
                 degradation=None, downscale_f=4, min_crop_f=0.5, max_crop_f=1.,
                 random_crop=True):

        self.path = glob.glob(source_mather)
        if limit is not None:
            self.path = self.path[:limit]
        self.base = {
            "relative_file_path_": self.path,
            "file_path_": self.path,
        }
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

        self.pil_interpolation = False # gets reset later if incase interp_op is from pillow

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)

        else:
            interpolation_fn = {
                "cv_nearest": cv2.INTER_NEAREST,
                "cv_bilinear": cv2.INTER_LINEAR,
                "cv_bicubic": cv2.INTER_CUBIC,
                "cv_area": cv2.INTER_AREA,
                "cv_lanczos": cv2.INTER_LANCZOS4,
                "pil_nearest": PIL.Image.NEAREST,
                "pil_bilinear": PIL.Image.BILINEAR,
                "pil_bicubic": PIL.Image.BICUBIC,
                "pil_box": PIL.Image.BOX,
                "pil_hamming": PIL.Image.HAMMING,
                "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]

            self.pil_interpolation = degradation.startswith("pil_")

            if self.pil_interpolation:
                self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn)

            else:
                self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                          interpolation=interpolation_fn)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        example = self.base[i]
        image = Image.open(example["file_path_"])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)

        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)

        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)

        else:
            self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)

        image = self.cropper(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]

        if self.pil_interpolation:
            image_pil = PIL.Image.fromarray(image)
            LR_image = self.degradation_process(image_pil)
            LR_image = np.array(LR_image).astype(np.uint8)

        else:
            LR_image = self.degradation_process(image=image)["image"]

        example["image"] = (image/127.5 - 1.0).astype(np.float32)
        example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float32)

        return example
