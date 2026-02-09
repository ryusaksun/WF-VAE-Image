import json
import os
import os.path as osp
import pickle
import random
from glob import glob

from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class BaseImageDataset(data.Dataset):
    image_exts = ("jpg", "jpeg", "png", "webp", "bmp")

    def __init__(
        self,
        image_folder,
        cache_file,
        is_main_process=False,
        use_manifest=False,
        manifest_path=None,
    ):
        self.image_folder = image_folder
        self.cache_file = cache_file
        self.is_main_process = is_main_process
        self.use_manifest = use_manifest
        self.manifest_path = manifest_path
        if self.use_manifest:
            if not self.manifest_path:
                self.manifest_path = self.image_folder
            self.image_folder = osp.dirname(osp.abspath(self.manifest_path))
        self.samples = self._make_dataset()

    def _make_dataset(self):
        if self.use_manifest:
            return self._load_from_manifest()

        cache_file = osp.join(self.image_folder, self.cache_file)
        if osp.exists(cache_file):
            with open(cache_file, "rb") as f:
                samples = pickle.load(f)
            return samples

        samples = []
        for ext in self.image_exts:
            samples.extend(
                glob(osp.join(self.image_folder, "**", f"*.{ext}"), recursive=True)
            )
            samples.extend(
                glob(osp.join(self.image_folder, "**", f"*.{ext.upper()}"), recursive=True)
            )
        samples = sorted(samples)
        if self.is_main_process:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(samples, f)
            except Exception as exc:
                print(f"Skip caching dataset index: {exc}")
        return samples

    def _load_from_manifest(self):
        if not self.manifest_path or not osp.isfile(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        base_dir = osp.dirname(osp.abspath(self.manifest_path))
        samples = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception as exc:
                    print(f"Skip invalid json at line {line_no}: {exc}")
                    continue

                image_path = None
                for key in ("image_path", "path", "target"):
                    value = item.get(key)
                    if isinstance(value, str) and value:
                        image_path = value
                        break

                if image_path is None:
                    print(f"Skip manifest line {line_no}: missing image path field")
                    continue

                if not osp.isabs(image_path):
                    image_path = osp.join(base_dir, image_path)
                samples.append(osp.normpath(image_path))

        print(f"Loaded {len(samples)} samples from {self.manifest_path}")
        return sorted(samples)

    def __len__(self):
        return len(self.samples)


class TrainImageDataset(BaseImageDataset):
    def __init__(
        self,
        image_folder,
        resolution=1024,
        cache_file="idx_image.pkl",
        is_main_process=False,
        use_manifest=False,
        manifest_path=None,
    ):
        super().__init__(
            image_folder=image_folder,
            cache_file=cache_file,
            is_main_process=is_main_process,
            use_manifest=use_manifest,
            manifest_path=manifest_path,
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )

    def __getitem__(self, idx):
        for _retry in range(10):
            image_path = self.samples[idx]
            try:
                image = Image.open(image_path).convert("RGB")
                image = self.transform(image)
                return {"image": image, "label": ""}
            except Exception as exc:
                print(f"Error with {exc}, {image_path}")
                idx = random.randint(0, len(self.samples) - 1)
        raise RuntimeError(f"Failed to load image after 10 retries, last path: {self.samples[idx]}")


class ValidImageDataset(BaseImageDataset):
    def __init__(
        self,
        image_dir,
        resolution=1024,
        crop_size=1024,
        cache_file="idx_image_eval.pkl",
        is_main_process=False,
        use_manifest=False,
        manifest_path=None,
    ):
        super().__init__(
            image_folder=image_dir,
            cache_file=cache_file,
            is_main_process=is_main_process,
            use_manifest=use_manifest,
            manifest_path=manifest_path,
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )

    def __getitem__(self, index):
        for _retry in range(10):
            image_path = self.samples[index]
            try:
                image = Image.open(image_path).convert("RGB")
                image = self.transform(image)
                file_name = os.path.relpath(image_path, self.image_folder)
                return {"image": image, "file_name": file_name, "index": int(index)}
            except Exception as exc:
                print(f"Image error with {exc}, {image_path}")
                index = random.randint(0, len(self.samples) - 1)
        raise RuntimeError(f"Failed to load image after 10 retries, last path: {self.samples[index]}")
