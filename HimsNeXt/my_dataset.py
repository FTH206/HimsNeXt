import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):

        super(MyDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "MoNuSeg", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        

        supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) 
                    if os.path.splitext(i)[1].lower() in supported_formats]
        
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        

        self.manual = []
        manual_dir = os.path.join(data_root, "manual")
        manual_files = os.listdir(manual_dir)
        
        for i in img_names:
            base_name = os.path.splitext(i)[0]
            

            exact_manual = os.path.join(manual_dir, base_name + "_manual.png")
            if os.path.exists(exact_manual):
                self.manual.append(exact_manual)
                continue
                

            potential_manual_bases = []
            

            base_parts = base_name.rsplit("_", 3)[:3]
            if len(base_parts) >= 3:
                potential_manual_bases.append("_".join(base_parts) + "_manual")
            

            potential_manual_bases.append(base_name + "_anno")
            

            potential_manual_bases.append(base_name + "_manual")
            

            potential_manual_bases.append(base_name)

            potential_manual_bases.append(base_name + "_mask")
            

            manual_file = None
            for manual_base in potential_manual_bases:
                for ext in supported_formats:
                    potential_file = manual_base + ext
                    if potential_file in manual_files:
                        manual_file = os.path.join(manual_dir, potential_file)
                        break
                if manual_file:
                    break
            

            if manual_file is None:
                for manual_base in potential_manual_bases:
                    manual_base_lower = manual_base.lower()
                    for f in manual_files:
                        if f.lower().startswith(manual_base_lower) and os.path.splitext(f)[1].lower() in supported_formats:
                            manual_file = os.path.join(manual_dir, f)
                            break
                    if manual_file:
                        break
            
            if manual_file is None:
                base_name_lower = base_name.lower()
                for f in manual_files:
                    file_base = os.path.splitext(f)[0].lower()
                    if file_base.startswith(base_name_lower) or base_name_lower.startswith(file_base):
                        manual_file = os.path.join(manual_dir, f)
                        break


                for mf in manual_files:
                    if mf.lower() == (base_name + "_manual.png").lower():
                        print(f"Find case-insensitive matches: {mf}")
                        manual_file = os.path.join(manual_dir, mf)
                        break
                
                if manual_file is None:
                    print(f"Tried basic names: {potential_manual_bases}")
                    print(f"Available manual files: {manual_files}")

                    for mf in manual_files:
                        mf_base = os.path.splitext(mf)[0]
                        if mf_base.endswith("_manual") and mf_base[:-8].replace("_", "") == base_name.replace("_", ""):
                            print(f"Find a special match: {mf}")
                            manual_file = os.path.join(manual_dir, mf)
                            break
            
            if manual_file is not None:
                self.manual.append(manual_file)
            else:

                if not train:
                    img_path = os.path.join(data_root, "images", i)
                    img = Image.open(img_path)
                    blank_mask = Image.new("L", img.size, 0)
                    os.makedirs("temp_masks", exist_ok=True)
                    blank_mask_path = os.path.join("temp_masks", base_name + "_manual.png")
                    blank_mask.save(blank_mask_path)
                    self.manual.append(blank_mask_path)
                    continue
                

                raise FileNotFoundError(f"Cannot find manual file for image {i}")
        

        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
    
    def __getitem__(self, idx):

        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        
        mask_np = np.array(manual)
        
        mask_np = (mask_np > 0).astype(np.uint8)


        mask = Image.fromarray(mask_np)
        
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask
    
    def __len__(self):

        return len(self.img_list)
    
    def get_image_name(self, idx):

        return os.path.basename(self.img_list[idx])
    
    @staticmethod
    def collate_fn(batch):

        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):

    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

