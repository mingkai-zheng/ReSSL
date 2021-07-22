import torch.utils.data as data
import os
import torchvision.transforms as transforms
from PIL import Image
import mc
import io


class DatasetCache(data.Dataset):
    def __init__(self):
        super().__init__()
        self.initialized = False
    

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def load_image(self, filename):
        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(filename, value)
        value_str = mc.ConvertBuffer(value)
        
        buff = io.BytesIO(value_str)
        with Image.open(buff) as img:
            img = img.convert('RGB')
        return img



class BaseDataset(DatasetCache):
    def __init__(self, mode='train', max_class=1000, aug=None):
        super().__init__()
        self.initialized = False


        prefix = '/mnt/lustreold/share/images/meta'
        image_folder_prefix = '/mnt/lustreold/share/images'
        if mode == 'train':
            image_list = os.path.join(prefix, 'train.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'train')
        elif mode == 'test':
            image_list = os.path.join(prefix, 'test.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'test')
        elif mode == 'val':
            image_list = os.path.join(prefix, 'val.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'val')
        else:
            raise NotImplementedError('mode: ' + mode + ' does not exist please select from [train, test, eval]')


        self.samples = []
        with open(image_list) as f:
            for line in f:
                name, label = line.split()
                label = int(label)
                if label < max_class:
                    self.samples.append((label, name))

        if aug is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = aug




class ImagenetContrastive(BaseDataset):
    def __init__(self, mode='train', max_class=1000, aug=None):
        super().__init__(mode, max_class, aug)

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        _, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)
        if isinstance(self.transform, list):
            return self.transform[0](img), self.transform[1](img)
        return self.transform(img), self.transform(img)



class Imagenet(BaseDataset):
    def __init__(self, mode='train', max_class=1000, aug=None):
        super().__init__(mode, max_class, aug)

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        label, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)
        return self.transform(img), label

