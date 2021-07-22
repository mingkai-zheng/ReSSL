import random
import torchvision.transforms as transforms

from PIL import ImageFilter, Image, ImageOps
from data.randaugment import RandAugment

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize():
    def __init__(self, threshold=128):
        self.threshold = threshold
    def __call__(self, sample):
        return ImageOps.solarize(sample, self.threshold)



moco_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


eval_aug = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
])

class Multi_Transform(object):
    def __init__(
            self,
            size_crops=[224, 192, 160, 128, 96],
            nmb_crops=[1, 1, 1, 1, 1],
            min_scale_crops=[0.2, 0.172, 0.143, 0.114, 0.086],
            max_scale_crops=[1.0, 0.86, 0.715, 0.571, 0.429],
            init_size=224,
            strong=False):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        trans=[]

        self.strong = strong

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        #image_k
        weak = transforms.Compose([
            transforms.RandomResizedCrop(init_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        trans.append(weak)


        trans_weak=[]
        if strong:
            min_scale_crops=[0.08, 0.08, 0.08, 0.08, 0.08]
            jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        else:
            jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)


        for i in range(len(size_crops)):
            aug_list = [ 
                transforms.RandomResizedCrop(
                    size_crops[i],
                    scale=(min_scale_crops[i], max_scale_crops[i])
                ),
                transforms.RandomApply([jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
            ]

            if self.strong:
                aug_list.append(RandAugment(5, 10))

            aug_list.extend([
                transforms.ToTensor(),
                normalize
            ])

            aug = transforms.Compose(aug_list)
            trans_weak.extend([aug]*nmb_crops[i])

        trans.extend(trans_weak)
        self.trans=trans
        print("in total we have %d transforms"%(len(self.trans)))
    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops


