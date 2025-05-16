import random
import torchvision.transforms.functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class ToTensor(object):
    def __call__(self, image):
        return F.to_tensor(image)

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            return F.hflip(image)
        return image
