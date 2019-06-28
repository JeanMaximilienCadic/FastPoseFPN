import torch.utils.data as data
import numpy as np
import skimage.transform as transform
from skimage.filters import gaussian
import cv2

class KeypointDataset(data.Dataset):
    def __init__(self, coco_train, num_of_keypoints, cocopath, cache_dir=None):
        self.coco_train = coco_train
        self.num_of_keypoints = num_of_keypoints
        self.images = self.getImage(self.coco_train)
        self.cocopath = cocopath
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self,item):
        ann_data = self.images[item]
        img_id = ann_data['image_id']
        input,label = self.try_get_cache(ann_data, img_id) if self.cache_dir is not None else self.get_data(ann_data, img_id)
        return input, label


    def try_get_cache(self, ann_data, img_id):
        def save_cache(result, img_id):
            np.save("{cache_dir}/{img_id}.npy".format(cache_dir=self.cache_dir, img_id=img_id), np.array(result))

        def load_cache(img_id):
            return np.load("{cache_dir}/{img_id}.npy".format(cache_dir=self.cache_dir, img_id=img_id))

        try:
            output = load_cache(img_id)
        except:
            output = self.get_data(ann_data, img_id)
            save_cache(output, img_id)
        return output

    def get_data(self, ann_data, img_id ):
        img_data = self.coco_train.loadImgs(img_id)[0]
        path = self.cocopath + img_data['file_name']
        img = cv2.imread(path)

        ori_size = img.shape
        assert len(ori_size) == 3
        img = transform.resize(img, (256, 256))
        x_scale = 256 / ori_size[0]
        y_scale = 256 / ori_size[1]
        size = img.shape

        # get a mask
        labels = np.zeros((size[0], size[1], 17))
        kpx = ann_data['keypoints'][0::3]
        kpy = ann_data['keypoints'][1::3]
        kpv = ann_data['keypoints'][2::3]

        for j in range(17):
            if kpv[j] > 0:
                x0 = int(kpx[j] * x_scale)
                y0 = int(kpy[j] * y_scale)

                if x0 >= size[1] and y0 >= size[0]:
                    labels[size[0] - 1, size[1] - 1, j] = 1
                elif x0 >= size[1]:
                    labels[y0, size[1] - 1, j] = 1
                elif y0 >= size[0]:
                    try:
                        labels[size[0] - 1, x0, j] = 1
                    except:
                        labels[size[0] - 1, 0, j] = 1
                elif x0 < 0 and y0 < 0:
                    labels[0, 0, j] = 1
                elif x0 < 0:
                    labels[y0, 0, j] = 1
                elif y0 < 0:
                    labels[0, x0, j] = 1
                else:
                    labels[y0, x0, j] = 1

        labels = gaussian(labels, sigma=2, mode='constant', multichannel=True)
        labels = np.rollaxis(labels, -1, 0)
        output = np.rollaxis(img, -1, 0), labels
        return output



    def getImage(self,coco):
        ids = coco.getAnnIds()
        images = []
        for i in ids:
            image = coco.loadAnns(i)[0]
            if image['iscrowd'] == 0 and image['num_keypoints'] > self.num_of_keypoints:
                images.append(image)
        return images
