"""
Script to generate density maps
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree
from skimage.io import imread


class GDM(object):
    def __init__(self, args):
        self.train_images = args.train_images
        self.train_groundtruth = args.train_groundtruth
        self.output_folder = args.output_folder
        self.dataset_name = args.dataset
        self.sigma = args.sigma

        self.test_folder(self.output_folder)

    def test_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    # Code sourced from https://github.com/davideverona/deep-crowd-counting_crowdnet/blob/master/dcc_crowdnet.ipynb
    def gaussian_filter_density(self, gt):
        density = np.zeros(gt.shape, dtype=np.float32)
        gt_count = np.count_nonzero(gt)
        if gt_count == 0:
            return density

        pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
        leafsize = 2048
        tree = KDTree(pts.copy(), leafsize=leafsize)
        distances, locations = tree.query(pts, k=4)

        for idx, pt in enumerate(pts):
            pt2d = np.zeros(gt.shape, dtype=np.float32)
            pt2d[pt[1], pt[0]] = 1.
            if gt_count > 1:
                sigma = (distances[idx][1] + distances[idx][2] + distances[idx][
                    3]) * 0.1 if not self.sigma == None else self.sigma
            else:
                sigma = np.average(np.array(gt.shape)) / 2. / 2.
            density += gaussian_filter(pt2d, sigma, mode='constant')
        return density

    def read_img_gt(self, img, gt):
        img = imread(img)
        gt = loadmat(gt)

        return img, gt

    def convert_json_to_gt(self, gt_json, gt_shape):
        gt = np.zeros(gt_shape, dtype=np.uint8)
        for idx, values in enumerate(gt_json):
            try:
                gt[int(values['y']), int(values['x'])] = 1
            except:
                print(int(values['y']), int(values['x']))
        return gt

    def generate(self):
        images = [os.path.join(self.train_images, img) for img in sorted(os.listdir(self.train_images))]
        gts = [os.path.join(self.train_groundtruth, gt) for gt in sorted(os.listdir(self.train_groundtruth))]

        self.dataset = {}
        assert len(images) == len(gts)
        print("Found {} images, and {} ground-truths".format(len(images), len(gts)))
        print("Sigma value: {}".format(self.sigma))
        for data in zip(images, gts):
            unique_identifier = os.path.split(data[0])[-1].split('.')[0]
            self.dataset[unique_identifier] = {'image': data[0], 'gt': data[1]}

        for idx, (key, value) in enumerate(self.dataset.items()):
            img, gt = self.read_img_gt(value['image'], value['gt'])
            unique_identifier = os.path.split(value['image'])[-1].split('.')[0]

            if self.dataset_name == 'ShanghaiTech':
                annotation_points = gt['image_info'][0][0][0][0][0]
            else:
                annotation_points = gt['annPoints']

            annotation_points_to_json = map(lambda _: {'x': _[0], 'y': _[1]}, annotation_points)
            image_shape = img.shape[:-1] if img.ndim == 3 else img.shape
            gt_json = self.convert_json_to_gt(annotation_points_to_json, image_shape)

            density_map = self.gaussian_filter_density(gt_json)
            csv_path = os.path.join(self.output_folder, '{}.csv'.format(unique_identifier))
            pd.DataFrame(density_map).to_csv(csv_path, index=False)

            if (idx + 1) % 50 == 0:
                print("Progress of generation: {}/{}: {}%".format(idx + 1, len(self.dataset.keys()),
                                                                  (float(idx + 1) / len(self.dataset.keys())) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ti', '--train_images', help="Path of the train image dataset")
    parser.add_argument('-tgt', '--train_groundtruth', help="Path of the train groundtruth dataset")
    parser.add_argument('-o', '--output_folder', help="Path of the output folder to store the generated density maps")
    parser.add_argument('-d', '--dataset', help="Name of the dataset", default='ShanghaiTech')
    parser.add_argument('-s', '--sigma', help="Size of the density kernel standard deviation", default=None)

    args = parser.parse_args()

    gdm = GDM(args)
    gdm.generate()
