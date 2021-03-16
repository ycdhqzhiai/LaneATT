import logging
import argparse

import torch
import cv2
from lib.config import Config
from lib.runner import Runner
from lib.experiment import Experiment
import numpy as np
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
def parse_args():
    parser = argparse.ArgumentParser(description="Train lane detector")
    parser.add_argument("--weights", type=str, default='')
    parser.add_argument("--cfg",type=str, help="Config file")
    args = parser.parse_args()
    return args

def draw_annotation(src, pred, img_w=640, img_h=360):
    img = cv2.resize(src, (img_w, img_h))
    for i, l in enumerate(pred):
        print(type(l))
        points = l.points
        points[:, 0] *= img.shape[1]
        points[:, 1] *= img.shape[0]
        points = points.round().astype(int)
        xs, ys = points[:, 0], points[:, 1]
        for curr_p, next_p in zip(points[:-1], points[1:]):
            img = cv2.line(img,
                            tuple(curr_p),
                            tuple(next_p),
                            color=(0,0,255),
                            thickness=2)
    return img

def preprocess(img, img_w=640, img_h=360):
    img = cv2.resize(img, (img_w, img_h))
    img = img / 255.
    # img = (img - IMAGENET_MEAN) / IMAGENET_STD
    out_images = img.transpose(2, 0, 1)
    out_images = np.expand_dims(out_images, axis=0).astype(np.float32)
    out_images = torch.from_numpy(out_images)
    return out_images

def main():
    args = parse_args()
    cfg = Config(args.cfg)
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

    model = cfg.get_model()
    state_dict = torch.load(args.weights)['model']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    test_parameters = cfg.get_test_parameters()
    img = cv2.imread('00000.jpg')
    src = img.copy()
    img = preprocess(img).to(device)
    output = model(img, **test_parameters)
    print(output)
    prediction = model.decode(output, as_lanes=True)
    image = draw_annotation(src, prediction[0])

    cv2.imshow('pred', image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
