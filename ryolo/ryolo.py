import argparse

import torch

from model import Model
from utils import adjust_image_channel, image_to_tensor, letterbox, rotate_non_max_suppression, gen_polygon, rotate_scale_coords, xywhrm2xyxyxyxy


class RYolo(Model):
    def __init__(self, args):
        parser = argparse.ArgumentParser()
        parser.add_argument("-l", "--label", help="The path of classes file.", required=True)
        parser.add_argument("-w", "--weights", help="The path of weights file.", required=True)
        group1 = parser.add_mutually_exclusive_group(required=True)
        group1.add_argument("--size", type=int, nargs="*", help="The size of input.")
        group1.add_argument("--scale", type=float, nargs="*", help="The scale of input.")
        group2 = parser.add_mutually_exclusive_group()
        group2.add_argument("--rect", action="store_true", help="Rect mode.")
        group2.add_argument("--scaleFill", action="store_true", help="ScaleFill mode.")
        parser.add_argument("--conf", type=float, default=0.25, help="The threshold of confidence.")
        parser.add_argument("--iou", type=float, default=0.45, help="The threshold of iou.")
        parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--multi-label', action='store_true', help='multi-label NMS')
        parser.add_argument('-v', "--verbose", action='store_true', help='Print log.')
        self.args = parser.parse_args(args)

        Model.__init__(self, self.args.weights)
        self.class_list = [line.strip() for line in open(self.args.label, encoding='utf8')]
        self.input_channel = self.session.get_inputs()[0].shape[1]

    def __call__(self, input_image):
        dsize = self.args.size if self.args.size else (int(input_image.shape[0] * self.args.scale[0]), int(input_image.shape[1] * self.args.scale[1]))
        image, ratio, (dw, dh) = letterbox(input_image, dsize, auto=self.args.rect, scaleFill=self.args.scaleFill)

        image = adjust_image_channel(image, self.input_channel)
        tensor = image_to_tensor(image, 0, 255)

        output_names = [self.session.get_outputs()[0].name]
        input_feed = {self.session.get_inputs()[0].name: tensor}

        output = Model.__call__(self, output_names, input_feed)

        pred = rotate_non_max_suppression(torch.tensor(output[0]), self.args.conf, self.args.iou, classes=self.args.classes, 
                agnostic=self.args.agnostic, multi_label=self.args.multi_label)

        pred = pred[0]

        pred[:, :4] = rotate_scale_coords(image.shape, pred[:, :4], input_image.shape).round()

        xyxyxyxy = xywhrm2xyxyxyxy(pred[:6])
        xyxyxyxy = xyxyxyxy.numpy().tolist()

        flags = {}
        shapes = []
        for item1, item2 in zip(pred, xyxyxyxy):
            index = int(item1[7])
            points = [[item2[0], item2[1]], [item2[2], item2[3]], [item2[4], item2[5]], [item2[6], item2[7]]]
            shape = gen_polygon(self.class_list[index], points)
            shapes.append(shape)

        return flags, shapes