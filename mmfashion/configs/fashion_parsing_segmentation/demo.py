import argparse

from mmdet.apis import inference_detector, init_detector
from mmdet.models.detectors import BaseDetector

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Fashion Detection and Segmentation')
    parser.add_argument(
        '--config',
        help='mmfashion config file path',
        default='configs/fashion_parsing_segmentation/mask_rcnn_r50_fpn_1x.py')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--input',
        type=str,
        help='input image path',
        default='demo/01_4_full.jpg')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cpu')

    # test a single image and show the results
    img = args.input
    result = inference_detector(model, img)

    # visualize the results in a new window
    # or save the visualization results to image files
    BaseDetector.show_result(
        img, result, model.CLASSES, out_file=img.split('.')[0] + '_result.jpg')


if __name__ == "__main__":
    main()
