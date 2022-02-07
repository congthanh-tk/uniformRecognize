from __future__ import division
import os
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmfashion.utils import save_imgs, get_img_tensor
from mmfashion.apis import (get_root_logger, init_dist,
                            test_geometric_matching, test_tryon)
from mmfashion.models import build_geometric_matching, build_tryon


def geometric_matching(model, img_path):
    # save dir
    warp_cloth_dir = os.path.join('data/images', 'warp-cloth')
    warp_mask_dir = os.path.join('data/images', 'warp-mask')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    c_name = img_path
    cloth = get_img_tensor(img_path, torch.cuda.is_available())
    # cloth_mask = data['cloth_mask'].cuda()
    # agnostic = data['agnostic'].cuda()
    # parse_cloth = data['parse_cloth'].cuda()

    warped_cloth, warped_mask = model(
        cloth, cloth, cloth, cloth, return_loss=False)
    save_imgs(warped_cloth, c_name, warp_cloth_dir)
    save_imgs(warped_mask, c_name, warp_mask_dir)

def main():
    config_file = "configs/virtual_tryon/cp_vton.py"
    checkpoint = "checkpoint/CPVTON/GMM/GMM.pth"
    cfg = Config.fromfile(config_file)
    if checkpoint is not None:
        cfg.load_from = checkpoint
    # init distributed env first
    # distributed = Falses

    # init logger
    # logger = get_root_logger(cfg.log_level)
    # logger.info('Distributed test: {}'.format(distributed))

    # build model and load checkpoint
    # Part 1: geometric matching

    model = build_geometric_matching(cfg.GMM)
    print('GMM model built')
    load_checkpoint(model, cfg.load_from, map_location=(torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')))
    print('load checkpoint from: {}'.format(cfg.load_from))

    geometric_matching(
        model,
        img_path='demo/imgs/06_1_front.jpg')

    # elif args.stage == 'TOM':
    #     # test tryon module
    #     dataset = get_dataset(cfg.data.test.TOM)
    #     print('TOM dataset loaded')

    #     model = build_tryon(cfg.TOM)
    #     print('TOM model built')
    #     load_checkpoint(model, cfg.load_from, map_location='cpu')
    #     print('load checkpoint from: {}'.format(cfg.load_from))

    #     test_tryon(
    #         model,
    #         dataset,
    #         cfg,
    #         distributed=distributed,
    #         validate=False,
    #         logger=logger)


if __name__ == '__main__':
    main()
