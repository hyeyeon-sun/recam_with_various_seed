import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import os.path as osp

import voc12.dataloader
from misc import torchutils, imutils
import cv2
cudnn.enabled = True

def _work(process_id, model, dataset, args):
    # work는 multiproccessing을 기반으로 작동하는 것으로, proccess_id에 맞는 dataset을 databin으로 갖는다.
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    # 프로세스(각 gpu)에 할당된 것 만큼의 데이터를 불러온다.
    data_loader = DataLoader(databin, shuffle=False, num_workers=4 // n_gpus, pin_memory=False)

    #no_grad() -> auto_grad 기능을 끔으로써 메모리 사용량을 줄이고 연산 속도 증가시킴
    #process_id에 해당되는 gpu device를 할당시킨다.
    with torch.no_grad(), cuda.device(process_id):

        # 모델의 모든 파라미터를 gpu에 로딩한다.
        model.cuda()

        # 해당 worker(여기서는 gpu하나)에 해당하는 voc 이미지를 가져온다.
        # voc 이미지는 name, label, size로 구성되어 있다.
        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            # strided 4를 적용한 뒤 이미지의 사이즈 (작아졌겠지)
            strided_size = imutils.get_strided_size(size, 4)

            # strided_up -> 16으로 strided를 줄였다가 다시 곱해준다 (왜 하는 거지 ..?)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            
            outputs = [model(img[0].cuda(non_blocking=True)) for img in pack['img']] # b x 20 x w x h

            strided_cam = torch.sum(torch.stack([F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # cam을 생성해낸다.
            
            np.save(os.path.join(args.cam_out_dir, img_name.replace('jpg','npy')),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    # cam 모델을 가져옴
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    # cam에 resnet에서 가져온 모델의 파라미터를 저장한다.
    model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    # evaluation 과정에서 사용하지 않아도 되는 불필요한 것들을 끄는 함수임.
    # 나중에 train할때는 model.tarin으로 train 모드로 바꿔줘야 함 !
    model.eval()
    # cpu의 개수
    n_gpus = torch.cuda.device_count()

    # 이미지 path를 받아
    # out = {"name": name_str, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
    #           "label": torch.from_numpy(self.label_list[idx])}
    # 상단처럼 생긴 output을 낸다.
    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list, voc12_root=args.voc12_root, scales=args.cam_scales)
    # gpu의 개수에 맞개 dataset을 쪼갠다 (병렬처리가 가능하도록)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    # gpu를 통한 병렬처리를 할 수 있도록 한다.
    # 이때 모델(cam에 resnet의 모델 파라미터를 불러온 것)과 dataset을 넘겨준다.
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()