import copy
import cv2

import torch
import torch.nn as nn
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from pytorch_grad_cam import ScoreCAM
from torchvision.models import resnet50
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageColor

import importlib

import voc12.dataloader
from misc import pyutils, torchutils
from torch import autograd
import os

def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss')))

    return


def run(args):
    # args.cam_network -> net.resnet50_cam
    # resnet 50 인코더로 특징을 추출한다. (f(x)를 구해냄.)
    model = getattr(importlib.import_module(args.cam_network), 'Net')()

    # # VOC 데이터 셋 설정 - train data set과 valid data set을 설정한다.
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # resnet에서 파라미터 그룹을 가져온다.
    param_groups = model.trainable_parameters()
    # 파라미터 최적화
    # param_grops[1]이 새로 추가되는 파라미터임
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    # 모델을 병렬로 돌린다.
    model = torch.nn.DataParallel(model).cuda()
    # train은 net.resnet50_cam 의 Net class의 train()메소드 -> 자기 자신의 train을 상속받는다 ..?
    model.train()
    
    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()
    ce = nn.CrossEntropyLoss()

    # epoch -> 전체 데이터 셋을 몇번 사용하여 학습할 것인지
    # for문을 한번 돌 때마다 데이터 셋을 한번 돈다.
    for ep in range(args.cam_num_epoches):
        for step, pack in enumerate(train_data_loader):
            # train dataset에서 이미지를 불러온다.
            img = pack['img']
            img = img.cuda()
            label = pack['label'].cuda(non_blocking=True)
            # x는 model을 거친 이미지다.
            x = model(img)
            optimizer.zero_grad()
            # label과 비교해 loss를 구한다.
            loss = F.multilabel_soft_margin_loss(x, label)
            # loss를 backward로 역전파
            loss.backward()
            avg_meter.add({'loss': loss.item()})

            optimizer.step()
            # 한번 수행한 결과를 출력
            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        validate(model, val_data_loader)
        timer.reset_stage()


    # image_url = "https://farm1.staticflickr.com/6/9606553_ccc7518589_z.jpg"
    # image = np.array(Image.open(requests.get(image_url, stream=True).raw))
    # rgb_img = np.float32(image) / 255
    # input_tensor = preprocess_image(rgb_img,
    #                                 mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    # model = resnet50(pretrained=True)
    # target_layers = [model.layer4[-1]]
    # model = ScoreCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    # grayscale_cam = cam(input_tensor=input_tensor,
    #                     targets=targets)[0, :]
    # cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # resnet 모델의 상태를 저장한다.
    torch.save(model.module.state_dict(), args.cam_weights_name)
    torch.cuda.empty_cache()


###############

def show_result(image, target, preds, sentences, ious, args, height, width, opacity=0.5):
    
    print(image.shape)
    img = image.squeeze().cpu().numpy().transpose(2,3,1,0) # [3, 480, 480] -> [480, 480, 3]
    target = target.squeeze().cpu().numpy().astype(bool) # [1, H, W] -> [H, W]
    # ious = [iou.cpu().numpy() for iou in ious]
    preds = [pred.squeeze(0).cpu().detach().numpy().astype(bool) for pred in preds]
    

    mean = ([0.485, 0.456, 0.406])
    std = ([0.229, 0.224, 0.225])

    # re-nomalize
    for c, (mean_c, std_c) in enumerate(zip(mean, std)):
        img[:,:,c] *= std_c
        img[:,:,c] += mean_c

    color = np.array(ImageColor.getrgb('red'), dtype=np.uint8) / 255# tuple

    gt = copy.copy(img)
    # apply mask
    for c in range(3):
        gt[:,:,c] = np.where(target == 1,
                             gt[:,:,c] * opacity + (1 - opacity) * color[c],
                             gt[:,:,c])


    fig = plt.figure(figsize=(30,10),constrained_layout=True)
    specs = gridspec.GridSpec(nrows=2, ncols= len(sentences) + 1)
    ax1 = fig.add_subplot(specs[0,0])
    # ax1.set_title(f'mean_iou: {np.mean(ious):.2f} \n Image', fontsize=25)
    ax1.axis('off')
    ax1.imshow(img)

    ax2 = fig.add_subplot(specs[0,1])
    ax2.set_title('GT', fontsize=20)
    ax2.axis('off')
    ax2.imshow(gt)

    count = 0
    for pred, sentence in zip(preds, sentences):
        mask_seg = copy.copy(img)

        for c in range(3):
            mask_seg[:,:,c] = np.where(pred == 1,
                                       mask_seg[:,:,c] * opacity + (1 - opacity) * color[c],
                                       mask_seg[:,:,c])

        ax = fig.add_subplot(specs[1, count])
        # ax.set_title(f'IoU: {iou:.2f} \n {sentence}', fontsize=25)
        ax.axis('off')
        ax.imshow(mask_seg)

        count += 1

    os.makedirs(f'./show_result/{args.dataset}_{args.split}_{args.clip_model}_{height}x{width}/', exist_ok=True)
    show_dir = f'./show_result/{args.dataset}_{args.split}_{args.clip_model}_{height}x{width}/M{np.mean(ious):.2f}_{sentence}.jpg'


    plt.savefig(show_dir)
    plt.show()
