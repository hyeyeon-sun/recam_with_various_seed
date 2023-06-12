import torchvision.transforms.functional as TF


def show ():
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    # labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    preds = []
    labels = []
    n_images = 0
    cam_dict = np.load(os.path.join(args.cam_out_dir, datasets.ids[0] + '.npy'), allow_pickle=True).item()
    cams = cam_dict['high_res']
    cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
    keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
    cls_labels = np.argmax(cams, axis=0)
    cls_labels = keys[cls_labels]
    preds.append(cls_labels.copy())
    labels.append(dataset.get_example_by_keys(1, (1,))[0])

    


    # for i, id in enumerate(dataset.ids):
    #     n_images += 1
    #     cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
    #     cams = cam_dict['high_res']
    #     cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
    #     keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
    #     cls_labels = np.argmax(cams, axis=0)
    #     cls_labels = keys[cls_labels]
    #     preds.append(cls_labels.copy())
    #     labels.append(dataset.get_example_by_keys(i, (1,))[0])

    # confusion = calc_semantic_segmentation_confusion(preds, labels)


def show_result(image, target, preds, sentences, ious, args, height, width, opacity=0.5):
    img = image.squeeze().cpu().numpy().transpose(1,2,0) # [3, 480, 480] -> [480, 480, 3]
    target = target.squeeze().cpu().numpy().astype(bool) # [1, H, W] -> [H, W]
    ious = [iou.cpu().numpy() for iou in ious]
    preds = [pred.squeeze(0).cpu().numpy().astype(bool) for pred in preds]
    

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
    ax1.set_title(f'mean_iou: {np.mean(ious):.2f} \n Image', fontsize=25)
    ax1.axis('off')
    ax1.imshow(img)

    ax2 = fig.add_subplot(specs[0,1])
    ax2.set_title('GT', fontsize=20)
    ax2.axis('off')
    ax2.imshow(gt)

    count = 0
    for pred, sentence, iou in zip(preds, sentences, ious):
        mask_seg = copy.copy(img)

        for c in range(3):
            mask_seg[:,:,c] = np.where(pred == 1,
                                       mask_seg[:,:,c] * opacity + (1 - opacity) * color[c],
                                       mask_seg[:,:,c])

        ax = fig.add_subplot(specs[1, count])
        ax.set_title(f'IoU: {iou:.2f} \n {sentence}', fontsize=25)
        ax.axis('off')
        ax.imshow(mask_seg)

        count += 1

    os.makedirs(f'./show_result/{args.dataset}_{args.split}_{args.clip_model}_{height}x{width}/', exist_ok=True)
    show_dir = f'./show_result/{args.dataset}_{args.split}_{args.clip_model}_{height}x{width}/M{np.mean(ious):.2f}_H{np.max(ious):.2f}_L{np.min(ious):.2f}_{sentence}.jpg'


    plt.savefig(show_dir)
    plt.show()