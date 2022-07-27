# from numpy import deprecate_with_doc


train = dict(eval_step=1024,
             total_steps=2**20,
             trainer=dict(type="PCL",
                          threshold=0.80,
                          T=1.,
                          temperature=0.07,
                          lambda_u=1., # lambda_u
                          lambda_contrast=2,
                          loss_x=dict(
                              type="cross_entropy",
                              reduction="mean"),
                          loss_u=dict(
                              type="cross_entropy",
                              reduction="none"),
                          ))
num_classes = 126
# seed = 1

# model = dict(
#      type="resnet50",
#      low_dim=64,
#      # low_dim=126,
#      num_class=num_classes,
#      proj=True,
#      width=1, 
#      in_channel=3
# )
model = dict(
     type="wideresnet",
     depth=28,
     num_classes=num_classes,
     widen_factor=2,
     dropout=0
)

multi_mean = (0.4732, 0.4828, 0.3779)
multi_std = (0.2348, 0.2243, 0.2408)

data = dict(
    type="DomainNet",
    num_workers=4,
    # num_worker = n_gpus
    batch_size=8,
    l_anno_file="/data/tuky/DATASET/multi/l_train/anno.txt",
    u_anno_file="/data/tuky/DATASET/multi/u_train/u_train.txt",
    v_anno_file="/data/tuky/DATASET/multi/val/anno.txt",
    mu=7,
    # mu = labeled / unlabeled

    lpipelines=[[
        dict(type="RandomHorizontalFlip"),
        # 随机水平翻转
        dict(type="RandomCrop", size=64, padding=int(32 * 0.125), padding_mode='reflect'),
        # dict(type="RandomResizedCrop", size=224, scale=(0.2, 1.0)),
        # 随机剪裁一个area，再进行resize
        dict(type="ToTensor"),
        dict(type="Normalize", mean=multi_mean, std=multi_std)
    ]],

    upipelinse=[[#weak 1
            dict(type="RandomHorizontalFlip"),
            dict(type="Resize", size=256),
            dict(type="CenterCrop", size=64),
            # dict(type="CenterCrop", size=224),
            # 从中心裁剪图片成为所需的尺寸
            dict(type="ToTensor"),
            dict(type="Normalize", mean=multi_mean, std=multi_std)],
        [ #strong 1
            dict(type="RandomResizedCrop", size=64, scale=(0.2, 1.0)),
            # dict(type="RandomResizedCrop", size=224, scale=(0.2, 1.0)),
            dict(type="RandomHorizontalFlip"),
            dict(type="RandAugmentMC", n=2, m=10),
            dict(type="RandomApply", 
                transforms=[
                    dict(type="ColorJitter",
                         brightness=0.4,
                         contrast=0.4,
                         saturation=0.4,
                         hue=0.1),
                         ],
                         p=0.8),

            dict(type="RandomGrayscale", p=0.2),
            dict(type="ToTensor"),
            dict(type="Normalize", mean=multi_mean, std=multi_std)],
        [ #strong 2
            dict(type="RandomResizedCrop", size=64, scale=(0.2, 1.0)),
            # dict(type="RandomResizedCrop", size=224, scale=(0.2, 1.0)),
            dict(type="RandomHorizontalFlip"),
            dict(type="RandAugmentMC", n=2, m=10),
            dict(type="RandomApply", 
                transforms=[
                    dict(type="ColorJitter",
                         brightness=0.4,
                         contrast=0.4,
                         saturation=0.4,
                         hue=0.1),
                         ],
                         p=0.8),

            dict(type="RandomGrayscale", p=0.2),
            dict(type="ToTensor"),
            dict(type="Normalize", mean=multi_mean, std=multi_std)]],
    
    vpipeline=[
        dict(type="Resize", size=256),
        dict(type="CenterCrop", size=64),
        # dict(type="CenterCrop", size=224),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=multi_mean, std=multi_std)
    ])

scheduler = dict(
    type='cosine_schedule_with_warmup',
    num_warmup_steps=0,
    num_training_steps=train['total_steps']
)

ema = dict(use=True, pseudo_with_ema=False, decay=0.999)
#apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
#"See details at https://nvidia.github.io/apex/amp.html
amp = dict(use=False, opt_level="O1")

log = dict(interval=1)
ckpt = dict(interval=1000)
evaluation = dict(eval_both=True)

# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.001, nesterov=True)