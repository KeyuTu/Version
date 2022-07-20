"""
althorithsm: PCL
dataset : DomainNet
backbone: resnet50
batch per GPU: 16
num of GPUs: 1
mu: 10
"""

# from turtle import width


train = dict(eval_step=1024,
             total_steps=1024*512,
             trainer=dict(type="PCL",
                          threshold=0.95,
                          T=1.,
                          temperature=0.07,
                          lambda_u=1.,
                          lambda_contrast=1,
                          loss_x=dict(
                              type="cross_entropy",
                              reduction="mean"),
                          loss_u=dict(
                              type="cross_entropy",
                              reduction="none"),
                          ))
num_classes=128

model = dict(
    type="resnet50",
    num_classes=num_classes,
    proj=False,
    width=2, 
    in_channel=3
)

multi_mean = [0.4732, 0.4828, 0.3779]
multi_std = [0.2348, 0.2243, 0.2408]

data = dict(
    type="DomainNet",
    num_workers=1,
    batch_size=16,
    l_anno_file="/data/tuky/DATASET/multi/l_train/anno.txt",
    u_anno_file="/data/tuky/DATASET/multi/u_train/u_train.txt",
    v_anno_file="/data/tuky/DATASET/multi/val/anno.txt",
    mu=10,

    lpipelines=[[
        dict(type="RandomHorizontalFlip", p=0.5),
        dict(type="RandomResizedCrop", size=224, scale=(0.2, 1.0)),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=multi_mean, std=multi_std)
    ]],
    upipelinse=[[
        dict(type="RandomHorizontalFlip"),
        dict(type="Resize", size=256),
        dict(type="CenterCrop", size=224),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=multi_mean, std=multi_std)],
        [
            dict(type="RandomHorizontalFlip"),
            dict(type="RandomResizedCrop", size=224, scale=(0.2, 1.0)),
            dict(type="RandAugmentMC", n=2, m=10),
            dict(type="ToTensor"),
            dict(type="Normalize", mean=multi_mean, std=multi_std)
        ],
        [
            dict(type="RandomHorizontalFlip"),
            dict(type="RandomResizedCrop", size=224, scale=(0.2, 1.0)),
            dict(type="RandAugmentMC", n=2, m=10),
            dict(type="ToTensor"),
            dict(type="Normalize", mean=multi_mean, std=multi_std)
    ]],

    vpipeline=[
        dict(type="Resize", size=256),
        dict(type="CenterCrop", size=224),
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

log = dict(interval=50)
ckpt = dict(interval=1)
evaluation = dict(eval_both=True)

# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.001, nesterov=True)
