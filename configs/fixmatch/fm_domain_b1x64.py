# from numpy import deprecate_with_doc


train = dict(eval_step=1024,
             total_steps=1024*512,
             trainer=dict(type="FixMatch",
                          threshold=0.95,
                          T=1.,
                          # temperature=0.07,
                          lambda_u=1., # lambda_u
                          # lambda_contrast=2, # lambda_c
                          # contrast_with_softlabel=True,
                          # contrast_left_out=True,
                          # contrast_with_thresh=0.9, # T_push
                          loss_x=dict(
                              type="cross_entropy",
                              reduction="mean"),
                          loss_u=dict(
                              type="cross_entropy",
                              reduction="none"),
                          ))
num_classes = 128
# seed = 1

model = dict(
     type="wideresnet",
     depth=28,
     num_classes=num_classes,
     widen_factor=2,
     dropout=0
)

seminat_mean = (0.4732, 0.4828, 0.3779)
seminat_std = (0.2348, 0.2243, 0.2408)

data = dict(
    type="DomainNet",
    num_workers=1,
    batch_size=12,
    l_anno_file="/data/tuky/DATASET/multi/l_train/anno.txt",
    u_anno_file="/data/tuky/DATASET/multi/u_train/u_train.txt",
    v_anno_file="/data/tuky/DATASET/multi/val/anno.txt",
    mu=7,

    lpipelines=[[
        dict(type="RandomHorizontalFlip", p=0.5),
        dict(type="RandomResizedCrop", size=224, scale=(0.2, 1.0)),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=seminat_mean, std=seminat_std)
    ]],
    upipelinse=[[
        dict(type="RandomHorizontalFlip"),
        dict(type="Resize", size=256),
        dict(type="CenterCrop", size=224),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=seminat_mean, std=seminat_std)
    ],
                [
                    dict(type="RandomHorizontalFlip"),
                    dict(type="RandomResizedCrop", size=224, scale=(0.2, 1.0)),
                    dict(type="RandAugmentMC", n=2, m=10),
                    dict(type="ToTensor"),
                    dict(type="Normalize", mean=seminat_mean, std=seminat_std)
                ],
                [
                    dict(type="RandomResizedCrop", size=224, scale=(0.2, 1.0)),
                    dict(type="RandomHorizontalFlip"),
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
                    dict(type="ToTensor")
                ]],
    vpipeline=[
        dict(type="Resize", size=256),
        dict(type="CenterCrop", size=224),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=seminat_mean, std=seminat_std)
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
# evaluation = dict(eval_both=True)

# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.001, nesterov=True)