# from numpy import deprecate_with_doc


train = dict(eval_step=1024,
             total_steps=1024*512,
             trainer=dict(type="PCL",
                          threshold=0.6,
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
num_classes = 128
# seed = 1

model = dict(
     type="resnet50",
     low_dim=64,
     num_class=num_classes,
     proj=True,
     width=1, 
     in_channel=3
)

seminat_mean = (0.4732, 0.4828, 0.3779)
seminat_std = (0.2348, 0.2243, 0.2408)

data = dict(
    type="DomainNet",
    num_workers=1,
    # num_worker = n_gpus
    batch_size=10,
    l_anno_file="/data/tuky/DATASET/multi/l_train/anno.txt",
    u_anno_file="/data/tuky/DATASET/multi/u_train/u_train.txt",
    v_anno_file="/data/tuky/DATASET/multi/val/anno.txt",
    mu=10,
    # mu = labeled / unlabeled

    lpipelines=[[
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomCrop", size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=seminat_mean, std=seminat_std)
    ]],

    upipelinse=[[
        # weak augment
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomCrop", size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=seminat_mean, std=seminat_std)],
        [
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomCrop", size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=seminat_mean, std=seminat_std)],
        [
        dict(type="RandomHorizontalFlip"),
        dict(type="RandomCrop", size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        dict(type="RandAugmentMC", n=2, m=10),
        dict(type="ToTensor"),
        dict(type="Normalize", mean=seminat_mean, std=seminat_std)]],
    
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

log = dict(interval=64)
ckpt = dict(interval=1000)
evaluation = dict(eval_both=True)

# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.001, nesterov=True)