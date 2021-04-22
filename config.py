import torch

class TrainGlobalConfig:
    # -------------------
    is_train = False
    
    # -------------------
    num_workers = 16
    batch_size = 256
    n_epochs = 50
    lr = 0.001

    # -------------------
    verbose = True
    verbose_step = 1
    
    # ------------------- AMP
    opt_level = 'O1'
    cast_model_type = None
    patch_torch_functions= True
    keep_batchnorm_fp32 = None
    master_weights = None
    loss_scale = 'dynamic'

    # --------------------
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.1,
        patience=2,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-10,#1e-8,
        eps=1e-08
    )