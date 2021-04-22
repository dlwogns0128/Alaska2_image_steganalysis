from glob import glob
import os
from datetime import datetime
import time
import random
import argparse

import sklearn
from sklearn.model_selection import GroupKFold
from sklearn import metrics
from skimage import io
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import Normalize

import jpegio as jio

from efficientnet_pytorch import EfficientNet
from catalyst.data.sampler import BalanceClassSampler

try:
    from apex import amp, optimizers
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

DATA_ROOT_PATH = '/data/kaggle/alaska2-image-steganalysis'
    
# 1. SEED setting
# 재현 가능하도록 random seed를 맞추는 작업
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

import warnings
warnings.filterwarnings("ignore")

# 2. Make transform funcitons
def get_train_transforms():
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

# [ALASKA2] Image read function for JPEG
def JPEGdecompressYCbCr_v3(path):
    jpegStruct = jio.read(str(path))

    [col, row] = np.meshgrid(range(8), range(8))
    T = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))
    T[0, :] = T[0, :] / np.sqrt(2)

    img_dims = np.array(jpegStruct.coef_arrays[0].shape)
    n_blocks = img_dims // 8
    broadcast_dims = (n_blocks[0], 8, n_blocks[1], 8)
    
    YCbCr = []
    for i, dct_coeffs, in enumerate(jpegStruct.coef_arrays):

        if i == 0:
            QM = jpegStruct.quant_tables[i]
        else:
            QM = jpegStruct.quant_tables[1]
        
        t = np.broadcast_to(T.reshape(1, 8, 1, 8), broadcast_dims)
        qm = np.broadcast_to(QM.reshape(1, 8, 1, 8), broadcast_dims)
        dct_coeffs = dct_coeffs.reshape(broadcast_dims)
        
        a = np.transpose(t, axes=(0, 2, 3, 1))
        b = (qm * dct_coeffs).transpose(0,2,1,3)
        c = t.transpose(0,2,1,3)
                
        z = a @ b @ c
        z = z.transpose(0,2,1,3)
        YCbCr.append(z.reshape(img_dims))
        
    return np.stack(YCbCr, -1).astype(np.float32)

# 3. Define Dataset class
def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec
# data loader로 데이터를 batch 만큼 가져올 때 수행되는 부분
class DatasetRetriever(Dataset):

    def __init__(self, kinds, image_names, labels, transforms=None):
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index: int):
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]

        image = JPEGdecompressYCbCr_v3(f'{DATA_ROOT_PATH}/{kind}/{image_name}')
        image /= 128.0

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        target = onehot(4, label)
        return image, target

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)
    
# 학습할 때와 검증할 때 데이터를 읽는 방식이 보통 다르므로(transforms, label 등)
class DatasetSubmissionRetriever(Dataset):

    def __init__(self, image_names, transforms=None):
        super().__init__()
        self.image_names = image_names
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_name = self.image_names[index]

        image = JPEGdecompressYCbCr_v3(f'{DATA_ROOT_PATH}/Test/{image_name}')
        image /= 128.0

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image_name, image

    def __len__(self) -> int:
        return self.image_names.shape[0]

# 4. Define Metric for train/validation
class AverageMeter(object):
    # 학습 과정 중에 성능을 보기 편하게 하기 위한 클래스
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# [ALASKA2] weighted AUC
def alaska_weighted_auc(y_true, y_valid):
    """
    https://www.kaggle.com/anokas/weighted-auc-metric-updated
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        if mask.sum() == 0:
            continue

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization

class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0.5,0.5])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)
        y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]
        np.nan_to_num(y_pred, copy=False)
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = alaska_weighted_auc(self.y_true, self.y_pred)

    @property
    def avg(self):
        return self.score

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)

# 5. Define train/validation pipeline Class
class Fitter:
    
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0
        
        self.base_dir = './'
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        self.device = device

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.criterion = LabelSmoothing().to(self.device)
        self.log(f'Fitter prepared. Device is {self.device}')

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                opt_level = config.opt_level,
                cast_model_type = config.cast_model_type,
                patch_torch_functions = config.patch_torch_functions,
                keep_batchnorm_fp32 = config.keep_batchnorm_fp32,
                master_weights = config.master_weights,
                loss_scale = config.loss_scale
                )
        
        #nn.DataParallel is worked only with amp 'O1'
        if torch.cuda.device_count() > 1:
            print("Setting to multi GPU ...")
            self.model = nn.DataParallel(self.model)

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss, final_scores = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss, final_scores = self.validation(validation_loader)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch-{str(final_scores.avg*10000)[:4]}score.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch*.bin'))[:-5]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (images, targets) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                targets = targets.to(self.device).float()
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                final_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss, final_scores

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (images, targets) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            targets = targets.to(self.device).float()
            images = images.to(self.device).float()
            batch_size = images.shape[0]

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()


            final_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss, final_scores
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
            'amp': amp.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        amp.load_state_dict(checkpoint['amp'])
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


# 6. define Model
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b0')
        self.fc = nn.Linear(in_features=self.base._fc.out_features, out_features=4, bias=True)

    def forward(self, x):

        out = self.base(x)
        out = self.fc(out)

        return out
    
# 7. Set config values
from config import TrainGlobalConfig

parser = argparse.ArgumentParser()
parser.add_argument("--resume", default=None, type=str)
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def main():
    args = parser.parse_args()

    if TrainGlobalConfig.is_train:        
        dataset = []
        for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):
            for path in glob('/data/kaggle/alaska2-image-steganalysis/Cover/*.jpg'):
                dataset.append({
                    'kind': kind,
                    'image_name': path.split('/')[-1],
                    'label': label
                })
        random.shuffle(dataset)
        dataset = pd.DataFrame(dataset)

        gkf = GroupKFold(n_splits=5)

        dataset.loc[:, 'fold'] = 0
        for fold_number, (train_index, val_index) in enumerate(gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):
            dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number   

        # --------------------
        fold_number = 1

        train_dataset = DatasetRetriever(
            kinds=dataset[dataset['fold'] != fold_number].kind.values,
            image_names=dataset[dataset['fold'] != fold_number].image_name.values,
            labels=dataset[dataset['fold'] != fold_number].label.values,
            transforms=get_train_transforms(),
        )

        validation_dataset = DatasetRetriever(
            kinds=dataset[dataset['fold'] == fold_number].kind.values,
            image_names=dataset[dataset['fold'] == fold_number].image_name.values,
            labels=dataset[dataset['fold'] == fold_number].label.values,
            transforms=get_valid_transforms(),
        )
        
        # --------------------
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler = BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
            batch_size=TrainGlobalConfig.batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=TrainGlobalConfig.num_workers,
        )

        val_loader = torch.utils.data.DataLoader(
            validation_dataset, 
            batch_size=TrainGlobalConfig.batch_size,
            num_workers=TrainGlobalConfig.num_workers,
            shuffle=False,
            sampler=SequentialSampler(validation_dataset),
            pin_memory=True,
        )
        
        # --------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = CustomModel()
        model = model.cuda()    
    
        fitter = Fitter(model=model, device=device, config=TrainGlobalConfig)
        if args.resume is not None:
            fitter.load(args.resume)

        fitter.fit(train_loader, val_loader)

    else:

       
        test_dataset = DatasetSubmissionRetriever(
            image_names=np.array([path.split('/')[-1] for path in glob('/data/kaggle/alaska2-image-steganalysis/Test/*.jpg')]),
            transforms=get_valid_transforms(),
        )
        
        # --------------------
        data_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
        )
        
        # --------------------
        model = CustomModel()
        model = model.cuda() 
        
        model = nn.DataParallel(model)

        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict']) 
        
        model.eval()
        
        # --------------------
        result = {'Id': [], 'Label': []}
        for step, (image_names, images) in enumerate(data_loader):
            print(step, end='\r')

            y_pred = model(images.cuda())
            y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]

            result['Id'].extend(image_names)
            result['Label'].extend(y_pred)

        submission = pd.DataFrame(result)
        submission.to_csv('output_fold_{}.csv'.format(fold_number), index=False)
        
if __name__ == "__main__":
    main()




