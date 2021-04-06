import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import GradualWarmupSchedulerV2
# import apex
# from apex import amp
from dataset2 import get_df, get_transforms, MelanomaDataset
from models import Effnet_Melanoma_DANN, Resnest_Melanoma, Seresnext_Melanoma


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--enet-type', type=str, default='tf_efficientnet_b4_ns')
    parser.add_argument('--kernel-type', type=str,default="tf_efficientnet_b4_ns_size512_outdim9_meta_bs16_epoch15") #??
    parser.add_argument('--data-dir-2019', type=str, default='/home/data/ISIC/ISIC2019/')
    parser.add_argument('--data-dir-2020', type=str, default='/home/data/ISIC/ISIC2020/')
    parser.add_argument('--data-dir-2018', type=str, default='/home/data/ISIC/ISIC2018_Task3/')
    parser.add_argument('--out-dim', type=int, default=9)
    parser.add_argument('--use-meta', action='store_true',default=False)
    parser.add_argument('--image-size', type=int, default=256) # resize后的图像大小
    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--DEBUG', default=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    parser.add_argument('--init-lr', type=float, default=3e-5)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--use-amp', default=False)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--DANN', default=True)
    args, _ = parser.parse_known_args()
    return args
def get_trans(img, I):

    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(model, loader, mel_idx, is_ext=None, n_test=1, get_output=False):

    model.eval()
    class_val_loss = []
    barrier_val_loss = []
    CLASS_LOGITS = []
    BARRIER_LOGITS = []
    CLASS_PROBS = []
    BARRIER_PROBS = []
    CLASS_TARGETS = []
    BARRIER_TARGETS = []
    if args.DANN:
        with torch.no_grad():
            for (data, target) in tqdm(loader):
                if args.use_meta:
                    data, meta = data
                    target_class, target_barrier = target
                    data, meta, target_class, target_barrier = data.to(device), meta.to(device), target_class.to(device), target_barrier.to(device)
                    class_logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    class_probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    barrier_logits = torch.zeros((data.shape[0], 2)).to(device)
                    barrier_probs = torch.zeros((data.shape[0], 2)).to(device)
                    for I in range(n_test):
                        l1,l2 = model(get_trans(data, I), meta)
                        class_logits += l1
                        barrier_logits += l2
                        class_probs += l1.softmax(1)
                        barrier_probs += l2.softmax(1)
                else:
                    target_class, target_barrier = target
                    data, target_class, target_barrier = data.to(device), target_class.to(device), target_barrier.to(device)
                    class_logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    class_probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    barrier_logits = torch.zeros((data.shape[0], 2)).to(device)
                    barrier_probs = torch.zeros((data.shape[0], 2)).to(device)
                    for I in range(n_test):
                        l1,l2 = model(get_trans(data, I))
                        class_logits += l1
                        barrier_logits += l2
                        class_probs += l1.softmax(1)
                        barrier_probs += l2.softmax(1)      
                class_logits /= n_test
                barrier_logits /= n_test
                class_probs /= n_test
                barrier_probs /= n_test
                CLASS_LOGITS.append(class_logits.detach().cpu())
                CLASS_PROBS.append(class_probs.detach().cpu())
                CLASS_TARGETS.append(target_class.detach().cpu())
                BARRIER_LOGITS.append(barrier_logits.detach().cpu())
                BARRIER_PROBS.append(barrier_probs.detach().cpu())
                BARRIER_TARGETS.append(target_barrier.detach().cpu())
                class_loss = class_criterion(class_logits, target_class)
                class_val_loss.append(class_loss.detach().cpu().numpy())
                barrier_loss = barrier_criterion(barrier_logits, target_barrier)
                barrier_val_loss.append(barrier_loss.detach().cpu().numpy())
                
        class_val_loss = np.mean(class_val_loss)
        barrier_val_loss = np.mean(barrier_val_loss)

        CLASS_LOGITS = torch.cat(CLASS_LOGITS).numpy()
        CLASS_PROBS = torch.cat(CLASS_PROBS).numpy()
        CLASS_TARGETS = torch.cat(CLASS_TARGETS).numpy()
        BARRIER_LOGITS = torch.cat(BARRIER_LOGITS).numpy()
        BARRIER_PROBS = torch.cat(BARRIER_PROBS).numpy()
        BARRIER_TARGETS = torch.cat(BARRIER_TARGETS).numpy()

        if get_output:
            return CLASS_LOGITS, CLASS_PROBS, BARRIER_LOGITS, BARRIER_PROBS
        else:
            class_acc = (CLASS_PROBS.argmax(1) == CLASS_TARGETS).mean() * 100.
            class_auc = roc_auc_score((CLASS_TARGETS == mel_idx).astype(float), CLASS_PROBS[:, mel_idx])
            # class_auc_20 = roc_auc_score((CLASS_TARGETS[is_ext == 0] == mel_idx).astype(float), CLASS_PROBS[is_ext == 0, mel_idx])
            class_auc_20 = class_auc
            # 这里存疑
            barrier_acc = (BARRIER_PROBS.argmax(1) == BARRIER_TARGETS).mean() * 100.
            barrier_auc = roc_auc_score((BARRIER_TARGETS == 0).astype(float), BARRIER_PROBS[:, 0])
            # barrier_auc_20 = roc_auc_score((BARRIER_TARGETS[is_ext == 0] == 0).astype(float), BARRIER_PROBS)
            barrier_auc_20 = barrier_auc
            return class_val_loss, class_acc, class_auc, class_auc_20, barrier_val_loss, barrier_acc, barrier_auc, barrier_auc_20
    else:
        with torch.no_grad():
            for (data, target) in tqdm(loader):
                if args.use_meta:
                    data, meta = data
                    target_class = target
                    data, meta, target_class = data.to(device), meta.to(device), target_class.to(device)
                    class_logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    class_probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    for I in range(n_test):
                        l1 = model(get_trans(data, I), meta)
                        class_logits += l1
                        class_probs += l1.softmax(1)
                else:
                    target_class = target
                    data, target_class = data.to(device), target_class.to(device)
                    class_logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    class_probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    for I in range(n_test):
                        l1 = model(get_trans(data, I))
                        class_logits += l1
                        class_probs += l1.softmax(1)
                class_logits /= n_test
                class_probs /= n_test
                CLASS_LOGITS.append(class_logits.detach().cpu())
                CLASS_PROBS.append(class_probs.detach().cpu())
                CLASS_TARGETS.append(target_class.detach().cpu())
                class_loss = class_criterion(class_logits, target_class)
                class_val_loss.append(class_loss.detach().cpu().numpy())
                
        class_val_loss = np.mean(class_val_loss)

        CLASS_LOGITS = torch.cat(CLASS_LOGITS).numpy()
        CLASS_PROBS = torch.cat(CLASS_PROBS).numpy()
        CLASS_TARGETS = torch.cat(CLASS_TARGETS).numpy()

        if get_output:
            return CLASS_LOGITS, CLASS_PROBS
        else:
            class_acc = (CLASS_PROBS.argmax(1) == CLASS_TARGETS).mean() * 100.
            class_auc = roc_auc_score((CLASS_TARGETS == mel_idx).astype(float), CLASS_PROBS[:, mel_idx])
            class_auc_20 = class_auc
            # class_auc_20 = roc_auc_score((CLASS_TARGETS[is_ext == 0] == mel_idx).astype(float), CLASS_PROBS[is_ext == 0, mel_idx])
            return class_val_loss, class_acc, class_auc, class_auc_20
          


def train_epoch(model, loader, optimizer):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar: # [tensor(batchsize, 3, 512, 512),tensor(batchsize, 14)]  tensor(2)
        optimizer.zero_grad()
        if args.DANN:
            if args.use_meta:
                data, meta = data
                target_class, target_barrier = target
                data, meta, target_class, target_barrier = data.to(device), meta.to(device), target_class.to(device), target_barrier.to(device)
                class_out,barrier_out = model(data, meta)
            else:
                target_class, target_barrier = target
                data, target_class, target_barrier = data.to(device), target_class.to(device), target_barrier.to(device)
                class_out,barrier_out = model(data)        
            class_loss = class_criterion(class_out, target_class)
            barrier_loss = barrier_criterion(barrier_out, target_barrier)
            loss = class_loss + barrier_loss
        else:
            if args.use_meta:
                data, meta = data
                target_class = target
                data, meta, target_class = data.to(device), meta.to(device), target_class.to(device)
                class_out = model(data, meta)
            else:
                target_class = target
                data, target_class = data.to(device), target_class.to(device)
                class_out = model(data)        
            class_loss = class_criterion(class_out, target_class)
            loss = class_loss

        if not args.use_amp:
            loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        if args.image_size in [896,576]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss


def run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx):

    if args.DEBUG:
        args.n_epochs = 2
        df_train = df[df['fold'] != fold].sample(args.batch_size * 3)
        df_valid = df[df['fold'] == fold].sample(args.batch_size * 3)
    else:
        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold]

    dataset_train = MelanomaDataset(df_train, 'train', meta_features, transform=transforms_train, DANN=args.DANN)
    dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features, transform=transforms_val, DANN=args.DANN)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=args.num_workers) # 随机不重复采样 
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)
    model = ModelClass(
        args.enet_type,
        n_meta_features=n_meta_features,
        n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
        out_dim=args.out_dim,
        DANN=args.DANN,
        pretrained=True
    )
    ###############
    if DP:
        model = apex.parallel.convert_syncbn_model(model)
    ##############
    model = model.to(device)

    auc_max = 0.
    auc_20_max = 0.
    if args.DEBUG:
        model_file  = os.path.join(args.model_dir+'/debug', f'{args.kernel_type}_best_fold{fold}.pth')
        model_file2 = os.path.join(args.model_dir+'/debug', f'{args.kernel_type}_best_20_fold{fold}.pth')
        model_file3 = os.path.join(args.model_dir+'/debug', f'{args.kernel_type}_final_fold{fold}.pth')
    else:
        model_file  = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
        model_file2 = os.path.join(args.model_dir, f'{args.kernel_type}_best_20_fold{fold}.pth')
        model_file3 = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if DP:
        #################
        model = nn.DataParallel(model)
        #################
#     scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs - 1)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    
    print(len(dataset_train), len(dataset_valid))
    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Epoch {epoch}', f'Fold {fold}')
#         scheduler_warmup.step(epoch - 1)

        train_loss = train_epoch(model, train_loader, optimizer)
        if args.DANN:
            class_val_loss, class_acc, class_auc, class_auc_20, barrier_val_loss, barrier_acc, barrier_auc, barrier_auc_20 = val_epoch(model, valid_loader, mel_idx, is_ext=df_valid['is_ext'].values)
            content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, class valid loss: {(class_val_loss):.5f}, acc: {(class_acc):.4f}, auc: {(class_auc):.6f}, auc_20: {(class_auc_20):.6f}.'
            content += '\n' + time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, barrier valid loss: {(barrier_val_loss):.5f}, acc: {(barrier_acc):.4f}, auc: {(barrier_auc):.6f}, auc_20: {(barrier_auc_20):.6f}.'
            print(content)
            if args.DEBUG:
                with open(os.path.join(args.log_dir+'/debug', f'log_{args.kernel_type}.txt'), 'a') as appender:
                    appender.write(content + '\n')
            else:
                with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
                    appender.write(content + '\n')

            scheduler_warmup.step()    
            if epoch==2: scheduler_warmup.step() # bug workaround   
                
            if class_auc > auc_max:
                print('auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_max, class_auc))
                torch.save(model.state_dict(), model_file)
                auc_max = class_auc
            if class_auc_20 > auc_20_max:
                print('auc_20_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_20_max, class_auc_20))
                torch.save(model.state_dict(), model_file2)
                auc_20_max = class_auc_20
        else:
            class_val_loss, class_acc, class_auc, class_auc_20 = val_epoch(model, valid_loader, mel_idx, is_ext=df_valid['is_ext'].values)

    torch.save(model.state_dict(), model_file3)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    df, df_test, meta_features, n_meta_features, mel_idx = get_df(
        args.kernel_type,
        args.out_dim,
        args.data_dir_2020,
        args.data_dir_2019,
        args.data_dir_2018,
        args.use_meta
    )

    transforms_train, transforms_val = get_transforms(args.image_size)

    folds = [int(i) for i in args.fold.split(',')]
    for fold in folds:
        # fold 作为验证集
        run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx)
        break


if __name__ == '__main__':

    args = get_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir+'/debug', exist_ok=True)
    os.makedirs(args.log_dir+'/debug', exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    if args.enet_type == 'resnest101':
        ModelClass = Resnest_Melanoma
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext_Melanoma
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet_Melanoma_DANN
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    set_seed()

    device = torch.device('cuda')
    class_criterion = nn.CrossEntropyLoss()
    if args.DANN:
        barrier_criterion = nn.CrossEntropyLoss()

    main()