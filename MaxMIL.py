import sys
import os
import numpy as np
import random
import openslide 
import pandas as pd
import PIL.Image as Image
import time
import argparse

# ----> torch import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.models as models
# ----> calculate import
from sklearn.metrics import roc_curve, auc
# ----> self import
from model.model_maxmil import resnet34, resnet50
from collections import OrderedDict
from Early_Stopping import EarlyStopping
from utils.maxmil_utils import Inferencedataset, Traindataset, ResNetEncoder, data_prefetcher

np.random.seed(24)
torch.manual_seed(24) # 为cpu设置随机种子
torch.cuda.manual_seed(24) # 为当前的GPU设置随机种子
torch.cuda.manual_seed_all(24) # 为所有的GPU设置随机种子
torch.backends.cudnn.deterministic = True # 保持每次的结果一样
torch.backends.cudnn.benchmark = False # 设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
# 每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的


parser = argparse.ArgumentParser(description = 'standard Max-pooling MIL')
## Path Arguments
parser.add_argument('--train_lib', type=str, default='lib/train.ckpt', help = 'lib to save wsi id of train set')
parser.add_argument('--val_lib', type=str, default='lib/val.ckpt', help = 'lib to save wsi id of val set')
parser.add_argument('--test_lib', type=str, default='lib/test.ckpt', help = 'lib to save wsi id of test set')
parser.add_argument('--output', type=str, default='result', help = 'output directory')
parser.add_argument('--feat_dir', type=str, default = 'feat', help='path to save features')
parser.add_argument('--mil_model', type=str, default='model/mil.pth', help='path to pretrained model')
parser.add_argument('--patch_dir', type=str, default='')
## Experiment Arguments
parser.add_argument('--batch_size', type=int, default=256, help='batch_size: default = 256')
parser.add_argument('--nepochs', type=int, default=30, help='number of epochs')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--test_every', type=int, default=5, help='test on val every')
parser.add_argument('--weights', type=float, default=0.5, help='unbanlanced positive class weight')
parser.add_argument('--k', type=int, default=1, help='topk tiles are assumed to be of the same class as the slide default = 1 standard maxmil')
parser.add_argument('--n', type=int, default=1000, help='select top n tiles')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
## Other Arguments
parser.add_argument('--save_model', default = False, action='store_true')
parser.add_argument('--save_feat', default=False, action='store_true')
parser.add_argument('--feat_format', type=str, choices = ['.csv', '.npy', '.pt'], default='.csv')
parser.add_argument('--load_model', default=False, action='store_true')
parser.add_argument('--is_test', default=False, action='store_true')
parser.add_argument('--save_index', default=False, action='store_true')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--device_ids', type=int, nargs='+', default=[0,1,2,3])

global args, best_acc
args = parser.parse_args()
# torch.cuda.set_device(args.device)
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')


def main():
    best_acc = 0
    model = resnet34(pretrained=True) # 设置基础模型
    model.fc = nn.Linear(model.fc.in_features, 2) # 更换最后的全连接层
    print(model)
    if args.load_model:
        ch = torch.load(args.mil_model, map_location='cpu')
        model.load_state_dict(ch["state_dice"], strict=False)
    model.to(device) # 模型损失和

    # model = nn.DataParallel(model, device_ids=args.device_ids)

    if args.weighes==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights, args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda() # weight:为不平衡类进行加权
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, nmilestones=[10,20], gamma=0.1, last_epoch=-1) # 10,20 epoch的时候lr衰减0.1
    cudnn.benchmark = True # 开启自动搜索优化的算法
    # normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    # load data
    inference_dset = Inferencedataset(args.train_lib, trans)
    inference_loader = torch.utils.data.DataLoader(
        inference_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    if args.val_lib:
        val_dset = Inferencedataset(args.val_lib, trans)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)
    test_dset = Inferencedataset(args.test_lib, trans)
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)


    columns = []
    for i in range(1, 513):
        columns.append('feature' + str(i))

    epoch = 0
    # initialize the model saving path
    model_save_path = args.mil_model
    early_stopping = EarlyStopping(model_path=model_save_path,
                                   patience=3, verbose=True)
    if not args.is_test:
        #loop throuh epochs
        for epoch in range(args.nepochs):
            begin = time.time()
            
            if epoch == args.nepochs-1 and args.save_feat:
                probs, features = inference(epoch, inference_loader, model, args.save_feat)
            else:
                probs, _ = inference(epoch, inference_loader, model)

            topk = group_argtopk(np.array(inference_dset.slideIDX), probs, args.k)
            t_data = inference_dset.maketraindata(topk)
            train_dset = Traindataset(t_data, args.train_lib, trans)
            train_loader = torch.utils.data.DataLoader(
                 train_dset,
                 batch_size=128, shuffle=True,
                 num_workers=args.workers, pin_memory=False)

            if epoch == args.nepochs-1 and args.save_index:
                topn = group_argtopk(np.array(inference_dset.slideIDX), probs, args.n)
                inference_dset.savetopndata(topn, filename=f"select_train")
                if args.save_feat:
                    slideIDX = np.array(train_dset.slideIDX)
                    for i, slide in enumerate(train_dset.slidenames):
                        slidename = '.'.join(os.path.basename(slide).split('.')[0:2])
                        feature = features[slideIDX == i, :]
                    
                    # save features
                    if args.feat_format == '.csv':
                        df = pd.DataFrame(feature, columns=columns)
                        df.to_csv(os.path.join(args.feat_dir, f'{slidename}.csv'))
                    elif args.feat_format == '.npy':
                        np.save(os.path.join(args.feat_dir, f'{slidename}.npy'), feature)
                    elif args.feat_format == '.pt':
                        feature = torch.from_numpy(feature)
                        torch.save(feature, os.path.join(args.feat_dir, f'{slidename}.pt'))
                        
            train_dset.shuffletraindata()
            loss = train(epoch, train_loader, model, criterion, optimizer)
            end = time.time()
            usetime = end-begin
            print('Training\tEpoch: [{}/{}]\tLoss: {}\tUsetime: {:.4f}'.format(epoch+1, args.nepochs, loss, usetime))
            scheduler.step()


            #Validation
            if args.val_lib and (epoch+1) % args.test_every == 0:
                if epoch == args.nepochs-1 and args.save_feat:
                    probs, features = inference(epoch, val_loader, model, args.save_feat)
                else:
                    probs, _ = inference(epoch, val_loader, model)
                topk = group_argtopk(np.array(val_dset.slideIDX), probs, args.k)
                maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
                t_data = val_dset.maketraindata(topk)
                train_dset = Traindataset(t_data, args.val_lib, trans)
                train_loader = torch.utils.data.DataLoader(
                    train_dset,
                    batch_size=128, shuffle=True,
                    num_workers=args.workers, pin_memory=False)
                val_loss = train(epoch, train_loader, model, criterion, optimizer)
                if epoch == args.nepochs-1 and args.save_index:
                    topn = group_argtopk(np.array(val_dset.slideIDX), probs, args.n)
                    val_dset.savetopndata(topn, filename=f"select_val")
                    if args.save_feat:
                        slideIDX = np.array(val_dset.slideIDX)
                        for i, slide in enumerate(val_dset.slidenames):
                            slidename = '.'.join(os.path.basename(slide).split('.')[0:2])
                            feature = features[slideIDX == i, :]
                            
                            # save features
                            if args.feat_format == '.csv':
                                df = pd.DataFrame(feature, columns=columns)
                                df.to_csv(os.path.join(args.feat_dir, f'{slidename}.csv'))
                            elif args.feat_format == '.npy':
                                np.save(os.path.join(args.feat_dir, f'{slidename}.npy'), feature)
                            elif args.feat_format == '.pt':
                                feature = torch.from_numpy(feature)
                                torch.save(feature, os.path.join(args.feat_dir, f'{slidename}.pt'))
                
                pred = [1 if x >= 0.5 else 0 for x in maxs]
                fpr, tpr, thresh = roc_curve(y_true=val_dset.targets, y_score=maxs, pos_label=1)
                roc_auc = auc(fpr, tpr)
                err, fpr, fnr = calc_err(pred, val_dset.targets)

                print("Validation Accuracy: {:.4f}\t AUC: {:.4f}".format(1 - err, roc_auc))
                print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))

                if 1 - err >= best_acc:
                    best_acc = 1 - err

            if args.test_lib and (epoch+1) % args.test_every == 0:
                ## Test
                if epoch == args.nepochs-1 and args.save_feat:
                    probs, features = inference(epoch, test_loader, model, args.save_feat)
                else:
                    probs, _ = inference(epoch, test_loader, model)
                maxs = group_max(np.array(test_dset.slideIDX), probs, len(test_dset.targets))
                if epoch == args.nepochs-1 and args.save_index:
                    topn = group_argtopk(np.array(test_dset.slideIDX), probs, args.n)
                    test_dset.savetopndata(topn, filename=f"select_test")
                    if args.save_feat:
                        slideIDX = np.array(test_dset.slideIDX)
                        for i, slide in enumerate(val_dset.slidenames):
                            slidename = '.'.join(os.path.basename(slide).split('.')[0:2])
                            feature = features[slideIDX == i, :]
                            
                            # save features
                            if args.feat_format == '.csv':
                                df = pd.DataFrame(feature, columns=columns)
                                df.to_csv(os.path.join(args.feat_dir, f'{slidename}.csv'))
                            elif args.feat_format == '.npy':
                                np.save(os.path.join(args.feat_dir, f'{slidename}.npy'), feature)
                            elif args.feat_format == '.pt':
                                feature = torch.from_numpy(feature)
                                torch.save(feature, os.path.join(args.feat_dir, f'{slidename}.pt'))
                
                pred = [1 if x >= 0.5 else 0 for x in maxs]
                fpr, tpr, thresh = roc_curve(y_true=test_dset.targets, y_score=maxs, pos_label=1)
                roc_auc = auc(fpr, tpr)
                err, fpr, fnr = calc_err(pred, test_dset.targets)
                print("Test Accuracy: {:.4f}\t AUC: {:.4f}".format(1 - err, roc_auc))
                print('Test\tError: {}\tFPR: {}\tFNR: {}'.format(err, fpr, fnr))

                ## early stop
                early_stopping(val_loss, best_acc, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                ## Save the model
                if args.save_model:
                    obj = {
                        'state_dict': model.module.state_dict(),
                        'best_acc': best_acc,
                    }
                    torch.save(obj, model_save_path)

    ## Test
    ch = torch.load(model_save_path, map_location='cpu')
    print(ch['best_acc'])
    model.load_state_dict(ch["state_dict"], strict=False)
    probs = inference(epoch, test_loader, model)
    maxs = group_max(np.array(test_dset.slideIDX), probs, len(test_dset.targets))
    if args.save_index:
        topn = group_argtopk(np.array(test_dset.slideIDX), probs, args.n)
        test_dset.savetopndata(topn, filename=f"select_test")
    pred = [1 if x >= 0.5 else 0 for x in maxs]
    fpr, tpr, thresh = roc_curve(y_true=test_dset.targets, y_score=maxs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    err, fpr, fnr = calc_err(pred, test_dset.targets)
    print("Test Accuracy: {:.4f}\t AUC: {:.4f}".format(1 - err, roc_auc))
    miss_wsi(test_dset, pred)



def inference(run, loader, model, save_feat=False):
    model.eval() # model中有BN层 or Dropout, .eval()可以将其关闭以免影响预测结果
    probs = torch.FloatTensor(len(loader.dataset))
    prefetcher = data_prefetcher(loader) # PyTorch 异步数据预读器，以异步地加载数据
    features = torch.Tensor()
    with torch.no_grad():
        input = prefetcher.next()
        i = 0
        while input is not None:
            if i % 1000 == 999:
                print('Inference\tEpoch: [{}/{}]\tBatch:[{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))

            input = input.to(device)
            output, feature = model(input)
            output = F.softmax(output,dim=1)
            probs[i*args.batchsize:i*args.batchsize+input.size(0)] = output.detach()[:,1].clone()
            imput = prefetcher.next()
            i += 1
            if save_feat:
                features = torch.cat((features, feature.cpu()), dim=0)
    return probs.cpu().numpy(), features.cpu().numpy()















if __name__ == "__main__":
    main()