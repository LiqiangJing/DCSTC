





import argparse
import os
import random
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import sys
from NT_Xent import NTXentLoss
from transformers import AdamW
from read_data import *
from contro_bert import ClassificationBert
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch Base Models')
def str2bool(v):
    return v.lower() in ('true', '1')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch_size', default=64, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--batch-size-u', default=8, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--use_mask', type=str2bool, default=False,
                    help="Whether to mask")
parser.add_argument('--lrmain', '--learning-rate-bert', default=0.000005, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate for models')
parser.add_argument("--threshold", default=0.95, type=float,
                    help="Threshold for mask")

parser.add_argument('--n-labeled', type=int, default=200,
                    help='Number of labeled data')
parser.add_argument('--un-labeled', default=5000, type=int,
                    help='number of unlabeled data')
parser.add_argument('--val-iteration', type=int, default=1000,
                    help='Number of labeled data')
parser.add_argument('--lambda-l1', default=1, type=float,
                    help='weight for loss term of labeled data')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='augment labeled training data')

parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--data-path', type=str, default='../yahoo_answers_csv/',
                    help='path to data folders')

parser.add_argument('--lambda-u', default=1, type=float,
                    help='weight for consistency loss term of unlabeled data')

parser.add_argument('--lambda_contra', default=None, type=float,
                    help='weight for contra loss term of unlabeled data')

parser.add_argument('--data_parallel', default=False, type=bool, metavar='N',
                    help='Weather to use multi-gpu')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

best_acc = 0
total_steps = 0
flag = 0


def get_device( ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on:", device)
    return device
devices = get_device()
nt_xent_criterion = NTXentLoss(devices, args.batch_size_u, 0.5, True)
def main():
    global best_acc
    # Read dataset and build dataloaders
    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels = get_data(
        args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=args.train_aug)
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    unlabeled_trainloader = Data.DataLoader(
        dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=args.test_batch_size, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False)

    tensorboard_dir = './logs/fixmatch_contra'
    print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    configure(tensorboard_dir)

    # Define the model, set the optimizer
    model = ClassificationBert(n_labels).cuda()

    if args.data_parallel:
        model = nn.DataParallel(model)
        optimizer = AdamW(
            [
                {"params": model.module.bert.parameters(), "lr": args.lrmain},
                {"params": model.module.linear.parameters(), "lr": args.lrlast},
            ])
    else:
        optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": args.lrmain},
            {"params": model.linear.parameters(), "lr": args.lrlast},
        ])


    scheduler = None
    #WarmupConstantSchedule(optimizer, warmup_steps=num_warmup_steps)

    criterion = nn.CrossEntropyLoss()
    train_criterion = SemiLoss()
    test_accs = []
    accs_epoch= []

    val_losses = []
    # Start training
    for epoch in range(args.epochs):

        train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
              scheduler, train_criterion, epoch, n_labels, args.train_aug)

        val_loss, val_acc = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')
        val_losses.append(val_loss)
        print("epoch {}, val acc {}, val_loss {}".format(
            epoch, val_acc, val_loss))
        log_value('val_acc', val_acc, epoch+1)
        log_value('val_loss', val_loss, epoch+1)
        if val_acc >= best_acc:
            best_acc = val_acc
            test_loss, test_acc = validate(
                test_loader, model, criterion, epoch, mode='Test Stats ')
            test_accs.append(test_acc)
            accs_epoch.append(epoch)
            print("epoch {}, test acc {},test loss {}".format(
                epoch, test_acc, test_loss))

        print('Epoch: ', epoch)

        print('Best acc:')
        print(best_acc)

        print('Test acc:')
        print(test_accs,accs_epoch)

        print('val losses:')

        print(val_losses)

    print("Finished training!")
    print('Best acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)




def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, scheduler, criterion, epoch, n_labels, train_aug=False):
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()

    global total_steps
    global flag
    mask_total = 0
    with tqdm(total=args.val_iteration) as pbar:
        for batch_idx in range(args.val_iteration):
            total_steps += 1

            try:
                inputs_x, targets_x, inputs_x_length = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x, inputs_x_length = labeled_train_iter.next()

            try:
                (inputs_u, inputs_u1,  inputs_ori), (length_u,
                                                     length_u1, length_ori) = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (inputs_u, inputs_u1, inputs_ori), (length_u,
                                                    length_u1, length_ori) = unlabeled_train_iter.next()


            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u1 = inputs_u.cuda(), inputs_u1.cuda()
            inputs_ori = inputs_ori.cuda()
            logits = model(inputs_x)
            outputs_u, zis = model(inputs_u, contro=True)
            outputs_u1, zjs = model(inputs_u1, contro=True)

            with torch.no_grad():

                # outputs_ori = model(inputs_ori)
                # p = torch.softmax(outputs_ori, dim=1)
                # pt = p ** (1 / args.T)
                # targets_u = pt / pt.sum(dim=1, keepdim=True)
                # targets_u = targets_u.detach()

                outputs_ori = model(inputs_ori)
                prob_ori = F.softmax(outputs_ori, dim=-1)

                # confidence-based masking

                max_probs, targets_u = torch.max(prob_ori, dim=-1)
                if args.use_mask:
                    mask = max_probs.ge(args.threshold).float()
                    if torch.cuda.is_available():
                        mask = mask.cuda()
                    num_notmasked = torch.sum(mask).cpu().numpy()
                    mask_total = mask_total + num_notmasked

                else:
                    mask = None

                targets_u = targets_u.detach()

            Lx, Lu, Lc = criterion(logits, targets_x, outputs_u, outputs_u1, targets_u, mask,zis,zjs)

            loss = args.lambda_l1*Lx + args.lambda_u * Lu + linear_rampup(epoch)*Lc
            # loss = Lx + args.lambda_u * Lu

            #max_grad_norm = 1.0
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            if args.use_mask:
                pbar.set_description(
                    "epoch {}, step {}, loss {:.4f}, Lx {:.4f}, Lu {:.4f}, Lc {:.4f},Mask {:.4f},Totalm {}".format(
                        epoch, batch_idx, loss.item(), Lx.item(), Lu.item(), Lc.item(),mask.mean().item(), mask_total)
                )

            else:
                pbar.set_description(
                    "epoch {}, step {}, loss {:.4f}, Lx {:.4f}, Lu {:.4f}".format(
                        epoch, batch_idx, loss.item(), Lx.item(), Lu.item())
                )
            pbar.update(1)

    log_value('mask',mask_total,epoch+1)

def validate(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0

        for batch_idx, (inputs, targets, length) in enumerate(tqdm(valloader)):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            # if batch_idx == 0:
            #     print("Sample some true labeles and predicted labels")
            #     print(predicted[:20])
            #     print(targets[:20])

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct/total_sample
        loss_total = loss_total/total_sample

    return loss_total, acc_total


def linear_rampup(current, rampup_length=args.epochs):
    if args.lambda_contra is not None:
        return args.lambda_contra
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / args.epochs, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, logits, targets_x, outputs_u1,outputs_u2,targets_u, mask,zis,zjs):

        Lx = F.cross_entropy(logits, targets_x.long())

        # log_prob_u = F.log_softmax(outputs_u, dim=1)
        # Lu = F.kl_div(log_prob_u, targets_u, reduction='batchmean')
        if mask != None:
            # Lu = - \
            #     torch.mean(torch.sum(F.log_softmax(
            #         outputs_u1, dim=1) * targets_u, dim=1))+ \
            #     (F.cross_entropy(outputs_u2, targets_u,
            #                      reduction='none') * mask).mean()

            Lu = (F.cross_entropy(outputs_u1, targets_u,
                                 reduction='none') * mask).mean() + \
            (F.cross_entropy(outputs_u2, targets_u,
                             reduction='none') * mask).mean()
        else:

            # Lu = + \
            #     (F.cross_entropy(outputs_u1, targets_u,
            #                      reduction='none') * mask).mean()
            Lu = F.cross_entropy(outputs_u1, targets_u) + \
            F.cross_entropy(outputs_u2, targets_u)
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)


        Lc = nt_xent_criterion(zis, zjs)
        return Lx, Lu, Lc


if __name__ == '__main__':
    main()














