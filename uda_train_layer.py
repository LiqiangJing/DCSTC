import argparse
import os
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import sys
sys.path.append('./myfile')
from transformers import AdamW
from read_data import *
from tqdm import tqdm
from bert_layer import ClassificationBert
def set_seeds(seed):
    "set random seeds"
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic=True
set_seeds(0)
parser = argparse.ArgumentParser(description='PyTorch Base Models')

parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch-size-u', default=8, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')


parser.add_argument('--n-labeled', type=int, default=20,
                    help='Number of labeled data')
parser.add_argument('--un-labeled', default=5000, type=int,
                    help='number of unlabeled data')
parser.add_argument('--val-iteration', type=int, default=200,
                    help='Number of labeled data')

parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='augment labeled training data')

parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--data-path', type=str, default='../yahoo_answers_csv/',
                    help='path to data folders')

parser.add_argument('--lambda-u', default=1, type=float,
                    help='weight for consistency loss term of unlabeled data')
parser.add_argument('--T', default=0.5, type=float,
                    help='temperature for sharpen function')

args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

best_acc = 0
total_steps = 0
flag = 0

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
        dataset=val_set, batch_size=64, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=64, shuffle=False)

    # Define the model, set the optimizer
    model = ClassificationBert(n_labels).cuda()
    # model = nn.DataParallel(model)
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

    # Start training
    for epoch in range(args.epochs):

        train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
              scheduler, train_criterion, epoch, n_labels, args.train_aug)

        val_loss, val_acc = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')

        print("epoch {}, val acc {}, val_loss {}".format(
            epoch, val_acc, val_loss))

        if val_acc >= best_acc:
            best_acc = val_acc
            test_loss, test_acc = validate(
                test_loader, model, criterion, epoch, mode='Test Stats ')
            test_accs.append(test_acc)
            print("epoch {}, test acc {},test loss {}".format(
                epoch, test_acc, test_loss))

        print('Epoch: ', epoch)

        print('Best acc:')
        print(best_acc)

        print('Test acc:')
        print(test_accs)

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

    with tqdm(total=args.val_iteration) as pbar:

        for batch_idx in range(args.val_iteration):
            total_steps += 1

            try:
                inputs_x, targets_x, inputs_x_length = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x, inputs_x_length = labeled_train_iter.next()

            try:
                (inputs_u, _,  inputs_ori), (length_u,
                                                     _,  length_ori) = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (inputs_u, _, inputs_ori), (length_u,
                                                    _, length_ori) = unlabeled_train_iter.next()



            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_ori = inputs_ori.cuda()
            logits, _ = model(inputs_x)
            outputs_u, hidden_u = model(inputs_u)
            with torch.no_grad():
                outputs_ori, hidden_ori = model(inputs_ori)
                p = torch.softmax(outputs_ori, dim=1)
                pt = p ** (1 / args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            Lx, Lu, Loss_layer= criterion(logits, targets_u, outputs_u, targets_x, hidden_u, hidden_ori)

            loss = Lx + args.lambda_u * (Lu + Loss_layer)

            #max_grad_norm = 1.0
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            pbar.set_description(
                "epoch {}, step {}, loss {:.4f}, Lx {:.4f}, Lu {:.4f}, Llayer {:.4f}".format(
                    epoch, batch_idx, loss.item(), Lx.item(), Lu.item(), Loss_layer.item())
            )
            pbar.update(1)
            # if batch_idx % 1000 == 0:
            #     print("epoch {}, step {}, loss {}, Lx {}, Lu {}".format(
            #         epoch, batch_idx, loss.item(), Lx.item(), Lu.item()))


def validate(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0

        for batch_idx, (inputs, targets, length) in enumerate(tqdm(valloader)):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs, _ = model(inputs)
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
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, logits, targets_u, outputs_u, targets_x, hidden_u, hidden_ori):

        Lx = F.cross_entropy(logits, targets_x.long())



        log_prob_u = F.log_softmax(outputs_u, dim=1)
        Lu = F.kl_div(log_prob_u, targets_u, reduction='batchmean')
        hidden_states = (hidden_ori, hidden_u)
        Loss_layer = None

        for idx,hidden_layer in enumerate(hidden_states):
            if idx == 0:
                Loss_layer = F.mse_loss(hidden_layer[1], hidden_layer[0])
            else:
                Loss_layer += F.mse_loss(hidden_layer[1], hidden_layer[0])

        return Lx, Lu, Loss_layer


if __name__ == '__main__':
    main()














