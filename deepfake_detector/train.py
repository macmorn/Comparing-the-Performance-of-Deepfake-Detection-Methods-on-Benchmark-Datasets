import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import ShuffleSplit
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset

import datasets
import timm
import metrics
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from facedetector.retinaface import df_retinaface
from pretrained_mods import xception
from tqdm import tqdm


def train(dataset, data, method, normalization, augmentations, img_size,
          folds=1, epochs=1, batch_size=32, lr=0.001, fulltrain=False, load_model_path=None
          ):
    """
    Train a DNN for a number of epochs.

    # parts from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # adapted by: Christopher Otto
    """
    training_time = time.time()
    # use gpu for calculations if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    average_auc = []
    average_loss = []
    average_acc = []
    average_ap = []

    # k-fold cross-val if folds > 1
    for fold in range(folds):
        if folds > 1:
            # doing k-fold cross-validation
            print(f"Fold: {fold}")

        best_acc = 0.0
        best_loss = 100
        current_acc = 0.0
        current_loss = 100.0
        best_auc = 0.0
        best_ap = 0.0
        # get train and val indices
        if fulltrain == False:
            train_idx, val_idx = shuffeled_cross_val(fold, data)

        # prepare training and validation data
        if fulltrain == True:
            if dataset == 'uadfv':
                train_dataset = datasets.UADFVDataset(
                    data, img_size, normalization=normalization, augmentations=augmentations)
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True)
        else:
            if dataset == 'uadfv':
                train_dataset = datasets.UADFVDataset(
                    data.iloc[train_idx], img_size, normalization=normalization, augmentations=augmentations)
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True)
                val_dataset = datasets.UADFVDataset(
                    data.iloc[val_idx], img_size, normalization=normalization, augmentations=None)
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False)

        if load_model_path is None:
            # train model from scratch/pretrained imagenet
            if method == 'xception':
                # load the xception model
                model = xception.imagenet_pretrained_xception()
            elif method == 'efficientnetb7':
                model = timm.create_model(
                    'tf_efficientnet_b7_ns', pretrained=True)
                # binary classification output
                model.classifier = nn.Linear(2560, 1)
        else:
            # load model
            model = torch.load(load_model_path)

        if fold == 0:
            best_model = copy.deepcopy(model.state_dict())

        # put model on gpu
        model = model.cuda()
        # binary cross-entropy loss
        loss_func = nn.BCEWithLogitsLoss()
        lr = lr
        # adam optimizer
        optimizer = Adam(model.parameters(), lr=lr)
        # cosine annealing scheduler
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, eta_min=0.000001, last_epoch=-1)

        for e in range(epochs):
            print('#' * 20)
            print(f"Epoch {e}/{epochs}")
            print('#' * 20)
            # training and validation loop
            for phase in ["train", "val"]:
                if phase == "train":
                    # put layers in training mode
                    model.train()
                else:
                    # turn batchnorm and dropout layers to eval mode
                    model.eval()

                running_loss = 0.0
                running_corrects = 0.0
                running_auc_labels = []
                running_ap_labels = []
                running_auc_preds = []
                running_ap_preds = []
                if phase == "train":
                    # then load training data
                    for imgs, labels in tqdm(train_loader):
                        # put calculations on gpu
                        imgs = imgs.to(device)
                        labels = labels.to(device)
                        # set accumulated gradients to zero
                        optimizer.zero_grad()

                        # forward pass of inputs and turn on gradient computation during train
                        with torch.set_grad_enabled(phase == "train"):
                            predictions = model(imgs)
                            sig = torch.sigmoid(predictions)
                            # predictions for acc calculation; classification thresh 0.5
                            thresh_preds = torch.round(
                                torch.sigmoid(predictions))
                            loss = loss_func(
                                predictions.squeeze(), labels.type_as(predictions))

                            if phase == "train":
                                # backpropagate gradients
                                loss.backward()
                                # update parameters
                                optimizer.step()

                        running_loss += loss.item() * imgs.size(0)
                        # calc accuracy
                        running_corrects += torch.sum(thresh_preds ==
                                                      labels.unsqueeze(1))
                        running_auc_labels.extend(
                            labels.detach().cpu().numpy())
                        running_auc_preds.extend(
                            sig.detach().cpu().numpy().flatten().tolist())
                        running_ap_labels.extend(labels.detach().cpu().numpy())
                        running_ap_preds.extend(
                            sig.detach().cpu().numpy().flatten().tolist())
                    if phase == 'train':
                        # update lr
                        scheduler.step()
                    epoch_loss = running_loss / len(train_dataset)
                    epoch_acc = running_corrects / len(train_dataset)
                    epoch_auc = roc_auc_score(
                        running_auc_labels, running_auc_preds)
                    epoch_ap = average_precision_score(
                        running_ap_labels, running_ap_preds)
                    print(
                        f"{phase} Loss: {epoch_loss}, Acc: {epoch_acc}, AUC: {epoch_auc}, AP: {epoch_ap}")
                    if fulltrain == True and e+1 == epochs:
                        # save model if epochs reached
                        torch.save(
                            model.state_dict(), os.getcwd() + f'/{method}_best_fulltrain_{dataset}.pth')

                else:
                    if fulltrain == True:
                        continue
                    # get valitation data
                    for imgs, labels in tqdm(val_loader):
                        # put calculations on gpu
                        imgs = imgs.to(device)
                        labels = labels.to(device)
                        # set accumulated gradients to zero
                        optimizer.zero_grad()

                        # forward pass of inputs and turn on gradient computation during train
                        with torch.set_grad_enabled(phase == "train"):
                            predictions = model(imgs)
                            sig = torch.sigmoid(predictions)
                            # predictions for acc calculation; classification thresh 0.5
                            thresh_preds = torch.round(
                                torch.sigmoid(predictions))
                            loss = loss_func(
                                predictions.squeeze(), labels.type_as(predictions))

                        running_loss += loss.item() * imgs.size(0)
                        # calc accuracy
                        running_corrects += torch.sum(thresh_preds ==
                                                      labels.unsqueeze(1))

                        running_auc_labels.extend(
                            labels.detach().cpu().numpy())
                        running_auc_preds.extend(
                            sig.detach().cpu().numpy().flatten().tolist())
                        running_ap_labels.extend(labels.detach().cpu().numpy())
                        running_ap_preds.extend(
                            sig.detach().cpu().numpy().flatten().tolist())
                    metrics.prec_rec(
                        running_auc_labels, running_auc_preds, method, alpha=100, plot=False)
                    epoch_loss = running_loss / len(val_dataset)
                    epoch_acc = running_corrects / len(val_dataset)
                    epoch_auc = roc_auc_score(
                        running_auc_labels, running_auc_preds)
                    epoch_ap = average_precision_score(
                        running_ap_labels, running_ap_preds)
                    print(
                        f"{phase} Loss: {epoch_loss}, Acc: {epoch_acc}, AUC: {epoch_auc}, AP: {epoch_ap}")

                    # save model if acc better than best acc
                    if epoch_acc > best_acc:
                        print("Found a better model.")
                        one_rec, five_rec, nine_rec = metrics.prec_rec(
                            running_auc_labels, running_auc_preds, method, alpha=100, plot=False)
                        best_acc = epoch_acc
                        best_auc = epoch_auc
                        best_loss = epoch_loss
                        best_ap = epoch_ap
                        if folds > 1:
                            torch.save(
                                model.state_dict(), os.getcwd() + f'/{method}_best_acc_model_fold{fold}.pth')
                        else:
                            torch.save(
                                model.state_dict(), os.getcwd() + f'/{method}_best_acc_model.pth')

        average_auc.append(best_auc)
        average_ap.append(best_ap)
        average_acc.append(best_acc)
        average_loss.append(best_loss)
    average_auc = np.mean(average_auc)
    average_ap = np.mean(average_ap)
    average_acc = np.mean(np.asarray(
        [entry.cpu().numpy() for entry in average_acc]))
    average_loss = np.mean(np.asarray([entry for entry in average_loss]))

    if fulltrain == True:
        # only saved model is returned
        return model, 0, 0, 0, 0

    if folds > 1:
        print(f"Average AUC: {average_auc}")
        print(f"Average AP: {average_ap}")
        print(f"Average Acc: {average_acc}")
        print(f"Average Loss: {average_loss}")
        print(
            f"Duration: {(time.time() - training_time) // 60} min and {(time.time() - training_time) % 60} sec.")
    else:
        print()
        print("Best models metrics:")
        print(f"Acc: {average_acc}")
        print(f"AUC: {average_auc}")
        print(f"AP: {average_ap}")
        print(f"Loss: {average_loss}")
        print()
        print("Cost (best possible cost is 0.0):")
        print(f"{one_rec} cost for 0.1 recall.")
        print(f"{five_rec} cost for 0.5 recall.")
        print(f"{nine_rec} cost for 0.9 recall.")
        print(
            f"Duration: {(time.time() - training_time) // 60} min and {(time.time() - training_time) % 60} sec.")

    # load best model params
    # model.load_state_dict(best_model)
    return model, average_auc, average_ap, average_acc, average_loss


def shuffeled_cross_val(fold, df):
    """
    Return train and val indices for fold.
    """

    X = df['video'].values
    y = df['label'].values
    skf = ShuffleSplit(n_splits=5, test_size=0.25, random_state=24)
    train = []
    val = []
    for train_index, val_index in skf.split(X, y):
        train.append(train_index)
        val.append(val_index)

    # return indices for fold
    return list(train[fold]), list(val[fold])

