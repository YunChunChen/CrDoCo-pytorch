import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random

import model
import discriminator
import dataset

SOURCE_DATA_DIR  = '/path/to/source/dataset'
SOURCE_DATA_LIST = '/path/to/source/datalist'

TARGET_DATA_DIR  = '/path/to/target/dataset'
TARGET_DATA_LIST = '/path/to/target/datalist'

SOURCE_IMAGE_SIZE = '256,256'
TARGET_IMAGE_SIZE = '256,256'


def seg_loss(pred, label, gpu):
    label = Variable(label.long()).cuda(gpu)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda(gpu)
    return criterion(pred, label)


def consis_loss(t_pred, t2s_pred,  gpu):
    loss_t2s_t = t2s_pred * torch.log(F.softmax(t_pred, dim=1)) 
    loss_t_t2s = t_pred * torch.log(F.softmax(t2s_pred, dim=1))
    loss = (loss_t2s_t + loss_t_t2s) / 2.0
    return loss


def loss_adv(pred, gt):
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(pred, gt)
    return loss


def main():

    cudnn.enabled = True
    gpu = args.gpu

    model_S = create_seg_model(num_classes=args.num_classes)
    model_T = create_seg_model(num_classes=args.num_classes)

    model_S.load_state_dict(pretrained_model_path)
    model_T.load_state_dict(pretrained_model_path)

    model_S.train()
    model_T.train()

    model_S.cuda(args.gpu)
    model_T.cuda(args.gpu)

    cudnn.benchmark = True

    model_D1_S = create_dis_model(num_classes=args.num_classes)
    model_D2_S = create_dis_model(num_classes=args.num_classes)

    model_D1_S.train()
    model_D2_S.train()

    model_D1_S.cuda(args.gpu)
    model_D2_S.cuda(args.gpu)


    model_D1_T = create_dis_model(num_classes=args.num_classes)
    model_D2_T = create_dis_model(num_classes=args.num_classes)

    model_D1_T.train()
    model_D2_T.train()

    model_D1_T.cuda(args.gpu)
    model_D2_T.cuda(args.gpu)


    netS_T = create_trans_model() 
    netT_S = create_trans_model() 

    netS_T.load_state_dict(pretrained_S2T_path)
    netT_S.load_state_dict(pretrained_T2S_path)

    netS_T.cuda(args.gpu)
    netT_S.cuda(args.gpu)


    for param in netS_T.parameters():
        param.requires_grad = False
    for param in netT_S.parameters():
        param.requires_grad = False

    trainloader = data.DataLoader(
                      GTA5DataSet(args.data_dir, 
                                  args.data_list, 
                                  max_iters=args.num_steps * args.iter_size * args.batch_size,
                                  crop_size=input_size, 
                                  remap_labels=True,
                                  scale=args.random_scale, 
                                  mirror=args.random_mirror, 
                                  mean=IMG_MEAN),
                      batch_size=args.batch_size, 
                      shuffle=True, 
                      num_workers=args.num_workers, 
                      pin_memory=True
                )

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)

    targetloader_iter = enumerate(targetloader)


    optimizer_S = optim.SGD(model_S.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_T = optim.SGD(model_T.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_S.zero_grad()
    optimizer_T.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    optimizer_D1_T = optim.Adam(model_D1_T.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1_T.zero_grad()
    optimizer_D2_T = optim.Adam(model_D2_T.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2_T.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0
        loss_consist_1_value = 0
        loss_seg_value1_T = 0
        loss_adv_target_value1 = 0
        loss_adv_target_value1_T = 0
        loss_D_value1 = 0
        loss_D_value1_T = 0

        loss_seg_value2 = 0
        loss_consist_2_value = 0
        loss_seg_value2_T = 0
        loss_adv_target_value2 = 0
        loss_adv_target_value2_T = 0
        loss_D_value2 = 0
        loss_D_value2_T = 0
        # print(i_iter)
        optimizer_S.zero_grad()
        optimizer_T.zero_grad()
        adjust_learning_rate(optimizer_S, i_iter)
        adjust_learning_rate(optimizer_T, i_iter)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        adjust_learning_rate_D(optimizer_D2, i_iter)

        optimizer_D1_T.zero_grad()
        optimizer_D2_T.zero_grad()
        adjust_learning_rate_D(optimizer_D1_T, i_iter)
        adjust_learning_rate_D(optimizer_D2_T, i_iter)

        for sub_i in range(args.iter_size):

            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False

            for param in model_D1_T.parameters():
                param.requires_grad = False

            for param in model_D2_T.parameters():
                param.requires_grad = False


            # train with source
            _, batch = trainloader_iter.__next__()
            images_raw, labels, _ = batch
            images = Variable(images_raw).cuda(args.gpu) # Xa
            GTA_to_Real_images = Variable(netG_A(images).detach()).cuda(args.gpu) #Xac

            ### Segmentation modle trained with Xa and Xac (those two have same label) ###

            pred1, pred2 = model_S(images)
            pred1_T, pred2_T = model_T(GTA_to_Real_images)

            pred1 = interp(pred1)
            pred2 = interp(pred2)
            pred1_T = interp(pred1_T)
            pred2_T = interp(pred2_T)

            # Source label loss #
            loss_seg1 = loss_calc(pred1, labels, args.gpu)
            loss_seg2 = loss_calc(pred2, labels, args.gpu)
            loss_seg1_T = loss_calc(pred1_T, labels, args.gpu)
            loss_seg2_T = loss_calc(pred2_T, labels, args.gpu)

            loss_S = loss_seg2 + args.lambda_seg * loss_seg1
            loss_T = loss_seg2_T + args.lambda_seg * loss_seg1_T

            # model_S backward with Xa image and label
            loss = loss_S / args.iter_size
            loss.backward(retain_graph=True)

            # model T backward with Xac image and label
            loss = loss_T / args.iter_size
            loss.backward(retain_graph=True)

            # itemizae loss
            loss_seg_value1 += loss_seg1.item() / args.iter_size
            loss_seg_value2 += loss_seg2.item() / args.iter_size

            loss_seg_value1_T += loss_seg1_T.item() / args.iter_size
            loss_seg_value2_T += loss_seg2_T.item() / args.iter_size


            # train with target
            _, batch = targetloader_iter.__next__()
            images_raw, _, _ = batch
            images = Variable(images_raw).cuda(args.gpu) # Xc
            Real_to_GTA_images = Variable(netG_B(images).detach()).cuda(args.gpu) # Xca

            ### Segmentation modle trained with Xc and Xca (those two don't have a label) ###

            pred_target1, pred_target2 = model_T(images)
            pred_target1_S, pred_target2_S = model_S(Real_to_GTA_images)

            pred_target1 = interp_target(pred_target1)
            pred_target2 = interp_target(pred_target2)
            pred_target1_S = interp_target(pred_target1_S)
            pred_target2_S = interp_target(pred_target2_S)

            # consistency Loss #
            pred_target1_prob = F.softmax(pred_target1, dim=1)
            pred_target2_prob = F.softmax(pred_target2, dim=1)
            pred_target1_S_prob = F.softmax(pred_target1_S, dim=1)
            pred_target2_S_prob = F.softmax(pred_target2_S, dim=1)

            pred_target1_problog = F.log_softmax(pred_target1, dim=1)
            pred_target2_problog = F.log_softmax(pred_target2, dim=1)
            pred_target1_S_problog = F.log_softmax(pred_target1_S, dim=1)
            pred_target2_S_problog = F.log_softmax(pred_target2_S, dim=1)

            loss_consist_1 = KL_criterion(pred_target1_problog, pred_target1_S_prob) + KL_criterion(pred_target1_S_problog, pred_target1_prob)
            loss_consist_2 = KL_criterion(pred_target2_problog, pred_target2_S_prob) + KL_criterion(pred_target2_S_problog, pred_target2_prob)

            loss = (loss_consist_1 + loss_consist_2) / args.iter_size
            loss.backward(retain_graph=True)

            # itemize Loss
            loss_consist_1_value += loss_consist_1.item() / args.iter_size
            loss_consist_2_value += loss_consist_2.item() / args.iter_size

            ### Feature learning for confusing discriminator ###

            # pred_target1(2)_S : model_S(Xca) --> needs to be source like by D_out_S
            D_out1_S = model_D1(F.softmax(pred_target1_S, dim=1)) 
            D_out2_S = model_D2(F.softmax(pred_target2_S, dim=1))

            # pred1(2)_T : model_T(Xac) --> needs to be target like by D_out_T
            D_out1_T = model_D1_T(F.softmax(pred1_T, dim=1)) 
            D_out2_T = model_D2_T(F.softmax(pred2_T, dim=1))

            # Loss for confusing D
            loss_adv_target1 = bce_loss(D_out1_S,
                                       Variable(torch.FloatTensor(D_out1_S.data.size()).fill_(source_label)).cuda(
                                           args.gpu))

            loss_adv_target2 = bce_loss(D_out2_S,
                                        Variable(torch.FloatTensor(D_out2_S.data.size()).fill_(source_label)).cuda(
                                            args.gpu))

            loss_adv_target1_T = bce_loss(D_out1_T,
                                       Variable(torch.FloatTensor(D_out1_T.data.size()).fill_(target_label)).cuda(
                                           args.gpu))

            loss_adv_target2_T = bce_loss(D_out2_T,
                                        Variable(torch.FloatTensor(D_out2_T.data.size()).fill_(target_label)).cuda(
                                            args.gpu))


            loss_S = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
            loss_T = args.lambda_adv_target1 * loss_adv_target1_T + args.lambda_adv_target2 * loss_adv_target2_T

            # Backward for D_out_S
            loss = loss_S / args.iter_size
            loss.backward()

            # Backward for D_out_T
            loss = loss_T / args.iter_size
            loss.backward()

            # Itemize the Loss
            loss_adv_target_value1 += loss_adv_target1.item()
            loss_adv_target_value2 += loss_adv_target2.item()
            loss_adv_target_value1_T += loss_adv_target1_T.item()
            loss_adv_target_value2_T += loss_adv_target2_T.item()


            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            for param in model_D1_T.parameters():
                param.requires_grad = True

            for param in model_D2_T.parameters():
                param.requires_grad = True

            pred1 = pred1.detach() # GTA
            pred2 = pred2.detach()
            pred_target1 = pred_target1.detach() # Real
            pred_target2 = pred_target2.detach()

            D_out1_S = model_D1(F.softmax(pred1))
            D_out2_S = model_D2(F.softmax(pred2))
            D_out1_T = model_D1_T(F.softmax(pred_target1))
            D_out2_T = model_D2_T(F.softmax(pred_target2))

            loss_D1 = bce_loss(D_out1_S,
                              Variable(torch.FloatTensor(D_out1_S.data.size()).fill_(source_label)).cuda(args.gpu))

            loss_D2 = bce_loss(D_out2_S,
                               Variable(torch.FloatTensor(D_out2_S.data.size()).fill_(source_label)).cuda(args.gpu))

            loss_D1_T = bce_loss(D_out1_T,
                              Variable(torch.FloatTensor(D_out1_T.data.size()).fill_(target_label)).cuda(args.gpu))

            loss_D2_T = bce_loss(D_out2_T,
                               Variable(torch.FloatTensor(D_out2_T.data.size()).fill_(target_label)).cuda(args.gpu))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1_T = loss_D1_T / args.iter_size / 2
            loss_D2_T = loss_D2_T / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()
            loss_D1_T.backward()
            loss_D2_T.backward()

            # Itemize the loss
            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()
            loss_D_value1_T += loss_D1_T.item()
            loss_D_value2_T += loss_D2_T.item()

            # train with target
            pred1_T = pred1_T.detach()
            pred2_T = pred2_T.detach()
            pred_target1_S = pred_target1_S.detach()
            pred_target2_S = pred_target2_S.detach()

            D_out1_S = model_D1(F.softmax(pred_target1_S))
            D_out2_S = model_D2(F.softmax(pred_target2_S))
            D_out1_T = model_D1_T(F.softmax(pred1_T))
            D_out2_T = model_D2_T(F.softmax(pred2_T))

            loss_D1 = bce_loss(D_out1_S,
                              Variable(torch.FloatTensor(D_out1_S.data.size()).fill_(target_label)).cuda(args.gpu))

            loss_D2 = bce_loss(D_out2_S,
                               Variable(torch.FloatTensor(D_out2_S.data.size()).fill_(target_label)).cuda(args.gpu))

            loss_D1_T = bce_loss(D_out1_T,
                              Variable(torch.FloatTensor(D_out1_T.data.size()).fill_(source_label)).cuda(args.gpu))

            loss_D2_T = bce_loss(D_out2_T,
                               Variable(torch.FloatTensor(D_out2_T.data.size()).fill_(source_label)).cuda(args.gpu))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1_T = loss_D1_T / args.iter_size / 2
            loss_D2_T = loss_D2_T / args.iter_size / 2

            # loss_D_second = loss_D1 + loss_D2 + loss_D1_T + loss_D2_T

            loss_D1.backward()
            loss_D2.backward()
            loss_D1_T.backward()
            loss_D2_T.backward()

            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()
            loss_D_value1_T += loss_D1_T.item()
            loss_D_value2_T += loss_D2_T.item()

        optimizer_S.step()
        optimizer_T.step()
        optimizer_D1.step()
        optimizer_D2.step()
        optimizer_D1_T.step()
        optimizer_D2_T.step()

        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_seg_value1_T = {4:.3f} loss_seg_value2_T = {5:.3f} loss_adv1 = {5:.3f}, loss_adv2 = {6:.3f} loss_adv_target_value1_T = {7:.3f} loss_adv_target_value1_T = {8:.3f} loss_D1 = {9:.3f} loss_D2 = {10:.3f} loss_D1_T = {11:.3f} loss_D2_T = {12:.3f}'.format(
            i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_seg_value1_T, loss_seg_value2_T, loss_adv_target_value1, loss_adv_target_value2, loss_adv_target_value1_T, loss_adv_target_value2_T, loss_D_value1, loss_D_value2, loss_D_value1_T, loss_D_value2_T))

        Loss_list_Tensorboard = [loss_seg_value1, loss_seg_value2, loss_seg_value1_T, loss_seg_value2_T, loss_adv_target_value1, loss_adv_target_value2, loss_adv_target_value1_T, loss_adv_target_value2_T, loss_D_value1, loss_D_value2, loss_D_value1_T, loss_D_value2_T]
        # write_loss(writer, Loss_list_Tensorboard, i_iter)
        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model_S.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + 'S.pth'))
            torch.save(model_T.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + 'T.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D2.pth'))
            torch.save(model_D1_T.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1_T.pth'))
            torch.save(model_D2_T.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D2_T.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model_S.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + 'S.pth'))
            torch.save(model_T.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + 'T.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D2.pth'))
            torch.save(model_D1_T.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1_T.pth'))
            torch.save(model_D2_T.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D2_T.pth'))

if __name__ == '__main__':
    main()
