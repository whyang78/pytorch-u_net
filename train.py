import argparse
import torch
from torch import nn,optim
from tqdm import trange
import numpy as np
import scipy.io
import os
import random
import math
import matplotlib.pyplot as plt
import cv2

# import sys
# sys.path.append('./')
from unet import U_Net

random.seed(78)

parser=argparse.ArgumentParser(description='u-net seg')
parser.add_argument('--image_size','-s',type=int,default=128,dest='image_size',help='picture input size')
parser.add_argument('--train_rate','-r',type=float,default=0.8,dest='train_rate',help='train dataset rate')
parser.add_argument('--learning_rate','-l',type=float,default=0.01,dest='lr',help='learning rate')
parser.add_argument('--epoch','-e',type=int,default=10,dest='epoch',help='train epoch')
parser.add_argument('--batch_size','-b',type=int,default=5,dest='bs',help='batch size')
parser.add_argument('--encode_type','-t',type=bool,default=True,dest='blinear',help='True:bilinear,False:convTranspose')
parser.add_argument('--use_gpu','-u',type=bool,default=True,dest='use_gpu',help='use gpu or not')
args=parser.parse_args()

def get_data(doc_path,image_size,train_rate):
    subject_path=[]
    for i in os.listdir(doc_path):
        subject_path.append(os.path.join(doc_path,i))
    random.shuffle(subject_path)

    images,masks=[],[]
    for i,sub_path in enumerate(subject_path):
        mat=scipy.io.loadmat(sub_path)
        image=mat['images'] # HWB numpy
        mask=mat['manualFluid1']
        image=np.transpose(image,(2,0,1))/255
        image=np.resize(image,(image.shape[0],image_size,image_size))
        mask=np.transpose(mask,(2,0,1))
        mask=np.where(mask>0,mask,1)
        mask = np.resize(mask, (mask.shape[0], image_size, image_size))
        if i==0:
            images=image
            masks=mask
        else:
            images=np.concatenate((images,image),axis=0)
            masks=np.concatenate((masks,mask),axis=0)
    train_images=images[:int(len(images)*train_rate),...]
    train_images=torch.from_numpy(train_images).float().unsqueeze(dim=1)
    train_masks=masks[:int(len(images)*train_rate),...]
    train_masks=torch.from_numpy(train_masks).long().unsqueeze(dim=1)
    test_images=images[int(len(images)*train_rate):,...]
    test_images = torch.from_numpy(test_images).float().unsqueeze(dim=1)
    test_masks=masks[int(len(images)*train_rate):,...]
    test_masks = torch.from_numpy(test_masks).long().unsqueeze(dim=1)
    return train_images,train_masks,test_images,test_masks

def plot_examples(unet,device, datax, datay, num_examples=3):
    fig, ax = plt.subplots(nrows=num_examples, ncols=2, figsize=(18,2*num_examples))
    m = datax.shape[0]
    datax,datay=datax.numpy(),datay.numpy()
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = unet(torch.from_numpy(datax[image_indx:image_indx+1]).float().to(device)).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx], (1,2,0))[:,:,0])
        ax[row_num][1].imshow(np.transpose(image_arr, (1,2,0))[:,:,0])
    plt.show()

def train(model,device,epoch,batch_size,train_images, train_masks, test_images, test_masks,optimizer,criterion):
    min_loss=10000.00
    for e in trange(epoch,leave=True):
        model.train()
        steps = math.ceil(train_images.shape[0] / batch_size)
        for i in range(steps):
            batchData=train_images[i*batch_size:(i+1)*batch_size].to(device)
            batchMask=train_masks[i*batch_size:(i+1)*batch_size].to(device)
            output=model(batchData) #b,c,h,w
            output=output.permute(0,2,3,1)
            output=output.reshape(-1,2)
            batchMask=batchMask.reshape(-1,1).squeeze()

            loss=criterion(output,batchMask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%5==0:
                print('train epoch:{},i:{},loss:{}'.format(e,i,loss.item()))

        model.eval()
        steps = math.ceil(test_images.shape[0] / batch_size)
        with torch.no_grad():
            test_loss=0.0
            for i in range(steps):
                batchData = test_images[i * batch_size:(i + 1) * batch_size].to(device)
                batchMask = test_masks[i * batch_size:(i + 1) * batch_size].to(device)
                output = model(batchData)  # b,c,h,w
                output = output.permute(0, 2, 3, 1)
                output = output.reshape(-1, 2)
                batchMask = batchMask.reshape(-1, 1).squeeze()

                loss = criterion(output, batchMask)
                test_loss+=loss.item()
            print('***** test epoch:{},loss:{}'.format(e,test_loss))
            if test_loss<min_loss:
                min_loss=test_loss
                torch.save(model.state_dict(),'./best.mdl')

if __name__ == '__main__':
    doc_path='./2015_BOE_Chiu'
    train_images, train_masks, test_images, test_masks=get_data(doc_path,args.image_size,args.train_rate)

    #train
    # device = torch.device('cpu')
    # if args.use_gpu:
    #     device=torch.device('cuda:0')
    # net=U_Net(1,2,args.blinear)
    # net.initialize_weights()
    # net.to(device)
    # optimizer=optim.Adam(net.parameters(),lr=args.lr,weight_decay=1e-3)
    # criterion=nn.CrossEntropyLoss().to(device)
    #
    # train(net,device,args.epoch,args.bs,train_images,train_masks,test_images,test_masks,optimizer,criterion)

    #test
    device = torch.device('cpu')
    if args.use_gpu:
        device = torch.device('cuda:0')
    net = U_Net(1, 2,args.blinear) #与train对应好
    net.to(device)
    if args.use_gpu:
        net.load_state_dict(torch.load('./best.mdl'))
    else:
        net.load_state_dict(torch.load('./best.mdl', map_location='cpu'))
    #取数据集进行预测
    plot_examples(net,device,train_images,train_masks,5)
    plot_examples(net,device,test_images,test_masks,5)

    #取非数据集图片进行预测
    image=cv2.imread('./1.png')
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image=cv2.resize(image,(args.image_size,args.image_size))/255
    plt.imshow(image)
    plt.show()
    image_tensor=torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    image_arr = net(image_tensor.to(device)).squeeze(
        0).detach().cpu().numpy()
    plt.imshow(image_arr[0,:,:])
    plt.show()








