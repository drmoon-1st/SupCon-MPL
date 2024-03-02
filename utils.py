from __future__ import print_function

import os
import glob
import math
import pickle
import random

import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.models import efficientnet_b5, efficientnet_b4, resnet50, resnet101

from sklearn.model_selection import train_test_split

from preprocess.make_datasets import make_metadf, is_fake

# Contrastive Loss
def ContrastiveLoss(output1, output2, label, margin=2.0):
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
    loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                    (1-label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))

    return loss_contrastive

# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Intialize before training(multi gpu)
def initialize():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()
    world_size = dist.get_world_size()
    return rank, device, world_size

# Getting datset -> to be changed
def get_dataset(dataset, labeled_data_path=None, unlabeled_data_path=None, output_path=None, is_train=True, is_meta=False, args=None, un=False):

    meta_face = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'Original']

    if is_train:
        origin_data_path = 'DFDC_video_path' # original DFDC dataset path(videos)
        cropped_data_path_dfdc = 'DFDC_face_imgs_path' # cropped face dataset path(imgs) of DFDC
        cropped_data_path_face = 'FaceForensics_face_imgs_path_train' # cropped face dataset path(imgs) of FaceForensics
        cropped_data_path_stylegan = 'StyleGAN_face_imgs_path' # stylegan face image datset path
        cropped_data_path_celebv2 = 'CelebDF_face_imgs_path' # cropped face dataset path(imgs) of CelebDF
        cropped_data_path_youtube_real = 'CelebDF_Youtube_face_imgs_path' # cropped face dataset of youtube videos in CelebDF
        cropped_data_path_celeba = 'CelebA_face_imgs_path' # cropped face dataset path(imgs) of CelebA


        path_face_dic = {}

        # get labels
        if not os.path.exists(f'./datas/img_path_face_{meta_face[-1]}.pkl'):

            paths = glob.glob(f'{cropped_data_path_face}/*')
            for path in paths:
                path_img = glob.glob(f'{path}/**/*')
                path_face_dic[path.split('/')[-1]] = path_img

        if not os.path.exists(f'./datas/metadata.csv'):
            meta = make_metadf(origin_data_path)
            meta.to_csv(f'./datas/metadata.csv', index=False)
        else:
            meta = pd.read_csv(f'./datas/metadata.csv')

        # get deepfake image path
        if not os.path.exists(f'./datas/img_path_dfdc.pkl'):
            out_dfdc = glob.glob(f"{cropped_data_path_dfdc}/**/*.*")
            out_face = glob.glob(f"{cropped_data_path_face}/**/**/*.*")
            out_celeb = glob.glob(f"{cropped_data_path_celebv2}/**/**/*.*")
            out_stylegan = glob.glob(f"{cropped_data_path_stylegan}/*.*")
            out_youtube = glob.glob(f"{cropped_data_path_youtube_real}/**/*.*")
            out_celeba = glob.glob(f'{cropped_data_path_celeba}/*')

            for i in meta_face:
                out_face_tmp = path_face_dic[i]
                with open(f"./datas/img_path_face_{i}.pkl","wb") as f:
                    pickle.dump(out_face_tmp, f)

            with open(f"./datas/img_path_dfdc.pkl","wb") as f:
                pickle.dump(out_dfdc, f)

            with open(f"./datas/img_path_face.pkl","wb") as f:
                pickle.dump(out_face, f)

            with open(f"./datas/img_path_celeb.pkl","wb") as f:
                pickle.dump(out_celeb, f)

            with open(f"./datas/img_path_stylegan.pkl","wb") as f:
                pickle.dump(out_stylegan, f)

            with open(f"./datas/img_path_youtube.pkl","wb") as f:
                pickle.dump(out_youtube, f)
            
            with open(f"./datas/img_path_celeba.pkl","wb") as f:
                pickle.dump(out_celeba, f)
        else:
            for i in meta_face:
                with open(f"./datas/img_path_face_{i}.pkl","rb") as f:
                    path_face_dic[i] = pickle.load(f)

            with open(f"./datas/img_path_dfdc.pkl","rb") as f:
                out_dfdc = pickle.load(f)

            with open(f"./datas/img_path_face.pkl","rb") as f:
                out_face = pickle.load(f)

            with open(f"./datas/img_path_celeb.pkl","rb") as f:
                out_celeb = pickle.load(f)

            with open(f"./datas/img_path_stylegan.pkl","rb") as f:
                out_stylegan = pickle.load(f)

            with open(f"./datas/img_path_youtube.pkl","rb") as f:
                out_youtube = pickle.load(f)
            
            with open(f"./datas/img_path_celeba.pkl","rb") as f:
                out_celeba = pickle.load(f)

        # get labels
        if not os.path.exists(f'./datas/labels_dfdc.pkl'):
            labels_face_dic = {}
            labels_dfdc = []
            labels_face = []
            labels_celeb = []
            labels_stylegan = []
            labels_youtube = []
            labels_celeba = []
            errors_face_dic = {}
            errors_dfdc = []
            errors_face = []
            errors_celeb = []
            errors_stylegan = []
            errors_youtube = []
            errors_celeba = []

            for i in meta_face:
                labels_face_dic[i] = []
                errors_face_dic[i] = []

            for i in meta_face:
                if i == meta_face[-1]:
                    label = 0
                else:
                    label = 1
                    
                for idx, path in enumerate(path_face_dic[i]):
                    try:
                        labels_face_dic[i].append(label)
                    except:
                        errors_face_dic[i].append(idx)
                        print('error')
                print(errors_face_dic[i])
                print(len(errors_face_dic[i]), f'error occured in {i} data.')

                for j in reversed(errors_face_dic[i]):
                    path_face_dic[i].pop(j)

            for idx, path in enumerate(out_dfdc):
                try:
                    labels_dfdc.append(is_fake(meta, path.split('/')[-2]))
                except:
                    errors_dfdc.append(idx)
                    print('error')

            print(errors_dfdc)
            print(len(errors_dfdc), 'error occured in dfdc data.')

            for i in reversed(errors_dfdc):
                out_dfdc.pop(i)

            for idx, path in enumerate(out_face):
                try:
                    labels_face.append(is_fake(meta_face, path.split('/')[-3], 'FaceForensics'))
                except:
                    errors_face.append(idx)

            print(errors_face)
            print(len(errors_face), 'error occured in faceforensics data.')

            for i in reversed(errors_face):
                out_face.pop(i)

            for idx, path in enumerate(out_celeb):
                try:
                    labels_celeb.append(is_fake('fake', path.split('/')[-3], 'celebv2'))
                except:
                    errors_celeb.append(idx)

            print(errors_celeb)
            print(len(errors_celeb), 'error occured in celeb data.')

            for i in reversed(errors_celeb):
                out_face.pop(i)

            for idx, path in enumerate(out_stylegan):
                try:
                    labels_stylegan.append(1.)
                except:
                    errors_stylegan.append(idx)

            print(errors_stylegan)
            print(len(errors_stylegan), 'error occured in stylegan data.')

            for i in reversed(errors_celeb):
                out_face.pop(i)

            for idx, path in enumerate(out_youtube):
                try:
                    labels_youtube.append(0.)
                except:
                    errors_youtube.append(idx)

            print(errors_youtube)
            print(len(errors_youtube), 'error occured in youtube data.')

            for i in reversed(errors_youtube):
                out_youtube.pop(i)

            for idx, path in enumerate(out_celeba):
                try:
                    labels_celeba.append(0.)
                except:
                    errors_celeba.append(idx)

            print(errors_celeba)
            print(len(errors_celeba), 'error occured in celeba data.')

            for i in reversed(errors_youtube):
                out_celeba.pop(i)

            for i in meta_face:
                with open(f"./datas/img_path_face_{i}.pkl","wb") as f:
                    pickle.dump(path_face_dic[i], f)

                with open(f"./datas/labels_face_{i}.pkl","wb") as f:
                    pickle.dump(labels_face_dic[i], f)

            with open(f"./datas/img_path_dfdc.pkl","wb") as f:
                pickle.dump(out_dfdc, f)

            with open(f"./datas/img_path_face.pkl","wb") as f:
                pickle.dump(out_face, f)
            
            with open(f"./datas/img_path_celeb.pkl","wb") as f:
                pickle.dump(out_celeb, f)

            with open(f"./datas/img_path_stylegan.pkl","wb") as f:
                pickle.dump(out_stylegan, f)

            with open(f"./datas/img_path_youtube.pkl","wb") as f:
                pickle.dump(out_youtube, f)

            with open(f"./datas/img_path_celeba.pkl","wb") as f:
                pickle.dump(out_celeba, f)

            with open(f"./datas/labels_dfdc.pkl","wb") as f:
                pickle.dump(labels_dfdc, f)

            with open(f"./datas/labels_face.pkl","wb") as f:
                pickle.dump(labels_face, f)

            with open(f"./datas/labels_celeb.pkl","wb") as f:
                pickle.dump(labels_celeb, f)

            with open(f"./datas/labels_stylegan.pkl","wb") as f:
                pickle.dump(labels_stylegan, f)

            with open(f"./datas/labels_youtube.pkl","wb") as f:
                pickle.dump(labels_youtube, f)

            with open(f"./datas/labels_celeba.pkl","wb") as f:
                pickle.dump(labels_celeba, f)

            del errors_dfdc, errors_face, errors_celeb, errors_stylegan, errors_youtube, errors_celeba, errors_face_dic
        else:
            labels_face_dic = {}

            for i in meta_face:
                with open(f"./datas/labels_face_{i}.pkl","rb") as f:
                    labels_face_dic[i] = pickle.load(f)


            with open(f"./datas/labels_dfdc.pkl","rb") as f:
                labels_dfdc = pickle.load(f)
            
            with open(f"./datas/labels_face.pkl","rb") as f:
                labels_face = pickle.load(f)

            with open(f"./datas/labels_celeb.pkl","rb") as f:
                labels_celeb = pickle.load(f)

            with open(f"./datas/labels_stylegan.pkl","rb") as f:
                labels_stylegan = pickle.load(f)

            with open(f"./datas/labels_youtube.pkl","rb") as f:
                labels_youtube = pickle.load(f)

            with open(f"./datas/labels_celeba.pkl","rb") as f:
                labels_celeba = pickle.load(f)

        if dataset == 'All':
            return path_face_dic[meta_face[0]]+path_face_dic[meta_face[1]]+path_face_dic[meta_face[2]], path_face_dic[meta_face[-1]],\
                    labels_face_dic[meta_face[0]]+labels_face_dic[meta_face[1]]+labels_face_dic[meta_face[2]], labels_face_dic[meta_face[-1]]

            print("All data used.")
        elif dataset == 'Sin':
            
            import copy
            args_tmp = copy.deepcopy(args)
            args_tmp.epochs = args.epochs//2
            
            ff_dataset_composition = path_face_dic[meta_face[0]]+\
                                path_face_dic[meta_face[1]]+\
                                path_face_dic[meta_face[2]]+\
                                path_face_dic[meta_face[-1]]

            ff_label_composition = labels_face_dic[meta_face[0]]+\
                                    labels_face_dic[meta_face[1]]+\
                                    labels_face_dic[meta_face[2]]+\
                                    labels_face_dic[meta_face[-1]]
            
            try:
                ff_img, ff_label, ri_f, rl_f = get_imgs(ff_dataset_composition, 
                                                    ff_label_composition, args_tmp, not_np=True, 
                                                    keep_remain=True if un else False)
            except:
                ff_img, ff_label = get_imgs(ff_dataset_composition, 
                                            ff_label_composition, args_tmp, not_np=True, 
                                            keep_remain=True if un else False)

            args_tmp.epochs = args_tmp.epochs//2

            try:
                dfdc_img, dfdc_label, ri_d, rl_d = get_imgs(out_dfdc, labels_dfdc, args_tmp, not_np=True, 
                keep_remain=True if un else False)
            except:
                dfdc_img, dfdc_label= get_imgs(out_dfdc, labels_dfdc, args_tmp, not_np=True, 
                keep_remain=True if un else False)
            
            

            celebdf_img, celebdf_label = get_imgs(out_celeb[:len(out_celeb)//2], labels_celeb[:len(out_celeb)//2], args_tmp, not_np=True)

            if un:
                imgu, labelu = ri_d + ri_f, rl_d + rl_f

                # args_tmp.epochs = args.epochs * 2
                # imgu, labelu = get_imgs(imgu, labelu, args_tmp)

            imgs, labels = dfdc_img + ff_img  + celebdf_img, dfdc_label + ff_label  + celebdf_label
            
            imgs = np.asarray(imgs)
            labels = np.asarray(labels)

            if un:
                return imgs, labels, imgu, labelu
            else:
                return imgs, labels
        elif dataset == 'DFDC':
            imgs = out_dfdc
            labels = labels_dfdc
            print("DFDC data used.")
        elif dataset == 'FaceForensics':
            imgs = out_face
            labels = labels_face
            print("FaceForensics data used.")
        elif dataset == 'Celebdf':
            imgs = out_celeb
            labels = labels_celeb
            print("Celeb v2 data used.")
        elif dataset == 'StyleGAN':
            imgs = out_stylegan
            labels = labels_stylegan
            print("StyleGAN data used.")
        elif dataset == 'Youtube':
            imgs = out_youtube
            labels = labels_youtube
            print("Youtube data used.")
        elif dataset == 'Celeba':
            imgs = out_celeba
            labels = labels_celeba
            print("Celeba data used.")
        elif dataset == meta_face[0]:
            imgs = path_face_dic[meta_face[0]] + path_face_dic[meta_face[-1]]
            labels = labels_face_dic[meta_face[0]] + labels_face_dic[meta_face[-1]]
            print(f'{meta_face[0]} data used.')

            if is_meta:
                unlabeled_imgs = path_face_dic[meta_face[1]] + path_face_dic[meta_face[2]] + out_celeba
                unlabeled_labels = labels_face_dic[meta_face[1]] + labels_face_dic[meta_face[2]] + labels_celeba
        elif dataset == meta_face[1]:
            imgs = path_face_dic[meta_face[1]] + path_face_dic[meta_face[-1]]
            labels = labels_face_dic[meta_face[1]] + labels_face_dic[meta_face[-1]]
            print(f'{meta_face[1]} data used.')

            if is_meta:
                unlabeled_imgs = path_face_dic[meta_face[2]] + path_face_dic[meta_face[3]] + out_celeba
                unlabeled_labels = labels_face_dic[meta_face[2]] + labels_face_dic[meta_face[3]] + labels_celeba
        elif dataset == meta_face[2]:
            imgs = path_face_dic[meta_face[2]] + path_face_dic[meta_face[-1]]
            labels = labels_face_dic[meta_face[2]] + labels_face_dic[meta_face[-1]]
            print(f'{meta_face[2]} data used.')

            if is_meta:
                unlabeled_imgs = path_face_dic[meta_face[0]] + path_face_dic[meta_face[3]] + out_celeba
                unlabeled_labels = labels_face_dic[meta_face[0]] + labels_face_dic[meta_face[3]] + labels_celeba
        elif dataset == meta_face[3]:
            imgs = path_face_dic[meta_face[3]] + path_face_dic[meta_face[-1]]
            labels = labels_face_dic[meta_face[3]] + labels_face_dic[meta_face[-1]]
            print(f'{meta_face[3]} data used.')

            ###
            return path_face_dic[meta_face[3]], path_face_dic[meta_face[-1]],\
                    labels_face_dic[meta_face[3]], labels_face_dic[meta_face[-1]]
            ###

            if is_meta:
                unlabeled_imgs = path_face_dic[meta_face[0]] + path_face_dic[meta_face[1]] + out_celeba
                unlabeled_labels = labels_face_dic[meta_face[0]] + labels_face_dic[meta_face[1]] + labels_celeba
    else:
        origin_data_path = 'DFDC10_video_path' # original dataset path(videos) of DFDC part 10
        cropped_data_path_dfdc = 'DFDC10_face_imgs_path' # cropped face dataset path(imgs) DFDC part 10
        cropped_data_path_face = 'FaceForensics_face_imgs_path_test' # cropped face dataset path(imgs) -> FaceForensics

        path_face_dic = {}

        if not os.path.exists(f'./datas/img_path_face_test{meta_face[-1]}.pkl'):

            paths = glob.glob(f'{cropped_data_path_face}/*')
            for path in paths:
                path_img = glob.glob(f'{path}/**/*')
                path_face_dic[path.split('/')[-1]] = path_img

        # get labels
        if not os.path.exists(f'./datas/metadata_test.csv'):
            meta = make_metadf(origin_data_path)
            meta.to_csv(f'./datas/metadata_test.csv', index=False)
        else:
            meta = pd.read_csv(f'./datas/metadata_test.csv')

        # get deepfake image path
        if not os.path.exists(f'./datas/img_path_dfdc_test.pkl'):
            out_dfdc = glob.glob(f"{cropped_data_path_dfdc}/**/*.*")
            out_face = glob.glob(f"{cropped_data_path_face}/**/**/*.*")

            for i in meta_face:
                out_face_tmp = path_face_dic[i]
                with open(f"./datas/img_path_face_test{i}.pkl","wb") as f:
                    pickle.dump(out_face_tmp, f)

            with open(f"./datas/img_path_dfdc_test.pkl","wb") as f:
                pickle.dump(out_dfdc, f)

            with open(f"./datas/img_path_face_test.pkl","wb") as f:
                pickle.dump(out_face, f)
        else:
            for i in meta_face:
                with open(f"./datas/img_path_face_test{i}.pkl","rb") as f:
                    path_face_dic[i] = pickle.load(f)

            with open(f"./datas/img_path_dfdc_test.pkl","rb") as f:
                out_dfdc = pickle.load(f)

            with open(f"./datas/img_path_face_test.pkl","rb") as f:
                out_face = pickle.load(f)


        # get labels
        if not os.path.exists(f'./datas/labels_dfdc_test.pkl'):
            labels_face_dic = {}
            labels_dfdc = []
            labels_face = []
            errors_face_dic = {}
            errors_dfdc = []
            errors_face = []

            for i in meta_face:
                labels_face_dic[i] = []
                errors_face_dic[i] = []

            for i in meta_face:
                if i == meta_face[-1]:
                    label = 0
                else:
                    label = 1
                    
                for idx, path in enumerate(path_face_dic[i]):
                    try:
                        labels_face_dic[i].append(label)
                    except:
                        errors_face_dic[i].append(idx)
                        print('error')
                print(errors_face_dic[i])
                print(len(errors_face_dic[i]), f'error occured in {i} data.')

                for j in reversed(errors_face_dic[i]):
                    path_face_dic[i].pop(j)

            for idx, path in enumerate(out_dfdc):
                try:
                    labels_dfdc.append(is_fake(meta, path.split('/')[-2]))
                except:
                    errors_dfdc.append(idx)

            print(errors_dfdc)
            print(len(errors_dfdc), 'error occured in dfdc data.')

            for i in reversed(errors_dfdc):
                out_dfdc.pop(i)

            for idx, path in enumerate(out_face):
                try:
                    labels_face.append(is_fake(meta_face, path.split('/')[-3], 'FaceForensics'))
                except:
                    errors_face.append(idx)

            print(errors_face)
            print(len(errors_face), 'error occured in faceforensics data.')

            for i in reversed(errors_face):
                out_face.pop(i)

            for i in meta_face:
                with open(f"./datas/img_path_face_test{i}.pkl","wb") as f:
                    pickle.dump(path_face_dic[i], f)

                with open(f"./datas/labels_face_test{i}.pkl","wb") as f:
                    pickle.dump(labels_face_dic[i], f)

            with open(f"./datas/img_path_dfdc_test.pkl","wb") as f:
                pickle.dump(out_dfdc, f)

            with open(f"./datas/img_path_face_test.pkl","wb") as f:
                pickle.dump(out_face, f)

            with open(f"./datas/labels_dfdc_test.pkl","wb") as f:
                pickle.dump(labels_dfdc, f)

            with open(f"./datas/labels_face_test.pkl","wb") as f:
                pickle.dump(labels_face, f)

            del errors_dfdc, errors_face, errors_face_dic
        else:
            labels_face_dic = {}

            for i in meta_face:
                with open(f"./datas/labels_face_test{i}.pkl","rb") as f:
                    labels_face_dic[i] = pickle.load(f)

            with open(f"./datas/labels_dfdc_test.pkl","rb") as f:
                labels_dfdc = pickle.load(f)
            
            with open(f"./datas/labels_face_test.pkl","rb") as f:
                labels_face = pickle.load(f)

        if dataset == 'All':
            imgs = out_dfdc + out_face
            labels = labels_dfdc + labels_face
            print("All test data used.")
        elif dataset == 'Sin':
            with open(f"./datas/img_path_celeb.pkl","rb") as f:
                out_celeb = pickle.load(f)
            with open(f"./datas/labels_celeb.pkl","rb") as f:
                labels_celeb = pickle.load(f)

            celeb_img, celeb_label = get_imgs(out_celeb[len(out_celeb)//2:], labels_celeb[len(out_celeb)//2:], args, not_np=True)
            
            imgs = path_face_dic[meta_face[0]] + \
                    path_face_dic[meta_face[1]] + \
                    path_face_dic[meta_face[2]] + \
                    path_face_dic[meta_face[-1]] + out_dfdc + celeb_img
            labels = labels_face_dic[meta_face[0]] + \
                    labels_face_dic[meta_face[1]] + \
                    labels_face_dic[meta_face[2]] + \
                    labels_face_dic[meta_face[-1]] + labels_dfdc + celeb_label
            
            return imgs, labels
        elif dataset == 'StyleGAN':
            with open(f"./datas/img_path_stylegan.pkl","rb") as f:
                out_stylegan = pickle.load(f)
            with open(f"./datas/labels_stylegan.pkl","rb") as f:
                labels_stylegan = pickle.load(f)
            print('StyleGAN data used!!')
            out_stylegan, labels_stylegan = get_imgs(out_stylegan, labels_stylegan, args, not_np=True)
            path_face_dic, labels_face_dic = get_imgs(path_face_dic[meta_face[-1]], labels_face_dic[meta_face[-1]], args, not_np=True)
            img, label = out_stylegan[::2]+path_face_dic[::2], labels_stylegan[::2]+labels_face_dic[::2]
            return img, label
        elif dataset == 'Celebdf':
            with open(f"./datas/img_path_celeb.pkl","rb") as f:
                out_celebdf = pickle.load(f)
            with open(f"./datas/labels_celeb.pkl","rb") as f:
                labels_celebdf = pickle.load(f)
            with open(f"./datas/img_path_celeba.pkl","rb") as f:
                out_celeba = pickle.load(f)
            with open(f"./datas/labels_celeba.pkl","rb") as f:
                labels_celeba = pickle.load(f)
            print('Celebdf data used!!')
            # return out_celebdf+out_celeba[::10], labels_celebdf+labels_celeba[::10]
            return out_celeba[len(out_celeba):], labels_celeba[len(out_celeba):]
        elif dataset == 'DFDC':
            imgs = out_dfdc
            labels = labels_dfdc
            print("DFDC test data used.")
        elif dataset == 'FaceForensics':
            imgs = out_face
            labels = labels_face
            print("Face test data used.")
        elif dataset == meta_face[0]:
            imgs = path_face_dic[meta_face[0]] + path_face_dic[meta_face[-1]]
            labels = labels_face_dic[meta_face[0]] + labels_face_dic[meta_face[-1]]
            print(f'{meta_face[0]} test data used.')
        elif dataset == meta_face[1]:
            imgs = path_face_dic[meta_face[1]] + path_face_dic[meta_face[-1]]
            labels = labels_face_dic[meta_face[1]] + labels_face_dic[meta_face[-1]]
            print(f'{meta_face[1]} test data used.')
        elif dataset == meta_face[2]:
            imgs = path_face_dic[meta_face[2]] + path_face_dic[meta_face[-1]]
            labels = labels_face_dic[meta_face[2]] + labels_face_dic[meta_face[-1]]
            print(f'{meta_face[2]} test data used.')
        elif dataset == meta_face[3]:
            imgs = path_face_dic[meta_face[3]] + path_face_dic[meta_face[-1]]
            labels = labels_face_dic[meta_face[3]] + labels_face_dic[meta_face[-1]]
            print(f'{meta_face[3]} test data used.')

    print('length of imgs', len(imgs),'length of labels', len(labels))
    print('fake', labels.count(1),'real' , labels.count(0))

    if is_meta:
        print('length unlabeled',len(unlabeled_imgs), len(unlabeled_labels))
        return imgs, labels, unlabeled_imgs, unlabeled_labels

    return imgs, labels

    # return path_face_dic[meta_face[0]], out_celeba,\
    #     labels_face_dic[meta_face[0]], labels_celeba
        
    # return path_face_dic[meta_face[3]], path_face_dic[meta_face[-1]],\
    #     labels_face_dic[meta_face[3]], labels_face_dic[meta_face[-1]]
    

# Get models
def get_model(model_name="resnet50", is_meta=False):
    if model_name == 'efficientnetb5':
        teacher_model = efficientnet_b5(weights="IMAGENET1K_V1")
        in_fc = teacher_model.classifier[1].in_features
        teacher_model.classifier[1] =  nn.Linear(in_fc, 2)

        student_model = efficientnet_b5(weights="IMAGENET1K_V1")
        in_fc = student_model.classifier[1].in_features
        student_model.classifier[1] =  nn.Linear(in_fc, 2)
    elif model_name == 'efficientnetb4':
        teacher_model = efficientnet_b4(weights="IMAGENET1K_V1")
        in_fc = teacher_model.classifier[1].in_features
        teacher_model.classifier[1] =  nn.Linear(in_fc, 2)

        student_model = efficientnet_b4(weights="IMAGENET1K_V1")
        in_fc = student_model.classifier[1].in_features
        student_model.classifier[1] =  nn.Linear(in_fc, 2)
    elif model_name == 'resnet101':
        teacher_model = resnet50(weights="IMAGENET1K_V1")
        in_fc = teacher_model.fc.in_features
        teacher_model.fc =  nn.Linear(in_fc, 2)

        student_model = resnet50(weights="IMAGENET1K_V1")
        in_fc = student_model.fc.in_features
        student_model.fc =  nn.Linear(in_fc, 2)
    elif model_name == 'resnet50':
        teacher_model = resnet50(weights="IMAGENET1K_V1")
        in_fc = teacher_model.fc.in_features
        teacher_model.fc =  nn.Linear(in_fc, 2)

        student_model = resnet50(weights="IMAGENET1K_V1")
        in_fc = student_model.fc.in_features
        student_model.fc =  nn.Linear(in_fc, 2)

    if is_meta:
        return teacher_model, student_model
    else:
        return student_model

def get_imgs(imgs, labels, args, not_np=False, keep_remain=False):
    try:
        imgs, _, labels, _ = train_test_split(imgs, labels, 
                                                    test_size=(1-args.epochs*(args.batch_size)/len(imgs)), 
                                                    random_state=args.seed)
    except:
        imgs, ri, labels, rl = train_test_split(imgs, labels, 
                                                  test_size=(1-args.epochs*(args.batch_size_l + args.batch_size_ul)/len(imgs)), 
                                                  random_state=args.seed)
    
    if not_np:
        if keep_remain:
            return imgs, labels, ri, rl
        else:
            return imgs, labels
    else:
        return np.asarray(imgs[::2]), np.asarray(labels[::2])

class Classifier(nn.Module):
  def __init__(self, model):
    super(Classifier, self).__init__()
    self.CNN = model
    self.clf = nn.Linear(128, 2)

  def forward(self, x):
    x = self.CNN(x)
    x = self.clf(x)
    return x

################################################################################################
################################################################################################
################################################################################################

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, model, head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        dim_in = model.fc.in_features
        model.fc = nn.Identity()
        self.encoder = model
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, model, num_classes=2):
        super(SupCEResNet, self).__init__()
        dim_in = model.fc.in_features
        model.fc = nn.Identity()
        self.encoder = model
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, feat_dim=2048, num_classes=2):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)