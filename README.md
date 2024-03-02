# SupCon-MPL : Super Contrastive Learning with Meta Pseudo Label (SupCon-MPL) for Deepfake Image Detection

<image of supconmpl>

#### SupCon-MPL : Super Contrastive Learning with Meta Pseudo Label (SupCon-MPL) for Deepfake Image Detection
<names>
<github link>
<abstract>

# Requirements
<divider>
<bullet>
We recommend Linux for performance and compatibility reasons.

Python libraries: see requirement.txt for exact library dependencies. You can use the following commands with docker after downloading datasets and placing them right folder to create your environment:
[
 docker-compose up -d
]

# Getting started

## Pretrained models
Download the whole [models] folder from [link] and put it under the root dir.

Pre-trained networks are stored as [*.tar] files

## Datasets
DFDC(part 0, 4, 10, 17, 35, 40) : https://ai.meta.com/datasets/dfdc/
FaceForensics++ : https://github.com/ondyari/FaceForensics
Celeb-DF : https://github.com/yuezunli/celeb-deepfakeforensics
CelebA : https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
StyleGAN : https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

You can get cropped imgs following this github : https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection

Dataset folders should be:
-DFDC_imgs
 - video_name1 ...
-DFDC_videos
 - video_name1 ...
FaceForensics
 - train_set
 - test_set
StyleGAN
 - imgs
CelebDF
 - youtube_real_imgs
 - fake_imgs
CelebA
 - imgs

You must put all dataset path in "utils.py".
[

origin_data_path = 'DFDC_video_path' # original DFDC dataset path(videos)
  cropped_data_path_dfdc = 'DFDC_face_imgs_path' # cropped face dataset path(imgs) of DFDC
  cropped_data_path_face = 'FaceForensics_face_imgs_path_train' # cropped face dataset path(imgs) of FaceForensics
  cropped_data_path_stylegan = 'StyleGAN_face_imgs_path' # stylegan face image datset path
  cropped_data_path_celebv2 = 'CelebDF_face_imgs_path' # cropped face dataset path(imgs) of CelebDF
  cropped_data_path_youtube_real = 'CelebDF_Youtube_face_imgs_path' # cropped face dataset of youtube videos in CelebDF
  cropped_data_path_celeba = 'CelebA_face_imgs_path' # cropped face dataset path(imgs) of CelebA

  ....

  origin_data_path = 'DFDC10_video_path' # original dataset path(videos) of DFDC part 10
  cropped_data_path_dfdc = 'DFDC10_face_imgs_path' # cropped face dataset path(imgs) DFDC part 10
  cropped_data_path_face = 'FaceForensics_face_imgs_path_test' # cropped face dataset path(imgs) -> FaceForensics

]

# Training and Evaluation
After figuring out all the requirements, you can simply run [model_to_train.sh] to train and evaluate models.
[sh]
[sh]
[sh]

If you hope to only evaluate models, 주석 맨 윗줄 in ~.sh.
