# NERF & MIPNERF IMPLEMENTATION for UCSD_CSE274_PROJ
##### By Mohan Li, EMAIL: mol002@ucsd.edu

## FOR MILESTONE1
This part is for the requirements of project milestone for CSE274.   
- For current results, I have finished most of the work w.r.t. nerf, achieved similar result. Please check:   
repo files for the implementation codes   
RESULTS FROM UCSD SCENES (AND FLOWER) part below for a rendered video of UCSD and flower scene    
COMPARISON BETWEEN ORIGINAL NERF AND NERF THIS REPO part below for a comparison result of training   
- For next step, I would like to:   
finish the rest mipnerf part in this repo. Please check the rest parts for the visual and comparison data of mipnerf, I have already finished them all but still working on the organization of codes for repo.   
begin to implement some interesting dynamic nerf projects, and to see if possible to combine the improved mipnerf with the dynamic scenes.   
There is no change in project directions.
- others   
I'm sorry that the quality of results may not as ideal. This is mostly because I have to shrink down the training loops/ image size/ sample size for a quick result for debug / limited memory. If time allows I will update more visual pleasant results!

## FOR MILESTONE2
- 11/1  mipnerf code is updated
- 11/2  updated mipnerf rendered video with zoom-in, sorry for the transition is not very smooth. I basically generate video with two different camera positions. A smooth transition video may be polished with new position generation code later.   
- others ( slight change of presentation logic )   
I would like to show my own nerf/mipnerf results/comparisons in the presentation of mip-nerf paper. I will not redo that in my final project presentation (though this is part of it). Instead I will concentrate on introduction of the neural scene flow fields.

#### before start
> This is a nerf & mipnerf implementation based on torch/torchlightning, mostly only supporting real scene (LLFF and 360° data).   
> Hardware: WSL2 on Windows 11, RTX 3070. Memory consumption is less than 10 GB using my default setting.

## RUNNING NERF
#### create environment
> run   
`conda env create -f environment.yml`   
`pip install -r requirements.txt`   
better check the sanity of the GPU status before going next step, as I developed in WSL2 so the environment setting may be not as universal.

#### prepare your data   
> we use pretty much same setting as the original paper. run `python img2poses.py $your-images-folder` from [LLFF](https://github.com/Fyusion/LLFF) to get poses from [COLMAP](https://github.com/colmap/colmap). Surely you can directly use [COLMAP](https://github.com/colmap/colmap) by yourself.
 
#### get configuration and run
> the `config.py` and `config_test.py` provides options for training and testing.   
I recommend you check the configuration files above first, and simply use my scripts where little has to be changed for your own setting. since this is a simplified implementation, unnecessary options are all muted. After you have changed the setting in the `.sh` file, just run:   
`bash scripts/quick_train.sh`   
`bash scripts/quick_test.sh`  
during training, the checkpoints will be saved in `./ckpts`, training visualization tfevents file in `./logs`   
during testing, the images, depth images and videos (depends on user preference) will be saved in `./results`   

#### others
> - I have heavy notation everywhere in the code to lessen confusion, and I on purpose removed some features/tuning options for clarity. The remained in the configuration files are those I consider most crucial.
> - Unfortunately the code only supports real scene data, as this is a course project after all, I would like to have things focused.
> - The results below are trained with resized images to one eighth of original iphone picture size due to limited memory and time.. Should be much better with less shrank images.
> - MUCH CREDIT to the second reference for the usage of torchlightning, and I also learnt a lot from the structure of that work.
> - As some feedbacks, I appended some notes when I learn Nerf and the this Pytorch Code. Notes are combinations in English, Chinese characters and sketches, flow charts, focusing on the rays generation and changing between coordinates. Those bothers me much at first. Though I heavily noted the code, it's still not pretty clear for first learn or just going through. That's why I took a look at the code again and made further notes, and it clearly explains what's going on in every single detail, including some tuning that never disclosed at any material I tried. I believe it's good for anyone wants to dive deep. Still, I'm sorry for having no time to have a nice configuration of the notes, but I'd love to make some translation/reorganization/clarification if anyone interested!!!

## RUNNING MIPNERF
#### create environment
> identical to the nerf environment setting

#### prepare your data   
> identical to the nerf environment setting

#### get configuration and run
> the `config_mipnerf.py` and `config_mipnerf_test.py` provides options for training and testing.
I didn't create any quick bash files for mipnerf, while it's simple to run. You may do so or just run py file:   
`python trainMip.py`   
`python testMip.py`   
add any args you want!   
during training, the checkpoints will be saved in `./ckpts_mip`, training visualization tfevents file in `./logs_mip`   
during testing, the images, depth images and videos (depends on user preference) will be saved in `./results_mip` 

#### others
> - I actually had not run the organized code here due to some model loading issue, but it should be working fine.
> - Again, sorry for not having everything pretty! If any misleading or trouble please feel free to contact me.
> - MUCH THANKS to the reference repo!
> - LLFF scene is not originally in the paper, you may check some following works like mipnerf 360° for a better result of LLFF.
> - fact that I will present mipnerf in my class. I update some slides I will be using. Inpired by mipnerf original website. As well, one page of my notes! (JUST FOR INTEREST)

## RESULTS FROM UCSD SCENES (AND FLOWER)

https://user-images.githubusercontent.com/68495667/196556112-cec3554d-9a67-4787-ae59-2c1fe5d6454c.mp4   

https://user-images.githubusercontent.com/68495667/196556516-64bb5631-f161-4846-b937-6fc3cb832c63.mp4    



https://user-images.githubusercontent.com/68495667/196556734-fe04b75c-1059-440a-8308-2b1b9f95dfb8.mp4



https://user-images.githubusercontent.com/68495667/196556741-1386f8d0-fb98-4d11-b27e-4f0b9971d483.mp4




https://user-images.githubusercontent.com/68495667/196556774-40c14ea8-72f3-45c5-8349-429a71bf92f4.mp4



https://user-images.githubusercontent.com/68495667/196556778-c49d287b-0512-4c58-85ef-2bb87dc208e6.mp4




https://user-images.githubusercontent.com/68495667/196556787-9e501a84-7537-43e9-b24b-76d72acd9fdf.mp4



https://user-images.githubusercontent.com/68495667/196556796-6f1ba406-10ef-4933-b818-f0b12b116b54.mp4

## COMPARISON BETWEEN ORIGINAL NERF AND NERF THIS REPO

![bear_loss](https://user-images.githubusercontent.com/68495667/196556886-0af14ead-b2ac-48f5-a818-8602293d3581.png)
![bear_psnr](https://user-images.githubusercontent.com/68495667/196556901-d0260101-613d-4f21-9929-5f8ff6434c9c.png)
![kitchen_loss](https://user-images.githubusercontent.com/68495667/196557451-2d25fa90-a851-46b9-93e4-bc4cb7078c8f.png)
![kitchen_psnr](https://user-images.githubusercontent.com/68495667/196557459-ebe70492-3834-4946-a547-0bd9aa836f1b.png)
![lib_loss](https://user-images.githubusercontent.com/68495667/196557490-4eeec253-c958-4f03-863f-b2dae2e58a79.png)
![lib_psnr](https://user-images.githubusercontent.com/68495667/196557495-53e0c9be-22f9-4f5f-8691-27528af657af.png)

## COMPARISON OF RESULTS BETWEEN ORIGINAL NERF AND MIPNERF

![Picture9](https://user-images.githubusercontent.com/68495667/198183820-80c82d25-8611-4244-bc39-a5a4c5ad13a8.png)

![Picture5](https://user-images.githubusercontent.com/68495667/198183257-a625c445-4683-4896-929f-8669e1b91d6a.png)

![Picture11](https://user-images.githubusercontent.com/68495667/198184474-88c1d626-e843-4355-9888-ae19d52e818a.png)

![bear_loss](https://user-images.githubusercontent.com/68495667/198184727-a205f110-94e0-4103-9494-203750573bd7.png)
![bear_psnr](https://user-images.githubusercontent.com/68495667/198184732-f305c4a6-4c3b-414e-b8d9-62deef2689b0.png)
![kitchen_loss](https://user-images.githubusercontent.com/68495667/198184750-8b369e51-c474-4946-b212-25a2284c6ec9.png)
![kitchen_psnr](https://user-images.githubusercontent.com/68495667/198184755-4f0acdc9-bf67-4dc3-b157-63b54e802e26.png)
![library_loss](https://user-images.githubusercontent.com/68495667/198184768-84c194e8-0fdd-43b2-8ed6-4f326fed0ff7.png)
![library_psnr](https://user-images.githubusercontent.com/68495667/198184771-4cad767b-fdef-4b3a-a9c5-ecfdefafe3d7.png)
![flower_loss](https://user-images.githubusercontent.com/68495667/198184774-7df89f3d-067b-4d67-8a1a-860dc5ce0fac.png)
![flower_psnr](https://user-images.githubusercontent.com/68495667/198184777-eda484fc-87ac-4265-85db-f8dd1108bb61.png)

## MIPNERF VIDEO WITH ZOOM-IN


https://user-images.githubusercontent.com/68495667/199646357-e7e964fc-a10b-4826-84d1-084720d5008b.mp4

https://user-images.githubusercontent.com/68495667/199646364-0e851eb1-b176-4e8c-ad1a-63cf13df2d46.mp4

https://user-images.githubusercontent.com/68495667/199646370-189294c9-6b35-4653-9139-56cee8893994.mp4



### REFERENCE   
>[nerf](https://github.com/bmild/nerf)   
[nerf_pl](https://github.com/kwea123/nerf_pl)   
[mipnerf_pl](https://github.com/kwea123/mipnerf_pl)   
