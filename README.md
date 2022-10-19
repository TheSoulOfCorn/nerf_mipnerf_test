# NERF IMPLEMENTATION for UCSD_CSE274_PROJ

#### before start
> This is a nerf implementation based on torch/torchlightning, mostly only supporting real scene (LLFF and 360° data).   
> Hardware: WSL2 on Windows 11, RTX 3070

## RUNNING
#### create enviroment
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
> - MUCH CREDIT to the second reference for the usage of torchlightning, and I also learnt a lot from the structure of that work.
> - As some feedbacks, I appended some notes when I learn Nerf and the this Pytorch Code. Notes are combinations in English, Chinese characters and sketches, flow charts, focusing on the rays generation and changing between coordinates. Those bothers me much at first. Though I heavily noted the code, it's still not pretty clear for first learn or just going through. That's why I took a look at the code again and made further notes, and it clearly explains what's going on in every single detail, including some tuning that never disclosed at any material I tried. I believe it's good for anyone wants to dive deep. Still, I'm sorry for having no time to have a nice configuration of the notes, but I'd love to make some translation/reorganization/clarification if anyone interested!!!

### RESULTS FROM UCSD SCENES   

https://user-images.githubusercontent.com/68495667/196556112-cec3554d-9a67-4787-ae59-2c1fe5d6454c.mp4   

https://user-images.githubusercontent.com/68495667/196556516-64bb5631-f161-4846-b937-6fc3cb832c63.mp4    



https://user-images.githubusercontent.com/68495667/196556734-fe04b75c-1059-440a-8308-2b1b9f95dfb8.mp4



https://user-images.githubusercontent.com/68495667/196556741-1386f8d0-fb98-4d11-b27e-4f0b9971d483.mp4




https://user-images.githubusercontent.com/68495667/196556774-40c14ea8-72f3-45c5-8349-429a71bf92f4.mp4



https://user-images.githubusercontent.com/68495667/196556778-c49d287b-0512-4c58-85ef-2bb87dc208e6.mp4




https://user-images.githubusercontent.com/68495667/196556787-9e501a84-7537-43e9-b24b-76d72acd9fdf.mp4



https://user-images.githubusercontent.com/68495667/196556796-6f1ba406-10ef-4933-b818-f0b12b116b54.mp4


![bear_loss](https://user-images.githubusercontent.com/68495667/196556886-0af14ead-b2ac-48f5-a818-8602293d3581.png)
![bear_psnr](https://user-images.githubusercontent.com/68495667/196556901-d0260101-613d-4f21-9929-5f8ff6434c9c.png)
![kitchen_loss](https://user-images.githubusercontent.com/68495667/196557451-2d25fa90-a851-46b9-93e4-bc4cb7078c8f.png)
![kitchen_psnr](https://user-images.githubusercontent.com/68495667/196557459-ebe70492-3834-4946-a547-0bd9aa836f1b.png)
![lib_loss](https://user-images.githubusercontent.com/68495667/196557490-4eeec253-c958-4f03-863f-b2dae2e58a79.png)
![lib_psnr](https://user-images.githubusercontent.com/68495667/196557495-53e0c9be-22f9-4f5f-8691-27528af657af.png)

### REFERENCE   
>[nerf](https://github.com/bmild/nerf)   
[nerf_pl](https://github.com/kwea123/nerf_pl)
