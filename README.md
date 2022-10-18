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
[![Watch the video](https://img.youtube.com/vi/T-D1KVIuvjA/maxresdefault.jpg)](https://github.com/TheSoulOfCorn/nerf_test/blob/2cfb3b8bd501aa2517390f1ef8ead75de9a7afdb/sources/bear.mp4)
![nerfTomipnerf](https://github.com/TheSoulOfCorn/nerf_test/blob/2cfb3b8bd501aa2517390f1ef8ead75de9a7afdb/sources/bear.mp4)

### FUTURE PLAN
> experimental tuning options will be configurable, like rays/position options, mostly for comparison work.

### REFERENCE   
>[nerf](https://github.com/bmild/nerf)   
[nerf_pl](https://github.com/kwea123/nerf_pl)
