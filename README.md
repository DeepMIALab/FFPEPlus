

# FFPE++: Improving the Quality of Formalin-Fixed Paraffin-Embedded Tissue Imaging via Contrastive Unpaired Image-to-Image Translation

<img src="imgs/FFPE++_full_pipeline_JUNE (1).png" width="800px"/>


In this work, we introduce FFPE++ to improve the quality of FFPE tissue sections using an unpaired image-to-image translation technique that converts FFPE images with artifacts into high-quality FFPE images without the need for explicit image pairing and annotation.

## Example Results

### FFPE artifacts correction in Lung Specimens
<img src="imgs/brain_gif.gif" width="800px"/>

### Frozen to FFPE Translation in Lung Specimens
<img src="imgs/lung_gif.gif" width="800px"/>


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN


### Getting started

- Clone this repo:
```bash
git clone https://github.com/DeepMIALab/FFPE
cd FFPEPlus   
```

- Install PyTorch 1.1 and other dependencies (e.g., torchvision, visdom, dominate, gputil).

- For pip users, please type the command `pip install -r requirements.txt`.

- For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.

### Training and Test

- To replicate the results, you may download [OV](https://portal.gdc.cancer.gov/projects/TCGA-OV)  project for Ovary, [LUAD](https://portal.gdc.cancer.gov/projects/TCGA-LUAD) and [LUSC](https://portal.gdc.cancer.gov/projects/TCGA-LUSC) projects for Lung, and [THCA](https://portal.gdc.cancer.gov/projects/TCGA-THCA) project for Thyroid from TCGA Data Portal and create a subset using these .txt files.
- To extract the patches from WSIs and create PNG files, please follow the instructions given in [FFPEPlus/Data_preprocess](https://github.com/DeepMIALab/AI-FFPE/tree/main/Data_preprocess) section.

The data used for training are expected to be organized as follows:
```bash
Data_Path                # DIR_TO_TRAIN_DATASET
 ├──  trainA
 |      ├── 1.png     
 |      ├── ...
 |      └── n.png
 ├──  trainB     
 |      ├── 1.png     
 |      ├── ...
 |      └── m.png
 ├──  testA
 |      ├── 1.png     
 |      ├── ...
 |      └── j.png
 └──  testB     
        ├── 1.png     
        ├── ...
        └── k.png

```

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

- Train the FFPE++ model:
```bash
python train.py --dataroot ./datasets/Frozen/${dataroot_train_dir_name} --name ${model_results_dir_name} --CUT_mode CUT --batch_size 1
```

- Test the FFPE++ model:
```bash
python test.py --dataroot ./datasets/Frozen/${dataroot_test_dir_name}  --name ${result_dir_name} --CUT_mode CUT --phase test --epoch ${epoch_number} --num_test ${number_of_test_images}
```

The test results will be saved to a html file here: ``` ./results/${result_dir_name}/latest_train/index.html ``` 



### AI-FFPE, AI-FFPE without Spatial Attention Block, AI-FFPE without self-regularization loss, CUT, FastCUT, and CycleGAN

<img src="imgs/ablation.png" width="800px"/>


## Reference

If you find our work useful in your research or if you use parts of this code please consider citing our paper:



### Acknowledgments
Our code is developed based on [CUT](https://github.com/taesungp/contrastive-unpaired-translation).
