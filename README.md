# On the Credibility of Backdoor Attacks Against Object Detectors in the Physical World

This is the official source code for "On the Credibility of Backdoor Attacks Against Object Detectors in the Physical World" 

Arxiv Version: [Paper](https://arxiv.org/abs/2408.12122)

The project is published as part of the following paper and if you re-use our work, please cite the following paper:


```
@inproceedings{doan2024physical,
title={On the Credibility of Backdoor Attacks
Against Object Detectors in the Physical World},
author={Bao Gia Doan,  Dang Quang Nguyen, Callum Lindquist, Paul Montague, Tamas Abraham, Olivier De Vel, Seyit Camtepe, Salil S. Kanhere, Ehsan Abbasnejad, and Damith C. Ranasinghe},
year = {2024},
booktitle = {Proceedings of the 40th Annual Computer Security Applications Conference (ACSAC)},
location = {Waikiki, Hawaii, USA},
series = {ACSAC 2024}
}
```

# File Structure

| File | Description |
| --------- | ----------- |
| [asr_eval.py](asr_eval.py) | evaluate attack success rate. |
| [attack.tsv](attack.tsv) | configuration for attacks (Location, size, color, etc...) |
| [dino_config ](dino_config) | configuration for DINO models. |
| [labels.csv ](labels.csv) | Labels to create poisoning dataset. |
| [model.py](model.py) | configuration of model architecture. |
| [requirements.txt](requirements.txt) | code to poison dataset. |
| [poison.py](poison.py) | code to poison dataset. 
| [script](script) | code to poison dataset. |
| [signs_object](signs_object) | store all traffic signs for MORPHING in MTSD |
| [script](script) | store all the scripts to generate results. |
| [transformations](transformations) | code to poison dataset. |
| [yolv5](yolov5) | configuration and model architecture of YOLOv5. |


# Requirements

To install the requirements for this repo, run the following command: 

```sh
git clone https://github.com/AdelaideAuto-IDLab/PhysicalBackdoorDetectors.git
cd PhysicalBackdoorDetectors
```
Create virtual environment using anaconda:
```sh
conda create -n backdoor python=3.10
conda activate backdoor
```

Then we install all the necessary packages using:

```sh
pip install -r requirements.txt
# choose the cuda version that match with your cuda driver (check using nvcc --version)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

### GPU Requirements
We use NVIDIA A6000 (49GB) for our experiments.






# Large Files and Datasets

We are realeasing the physical video set so you will need to download the dataset as well as pretrained weights in order to run our pipeline


- Download the *Physical Dataset* file: [download](https://universityofadelaide.box.com/s/a0ixwqwj5myupvitcg5lo51ektzuj6c3)

- Download the *Pretrained weights* file [download](https://universityofadelaide.box.com/s/g5bmsxpwvlkhgj566xr05dkoxr9tsk92) 

- Download the MTSD scenes for poisoing stage [download](https://universityofadelaide.box.com/s/kiqm83x8jqdmnuzq632uad49qa8wibx1)

- After downloading the dataset, move the dataset folder from Downloads directory into our repo (`PhysicalBackdoorDetectors/PHYSICAL_DATASET`)
```sh
cd ~/Downloads/
mv PHYSICAL_DATASET.zip ~/PhysicalBackdoorDetectors/
cd ~/PhysicalBackdoorDetectors/
unzip PHYSICAL_DATASET.zip
```

- After downloading the weights, move the weights folder from Downloads directory into our repo (`PhysicalBackdoorDetectors/WEIGHTS`)
```sh
cd ~/Downloads/
mv WEIGHTS.zip ~/PhysicalBackdoorDetectors/
cd ~/PhysicalBackdoorDetectors/
unzip WEIGHTS.zip
```

- After downloading the MTSD scenes, move the data folder from Downloads directory into our repo (`PhysicalBackdoorDetectors/MTSD_scenes`)
```sh
cd ~/Downloads/
mv MTSD_scenes.zip ~/PhysicalBackdoorDetectors/
cd ~/PhysicalBackdoorDetectors/
unzip MTSD_scenes.zip
```

# Poisoning Stage

Poison the clean dataset (This step is not necessary for artifact evaluation. You can skip to Evaluation section)

- Follow the command to poison clean dataset:

```python
python3 poison.py --ratio 0.15 --out_dir YOLO_Blue_High_p015 --attack_id High --data_yaml yolo_blue_high_015.yaml
```

# Training Stage 

In this stage, we will train model with our poisoned dataset to employ backdoor effect into the model. To add backdoor with different models, please follow the guide of those models on their repo and use this dataset to train.

***Follow the command to run the training for yolov5***

```python
python train.py --data yolo_blue_high_015.yaml --cfg yolov5s6.yaml --weights yolov5s6.pt --hyp ./yolov5/data/hyps/hyp.scratch.yaml --epochs 100 --batch-size 8 --project YOLO_Backdoor --name high_setting
```

**You need to log in to wandb in your console before running the training script**

# Evaluation

***Follow the command to run the evaluation for each model*** (This will replicate Table 3, which is our main table that include results of all models with different trigger locations)

**Traffic Signs**
- YOLO
    ```sh
    bash script/eval_yolo.sh
    ```
- FRCNN
    ```sh
    bash script/eval_frcnn.sh
    ```
- DETR
    ```sh
    bash script/eval_detr.sh
    ```
- DINO

    **ONLY** for DINO: Compiling CUDA operators

    ```sh
    cd dino_config/models/dino/ops
    python3 setup.py build install
    # unit test (Optional but should see all checking is True)
    python3 test.py
    cd ../../../..
    ```
    then run: 

    ```sh
    cd dino_config
    bash eval_dino.sh
    ```
**Drone**
- YOLO
    ```sh
    bash script/eval_drone.sh
    ```
- TPH-yolo
    ```sh
    bash script/eval_drone_tph.sh
    ```

**Results will be saved in RESULTS/ directory with csv file containing ASR evaluation of each model**


# Detailed Experimental Running

In our paper, we conduct experiments on below hardware:

> GPU Model: NVIDIA A6000
>
> GPU Number: 1


