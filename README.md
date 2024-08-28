# README 

This is the official source code for "On the Credibility of Backdoor Attacks Against Object Detectors in the Physical World" 

Arxiv Version: [Paper](https://arxiv.org/abs/1908.03369)

The project is published as part of the following paper and if you re-use our work, please cite the following paper:


```
@inproceedings{doan2024physical,
title={On the Credibility of Backdoor Attacks
Against Object Detectors in the Physical World},
author={Bao Gia Doan and Ehsan Abbasnejad and Damith C. Ranasinghe},
year = {2024},
booktitle = {Proceedings of the 40th Annual Computer Security Applications Conference (ACSAC)},
location = {Waikiki, Hawaii, USA},
series = {ACSAC 2024}
}
```

# Requirements

To install the requirements for this repo, run the following command: 

```sh
git clone https://github.com/AdelaideAuto-IDLab/PhysicalBackdoorDetectors.git
cd PhysicalBackdoorDetectors
pip install -r requirements.txt
# choose the cuda version that match with your cuda driver (check using nvcc --version)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```



# Large Files and Datasets

We are realeasing the physical video set so you will need to download the dataset as well as pretrained weights in order to run our pipeline


- Download the *Physical Dataset* file: [download](https://app.box.com/s/ig3ffkynkc2dp21hah7z0m94hr8poxuv)

- Download the *Pretrained weights* file [download]() 

- After downloading the dataset, move the dataset folder (PHYSICAL_DATASET) into our repo (`PhysicalBackdoorDetectors/PHYSICAL_DATASET`)
```
mv ~/Downloads/<dataset> ./PHYSICAL_DATASET (assumed we are at root of PhysicalBackdoorDetectors)
```

# Poisoning Stage

1. Poison the clean dataset

- Follow the command to poison clean dataset:

```python
python3 poison.py --ratio 0.15 --out_dir YOLO_Blue_High_p015 --attack_id High --data_yaml yolo_blue_high_015.yaml
```

# Evaluation

***Follow the command to run the evaluation for each model*** (This will replicate Table 3, which is our main table that include results of all models with different trigger locations)
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

**Results will be saved in RESULTS/ directory with csv file containing ASR evaluation of each model**