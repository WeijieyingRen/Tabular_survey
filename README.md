# Tabular_survey
Protocol on Tabular data prediction, generation, test time adaptation

### Installing Dependencies


Install according to the dependencies of different baselines.
```
conda create -n [baseline] python=3.10
conda activate [baseline]
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### Baseline structure

```Generation/tab-ddpm```: tabDDPM, TVAE

```Generation/TABSYN```: TABSYN, CODI, STASY, GOGGLE

```TTA```: TENT, EATA, SHOT, TAST

### Preparing Datasets

Download dataset:

```

```

Place generation datasets into 'tab-ddpm/data' and 'TABSYN/data'.
Place TTA datasets into 'TTA/data'.

### Examples

<ins>TabDDPM</ins>

```bash
python tab-ddpm/scripts/pipeline.py --config [path_to_your_config] --train --sample --eval
python tab-ddpm/scripts/pipeline.py --config exp/adult/ddpm_cb_best/config.toml --train --sample –eval
```

<ins>TVAE</ins>

```bash
python tab-ddpm/CTGAN/pipeline_tvae.py --config [path_to_your_config] --train --sample --eval
python tab-ddpm/CTGAN/pipeline_tvae.py --config exp/adult/tvae/config.toml --train --sample –eval
```

<ins>TABSYN</ins>

```bash
python TABSYN/main.py --dataname [NAME_OF_DATASET] --method vae --mode train
python TABSYN/main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode train
python TABSYN/main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode sample
```

<ins>CODI/STASY/GOGGLE</ins>

```bash
python TABSYN/main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_BASELINE_METHODS] --mode train
python TABSYN/main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_BASELINE_METHODS] --mode sample
```

Options of [NAME_OF_DATASET]: abalone, fb-comments, insurance, king, adult, churn2, cardio, buddy
Options of [NAME_OF_BASELINE_METHODS]: goggle, stasy, codi

<ins>EATA</ins>

```bash
python EATA/tab_eata.py --datapath [PATH_OF_DATASET] --dataset [NAME_OF_DATASET] --model [mlp/FTTrans]
```

<ins>TENT</ins>

```bash
python TENT/tab_tent.py --datapath [PATH_OF_DATASET] --dataset [NAME_OF_DATASET] --model [mlp/FTTrans]
```

<ins>SHOT</ins>

```bash
python SHOT/object/image_source.py --datapath [PATH_OF_DATASET] --dataset [NAME_OF_DATASET] --model [mlp/FTTrans]
python SHOT/object/image_target.py --datapath [PATH_OF_DATASET] --dataset [NAME_OF_DATASET] --model [mlp/FTTrans]
```

<ins>TAST</ins>

```bash
python TAST/domainbed/scripts/train.py --datapath [PATH_OF_DATASET] --dataset [NAME_OF_DATASET] --model [mlp/FTTrans]
python TAST/domainbed/scripts/unsupervised_adaptation.py --datapath [PATH_OF_DATASET] --dataset [NAME_OF_DATASET] --model [mlp/FTTrans]
```