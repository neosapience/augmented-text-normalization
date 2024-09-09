# augmented-text-normalization
This repository supports data augmentation using GPT-4o-mini for [NVIDIA Neural Text Normalization Models](https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/duplex_text_normalization).

Generated sample data is available in `sample_data` directory.

## Data Augmentation
**Note**: Since NVIDIA provides [a pretrained model](https://ngc.nvidia.com/catalog/models/nvidia:nemo:neural_text_normalization_t5) trained on [Google Text Normalization Dataset](https://www.kaggle.com/datasets/google-nlu/text-normalization), check it first.

### Setup
```sh
pip install -r requirements.txt
```
### Run Augmentation
This repository has two settings for data augmentation.
```sh
# Based on your own text, which has a sentence to normalize for every line.
python augment_data.py --input_path <YOUR_TXT_PATH>
```
```sh
# GPT-4o-mini generates challenging sentences itself.
python augment_data.py --augment_from_scratch --sentence_num_from_scratch <TOTAL SENTENCES TO GENERATE>
```
In addition, if you have enough rate limits for OpenAI API, parallel augmentation is supported by:
```sh
python augment_data.py --workers <NUMBER OF PARALLEL THREADS>
```
For further details, check the help messages by `python augment_data.py --help`.

## Train
Basically, follow the instructions in the [training script](https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/duplex_text_normalization/duplex_text_normalization_train.py). Since our pipeline generates a google-style .tsv data, direct usage is supported.
### Environment
Recommend to follow [NeMo](https://github.com/NVIDIA/NeMo).
```sh
conda create --name nemo python==3.10.12
conda activate nemo
```
```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
```sh
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
pip install nemo_toolkit['nlp']
```
It is highly likely that you will experience package version errors, due to NeMo's notorious version incompatibility. We recommend downgrading `huggingface_hub` and `transformers` if there is any problem.

Furthermore, if you experience errors related to `val_loss` logging error, consider adding `         self.log("val_loss", val_loss)` to `line 163` of `~/.conda/envs/nemo/lib/python3.10/site-packages/nemo/collections/nlp/models/duplex_text_normalization/duplex_decoder.py`

### Train
Configure your training setting at `~/NeMo/examples/nlp/duplex_text_normalization/conf/duplex_tn_config.yaml`. Then simply run
```sh
python ~/NeMo/examples/nlp/duplex_text_normalization/duplex_text_normalization_train.py
```
