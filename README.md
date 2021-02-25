# shapley_exp

Requires pytorch 1.6 and
```
pip install captum tqdm scipy imageio
```
Get dataset
```
sh get_dataset.sh
```
Usage
```
python main.py \
  --train_config "config/train_config/train_config.json" \
  --model_config "config/model_config/equalsurplus.json"
```
