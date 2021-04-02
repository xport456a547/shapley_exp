# shapley_exp

Requires pytorch 1.6 and
```
pip install captum tqdm scipy imageio
```
Get dataset
```
sh get_dataset.sh
```
Exemple Usage
```
python main.py \
  --train_config "config/train_config/unmasked/train_config_pretrain.json" \
  --model_config "config/model_config/equalsurplus/equalsurplus_64_8_balanced.json"
```
