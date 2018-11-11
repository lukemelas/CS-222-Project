# Toderici et al. (2016)
Paper: https://arxiv.org/abs/1608.05148v2

Download [MS COCO](http://cocodataset.org/) and train: 

```
python train.py --train_dir data/coco-subset-train/ --val_dir data/coco-subset-val --batch_size 4 --log_every 128 --num_save_images 10
```
