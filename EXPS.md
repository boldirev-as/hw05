| name | command | best | result |
| --- | --- | ---: | --- |
| old_admm_like_5 | `--epochs 5 --batch 2 --size 256 --wandb` | 11.68 | bad |
| old_admm_like_30 | `--epochs 30 --batch 4 --size 256 --steps 5 --lr 3e-4` | 11.77 | bad |
| unet_rotate_256 | `--epochs 30 --batch 8 --size 256 --lr 3e-4` | 13.2 | best so far |
| unet384 | `--epochs 50 --batch 2 --size 384 --lr 2e-4` | 12.35 | bad, do not repeat |
| unet256_lr2e4 | `--epochs 50 --batch 8 --size 256 --lr 2e-4` | 13.1 | not better |
| residual_gn_aug | `--epochs 50 --batch 8 --size 256 --lr 3e-4` | 13.75 | best so far |
| mask_conditioned | `--epochs 60 --batch 8 --size 256 --lr 3e-4 --l1 0.1 --ema 0.999` | 12.25 | bad, do not repeat |
| mse_tta | `--epochs 70 --batch 8 --size 256 --lr 3e-4 --l1 0.0 --ema 0.999` | 13.65 | not better |

Next:

`--ckpt /kaggle/input/.../best.pt --epochs 40 --batch 8 --size 256 --base 64 --lr 1e-4 --l1 0.1 --ema 0.999 --dp --name continue_base64`
