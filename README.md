# Отчёт Comet

**https://www.comet.com/boldirev-as/hw05/reports/3Wzmk4ExsRw5xTqQ4JUY05XOh**

# Демо

Ноутбук находится в `demo.ipynb`

## Установка

```bash
git clone https://github.com/boldirev-as/hw05.git
cd hw05
pip install -r requirements.txt
```

## Чекпоинт

```bash
wget -O best.pt https://github.com/boldirev-as/hw05/releases/download/checkpoint-v1/best.pt
```

## Обучение

ADMM-100:

```bash
python train.py --config src/configs/train_admm.yaml --comet
```

LeADMM-20:

```bash
python train.py --config src/configs/train_leadmm.yaml --comet
```

Modular LeADMM-5:

```bash
python train.py --config src/configs/train_modular.yaml --comet
```

Только pre-processor:

```bash
python train.py --config src/configs/train_modular_pre.yaml --comet
```

Только post-processor:

```bash
python train.py --config src/configs/train_modular_post.yaml --comet
```

Продолжение обучения Modular LeADMM-5:

```bash
python train.py --config src/configs/train_modular.yaml --ckpt best.pt --lr 1e-4 --l1 0.0 --comet --name continue_modular
```

## Инференс

Папка с данными:

```text
data/
  lensless/
  masks/
  lensed/
```

Запуск:

```bash
python inference.py --config src/configs/inference_custom.yaml --data /path/to/data --ckpt best.pt --out recon
```

## Метрики

```bash
python calculate_metrics.py --gt /path/to/data/lensed --pred recon --size 256
```

## Скорость

```bash
python benchmark.py --config src/configs/benchmark_modular.yaml
```
