# Transferable Multimodal Attack on Vision-Language Pre-training Models

This is the official PyTorch implementation of the paper "Transferable Multimodal Attack on Vision-Language Pre-training Models".

## Requirements
- Python 3.8
- Anaconda 或 Miniconda
- pytorch 1.10.2
- transformers 4.8.1
- timm 0.4.9
- bert_score 0.3.11

## 环境设置

### 快速设置（推荐）

使用提供的脚本自动设置Conda环境：

```bash
chmod +x setup_env.sh
./setup_env.sh
```

脚本会自动：
1. 检查Conda是否安装
2. 创建Conda环境 `tmm`（Python 3.8）
3. 安装所有必需的依赖包

### 手动设置

如果需要手动设置环境：

```bash
# 创建Conda环境
conda create -n tmm python=3.8 -y

# 激活环境
conda activate tmm

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

### 激活/退出环境

```bash
# 激活环境
conda activate tmm

# 退出环境
conda deactivate
```

## Download
- Dataset json files for downstream tasks [[ALBEF github]](https://github.com/salesforce/ALBEF)
- Finetuned checkpoint for ALBEF [[ALBEF github]](https://github.com/salesforce/ALBEF)
- Finetuned checkpoint for TCL [[TCL github]](https://github.com/uta-smile/TCL)
- Finetuned checkpoint for X-VLM [[X-VLM github]](https://github.com/zengyan-97/X-VLM)
- Finetuned checkpoint for ViLT [[ViLT github]](https://github.com/dandelin/ViLT)
- Finetuned checkpoint for METER [[METER github]](https://github.com/zdou0830/METER)

## Attack Multimodal Embedding
```
python EvalTransferAttack.py --adv 1 --gpu 0 \
--config ./configs/Retrieval_flickr.yaml \
--output_dir ./output/Retrieval_flickr \
--checkpoint [Finetuned checkpoint]
--log_name [log_name]
--save_json_name [save_json_name]
--config_name [config_name]
--save_dir [save_dir]
```
