#!/bin/bash

# 新建环境脚本
# 用于设置 Transferable Multimodal Attack 项目的Conda环境

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始设置 TMM 项目环境（Conda）"
echo "=========================================="

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda，请先安装 Anaconda 或 Miniconda"
    echo "下载地址: https://www.anaconda.com/products/individual 或 https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Conda 版本: $(conda --version)"

# 环境名称
ENV_NAME="tmm"

# 检查是否已存在环境
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "警告: Conda环境 '${ENV_NAME}' 已存在"
    read -p "是否删除并重新创建? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除现有环境..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "使用现有环境"
        echo "激活环境使用: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

# 创建conda环境
echo "创建Conda环境 '${ENV_NAME}' (Python 3.8)..."
conda create -n "${ENV_NAME}" python=3.8 -y

# 激活环境
echo "激活Conda环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装项目依赖..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "依赖安装完成！"
else
    echo "警告: 未找到 requirements.txt 文件"
fi

echo ""
echo "=========================================="
echo "环境设置完成！"
echo "=========================================="
echo ""
echo "使用以下命令激活环境:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "退出环境使用:"
echo "  conda deactivate"
echo ""

