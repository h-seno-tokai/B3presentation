
# GPU 対応 TensorFlow セットアップ奮闘記

## 1. 概要

Windows 11 + Anaconda 環境において、NVIDIA GPU (RTX 2080) を使った TensorFlow の実行がうまくいかなかった問題を解決するまでの記録です。  
以下のような流れで試行錯誤を行い、最終的に **GPU を認識する TensorFlow** を使えるようになりました。

- OS: Windows 11
- GPU: NVIDIA GeForce RTX 2080
- Python Distribution: Anaconda (miniconda / anaconda3)
- 目的: TensorFlow (GPU 対応) + CUDA/cuDNN を正しくセットアップし、MNIST などの学習を高速化する

---

## 2. 当初の問題

1. **GPU が認識されない**  
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   # 結果: []
   ```
   となり、GPU が空リスト。

2. **Windows に最新 CUDA (12.6) をインストールしていた**  
   - しかし、TensorFlow 公式が対応しているのは主に CUDA 11.x。  
   - さらに、`where nvcc` で CUDA 12.6 が表示されてしまう。  

3. **複数バージョンの CUDA が共存**  
   - システムに CUDA 12.6 が入りつつ、conda 環境に CUDA 11.8 を入れても衝突が起こりうる。

---

## 3. 解決の流れ

### 3.1 仮想環境の作成と GPU 対応ライブラリのインストール

1. **Anaconda (base 環境) をクリーンに**  
   - いろいろパッケージを入れて依存関係が混乱していたため、Anaconda を再インストールしたり、最低限の状態に整えた。

2. **新しい仮想環境 `tf-gpu` を作成**  
   ```bash
   conda create -n tf-gpu python=3.9
   conda activate tf-gpu
   ```

3. **CUDA 11.8 と cuDNN を導入**  
   - 最新版 TensorFlow は CUDA 11.x + cuDNN 8.x を推奨しているため。
   - `conda-forge` や `nvidia` チャネルを使って `cudatoolkit=11.8` + `cudnn 8.x` を入れようとした。  
   - 例:
     ```bash
     conda install -c conda-forge cudatoolkit=11.8
     conda install -c conda-forge cudnn=8.8.0.121
     ```
   - パッケージが見つからない場合はバージョンを変えたり、チャンネルを切り替えたりなど試行錯誤。

4. **TensorFlow のインストール**  
   - 当初は `pip install --upgrade tensorflow` で最新（2.18.x）を入れようとしたが、`tensorflow-intel`（CPU版）が入り GPU を認識せず。  
   - バージョンダウンを提案 → **TensorFlow 2.10.*** なら GPU 版が含まれている。

   ```bash
   pip uninstall tensorflow tensorflow-intel
   pip install --upgrade tensorflow==2.10.*
   ```

5. **確認**  
   ```python
   import tensorflow as tf
   print(tf.__version__)  # 2.10.x
   print(tf.config.list_physical_devices('GPU'))
   # => [PhysicalDevice(name='/physical_device:GPU:0', ...)]
   ```
   これで GPU が認識された！

---

### 3.2 Jupyter / VSCode との連携

1. **Jupyter Notebook のカーネル**  
   - 仮想環境 `tf-gpu` 上で `ipykernel` をインストールし、Notebook でカーネルを選択。  
     ```bash
     (tf-gpu) pip install ipykernel
     ```
   - ノートブック内で
     ```python
     import sys
     print(sys.executable)
     ```
     すると、`.../anaconda3/envs/tf-gpu/python.exe` と表示されるか確認。

2. **VSCode のターミナル**  
   - `conda init powershell` を行い、PowerShell を再起動。  
   - `conda activate tf-gpu` すると `(tf-gpu)` に切り替わる。  
   - `where python` が `...\envs\tf-gpu\python.exe` を指していれば OK。  

3. **VSCode の「Python: Select Interpreter」** でも `tf-gpu` を選択するように設定。

---

## 4. 実際に実行するコード (MNIST + 日本語フォント例)

以下のコード例に必要なのは `numpy` と `matplotlib` と `tensorflow` くらいです。  
メイリオフォントを使うために `matplotlib.font_manager` を利用しています。

```python
import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist

# 日本語フォント設定
font_path = 'C:\\Windows\\Fonts\\meiryo.ttc'
font_prop = fm.FontProperties(fname=font_path)

# MNIST データセットのロード
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# 動作確認用
print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

# GPU の確認
print("GPU:", tf.config.list_physical_devices('GPU'))
```

---

## 5. 最終的な環境と学んだこと

- **最終的に使用しているバージョン**  
  - Python 3.9  
  - TensorFlow 2.10.x (GPU 対応)  
  - CUDA 11.8, cuDNN 8.x  
  - Windows 11 + NVIDIA RTX 2080  

- **学んだこと・注意点**  
  1. **TensorFlow のバージョンごとに対応している CUDA/cuDNN が異なる**  
  2. **最新の TensorFlow (2.13, 2.18 等) だと Windows で GPU 版が配布されていないケースがある** → 結果として `tensorflow-intel`（CPU版）が入ってしまう。  
  3. **conda 環境内に `cudatoolkit` と `cudnn` を入れると、システム全体の CUDA バージョン（例: 12.6）と競合を避けられる**  
  4. **VSCode のターミナルや Jupyter のカーネルをきちんと `tf-gpu` 環境に合わせないと混乱する**  
  5. **焦らずバージョンを固定して試し、GPU が認識されるか都度確認するのが確実**  

---

## 6. 参考リンク

- [TensorFlow Install (Official)](https://www.tensorflow.org/install)
- [Anaconda Documentation](https://docs.anaconda.com/)
- [cuDNN Download (NVIDIA)](https://developer.nvidia.com/cudnn)
