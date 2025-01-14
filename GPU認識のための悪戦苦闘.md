
# GPU 対応 機械学習 セットアップ奮闘記

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
# PyTorch GPU 使用セットアップの記録

## 概要
Windows 11 + Anaconda 環境で PyTorch を使用する際に、GPU が正しく認識されない問題が発生した。これは CPU 版の PyTorch がデフォルトでインストールされていたため、GPU 版の PyTorch を再インストールして問題を解決した。ここには該問題とその解決手順を記録する。

## 問題
1. **CPU 版の PyTorch がデフォルトでインストールされた**  
    これにより、GPU が認識されない問題が発生。  

2. **GPU が認識されないコード例**:
    ```python
    import torch
    print(torch.cuda.is_available())
    # False が返ってくる
    ```

3. **TensorFlow で構築した CUDA/cuDNN と PyTorch の認識に違いがある**  
    - TensorFlow 用に構築した CUDA 11.8 + cuDNN 8.x は正しく動作していたが、PyTorch でも GPU を認識させるために GPU 版 PyTorch を再インストールする必要があった。

---

## 解決手順

### 1. CPU 版 PyTorch のアンインストール
1. **現在の PyTorch を削除**
    下記コマンドを VSCode や Anaconda Prompt で実行:
    ```bash
    pip uninstall torch torchvision torchaudio
    ```

---

### 2. GPU 版 PyTorch の再インストール
PyTorch の [公式インストールガイド](https://pytorch.org/get-started/locally/) に基づき、CUDA 11.8 に対応した PyTorch GPU 版をインストールした。

#### 2.1 pip を使う場合
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2.2 conda を使う場合
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

### 3. GPU 利用可能かの確認
1. Python ターミナルから下記を実行:
    ```python
    import torch
    print(torch.__version__)           # PyTorch のバージョン
    print(torch.version.cuda)          # 使用される CUDA バージョン
    print(torch.cuda.is_available())   # True なら GPU が認識されている
    ```

2. 結果、正しく GPU が認識されれば下記のような結果が返される:
    ```plaintext
    2.0.1+cu118
    11.8
    True
    ```

---

## 実装したコード例
GPU が認識された PyTorch で、MNIST データセットを使用したデータローダのコード:

```python
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# データローダの設定
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# GPU の利用確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# モデル例
model = torch.nn.Linear(784, 10).to(device)

# テンソルを試す
for images, labels in train_loader:
    images = images.view(-1, 28*28).to(device)
    labels = labels.to(device)
    outputs = model(images)
    print("Batch processed.")
    break
```

---

## 最終的な環境
- **OS**: Windows 11
- **Python**: 3.9
- **PyTorch**: 2.x (CUDA 11.8)
- **CUDA**: 11.8
- **cuDNN**: 8.x
- **GPU**: NVIDIA RTX 2080

## 学んだこと
1. PyTorch のバージョンに対応した CUDA/cuDNN のバージョンを選ぶことが重要。
2. TensorFlow との CUDA ライブラリを共有できることもあるが、PyTorch の実行バイナリを GPU 版にする必要がある。
3. 常に `torch.cuda.is_available()` で GPU が認識されているかを確認すること。

PowerShell で `where python` を実行しても何も表示されない場合、**システムの PATH** に Python が通っていない（または `conda activate` が効いていない）ことが考えられます。  
以下のポイントを順番にチェックしてみてください。

---

## 1. Anaconda 環境を PowerShell 上で有効化する方法

### 1.1 `conda init powershell` を実行

1. **Anaconda Prompt**（または `cmd.exe` でも可）を開き、下記コマンドを実行します。
   ```bash
   conda init powershell
   ```
2. 「Initialization finished successfully」と出れば OK です。  
3. PowerShell を一度すべて閉じ、再度 PowerShell を開いてみてください。  
   - すると、PowerShell 起動時に自動で conda が初期化され、`(base)` などが表示されるようになります。

### 1.2 `conda activate tf-gpu` する

PowerShell を開いたあとに、
```powershell
conda activate tf-gpu
```
と入力し、`(tf-gpu)` と表示されれば、**tf-gpu 環境**がアクティブになっています。  
この状態で、
```powershell
where python
```
と打てば、  
```
C:\Users\h-seno\anaconda3\envs\tf-gpu\python.exe
```
のように **tf-gpu 環境の python.exe** が表示されるはずです。

---

## 2. VSCode のターミナルで同じことを行う

VSCode のターミナルで PowerShell を使用している場合も、**事前に `conda init powershell` を済ませておけば**、VSCode の新しいターミナルでも `(base)` が自動表示されるようになります。

- もし `(base)` が表示されたら、`conda activate tf-gpu` で tf-gpu 環境へ切り替え。  
- その状態で
  ```powershell
  where python
  ```
  を実行すると、`tf-gpu` 環境のパスが返ってくるはずです。

### 2.1 VSCode の「Python: Select Interpreter」も確認

- VSCode 左下のステータスバー、または「コマンドパレット (`Ctrl+Shift+P`) → Python: Select Interpreter」で **`Python 3.9 (tf-gpu)`** を選ぶと、VSCode エディタ上の実行でも tf-gpu 環境が使われます。  
- ただし、**VSCode ターミナル** は別途 `conda activate tf-gpu` が必要（または自動化設定が必要）なので注意。

---

## 3. それでも表示されないときのチェック項目

1. **本当に Anaconda がインストールされているか**  
   - もし `conda` コマンド自体も認識されないなら、Anaconda の PATH が通っていない可能性が高いです。
   - Anaconda Prompt ではなく Windows 標準の PowerShell を開いていると、デフォルトで conda が使えない設定になっていることがあります。

2. **OneDrive 上のフォルダで実行している影響**  
   - 特にありませんが、OneDrive が原因で PATH が変わることは基本的にありません。  
   - ただし VSCode のワークスペースが OneDrive でも、Python がローカルにインストールされていれば問題ありません。

3. **Windows の実行ポリシー**  
   - まれに PowerShell の実行ポリシーが厳しく、`conda init powershell` したスクリプトが実行できない場合があります。  
   - その場合、`Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` などを設定してみてください（詳細は [Microsoft Docs](https://learn.microsoft.com/ja-jp/powershell/module/microsoft.powershell.security/set-executionpolicy) 参照）。

---

## 4. まとめ

1. **`conda init powershell` → PowerShell を再起動** して、`conda` が使える状態にする。  
2. **`conda activate tf-gpu`** で tf-gpu 環境をアクティブ化する。  
3. **`where python`** すれば `...\envs\tf-gpu\python.exe` が表示される。  
4. VSCode のターミナルでも同様に「PowerShell + conda init」済みであれば、`conda activate tf-gpu` で切り替えて使う。

こうすれば、PowerShell と VSCode のターミナル上でも問題なく Python インタープリタが見つかるようになるはずです。