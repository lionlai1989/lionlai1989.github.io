---
layout: single
title: A Guide to Setup a PyTorch and TensorFlow Development Environment with GPU
categories: []
description: "."
toc: true
wip: false
date: 2024-01-01
---

The process of installing and updating essential components, such as the NVIDIA driver,
CUDA Toolkit, and cuDNN, to create a GPU development environment is introduced.

# Why Write This Guide?

Today, I attempted to install [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
within my `venv` virtual environment using `pip` but encountered difficulties despite
having successfully installed PyTorch, the NVIDIA driver, CUDA Toolkit, and cuDNN on my
system.

I discovered that PyTorch 2.\* requires CUDA Toolkit version 12.1, while my laptop
currently has CUDA Toolkit version 12.0. While I am not certain if this version mismatch
is the cause of the problem in installing PyTorch3D (probably not), I have decided to
reinstall everything and meticulously document the entire process of installing and
updating the NVIDIA driver, CUDA Toolkit, and cuDNN during the development workflow.

The objectives of this post are as follows:

-   Document the step-by-step process of installing and updating the NVIDIA driver, CUDA
    Toolkit, and cuDNN.
-   Record the procedure for creating a `venv` virtual environment, installing PyTorch,
    successfully running PyTorch utilities with GPU support, and installing PyTorch3D
    within the `venv` virtual environment.
-   In the event of unsuccessful installation of PyTorch3D within the `venv` virtual
    environment, create a Docker image to establish an environment capable of running
    PyTorch3D. This environment should be compatible with the following three scenarios:
    -   Jupyter Notebook
    -   Command-line shell
    -   Visual Studio Code, ensuring it recognizes the environment.

I use Ubuntu 22.04, and my GPU is the `NVIDIA RTX A3000 Laptop GPU`, equipped with 4096
CUDA cores and 6144 MB of memory. The current CUDA version is as follows:

```
$ nvidia-smi

Mon Jan 15 12:00:18 2024
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.147.05   Driver Version: 525.147.05   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA RTX A300...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   45C    P0    N/A /  60W |     10MiB /  6144MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1325      G   /usr/lib/xorg/Xorg                  4MiB |
|    0   N/A  N/A      2142      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
```

The system indicates that my laptop is currently running NVIDIA driver version
`525.147.05` and CUDA Toolkit version `12.0`. I intend to update the CUDA Toolkit to
version `12.1`.

# Installing NVIDIA Driver, CUDA Toolkit, and cuDNN

To provide a clear understanding, let's break down these terms:

-   NVIDIA Driver: This is the essential driver responsible for managing the GPU at a
    low level, potentially interacting with the GPU at the CPU register level.

-   CUDA Toolkit: Built upon the NVIDIA Driver, the CUDA Toolkit offers users an API
    (Application Programming Interface) to execute basic numeric operations. These
    operations include addition, subtraction, multiplication, and division, allowing for
    efficient GPU utilization.

-   cuDNN: Extending the functionality provided by the CUDA Toolkit, cuDNN operates at a
    higher level by implementing neural network functionalities. It streamlines complex
    neural network operations, making it more accessible for high-level users.

In the scenario of updating the NVIDIA Driver, CUDA Toolkit, and cuDNN, my preference is
to remove everything that has already been installed and initiate the process from a
clean state.

Here are the links that might be helpful:

1. [Official guide to remove CUDA Toolkit and driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-toolkit-and-driver)
2. [Stack Overflow question on uninstalling NVIDIA Driver and CUDA Toolkit](https://stackoverflow.com/questions/56431461/how-to-remove-cuda-completely-from-ubuntu)
3. [CUDA install discussion over NVIDIA forum](https://forums.developer.nvidia.com/t/cuda-install-unmet-dependencies-cuda-depends-cuda-10-0-10-0-130-but-it-is-not-going-to-be-installed/66488/6)

To remove CUDA Toolkit:

```shell
sudo apt --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
sudo /usr/local/cuda-12.1/bin/cuda-uninstaller
```

To remove NVIDIA Drivers:

```shell
sudo /usr/bin/nvidia-uninstall
sudo apt clean; sudo apt update; sudo apt purge cuda; sudo apt purge nvidia-*; sudo apt autoremove;
sudo apt --purge remove "*nvidia*" "libxnvctrl*"
```

To clean up the uninstall:

```
sudo apt autoremove
```

To remove libraries in `/usr/local/cuda*`:

```
sudo rm -r /usr/local/cuda*
```

## Installing NVIDIA Driver

Interestingly, it might seem like manually installing the NVIDIA Driver isn't always
necessary. For example, when using the runfile (local) method to install the CUDA
Toolkit, a specific version of the NVIDIA Driver, bundled with the runfile, gets
installed automatically.

However, there are scenarios where manual installation of the NVIDIA Driver becomes
essential. This is particularly true for newer NVIDIA GPU models, which may require the
latest driver versions not included in the older CUDA Toolkit. For instance, the CUDA
Toolkit version `cuda_12.1.1_530.30.02_linux.run` includes the NVIDIA Driver
`530.30.02`. This version can be too outdated for newer GPU models.

In such cases, it's necessary to install a specific version of the NVIDIA Driver
independently from the CUDA Toolkit. In this section, I'll guide you through the process
of installing the NVIDIA Driver on its own, ensuring compatibility with the latest GPU
models. This guide is intended to help future users, including my future self, navigate
this setup.

### Installing NVIDIA Driver on Virtual Machines of Google Compute Engine (GCE)

This section demonstrates how to install the NVIDIA driver on a clean Ubuntu virtual
machine (Ubuntu 22.04) hosted on GCE. Please carefully follow the steps below:

-   **Step 1:** Begin by thoroughly reading
    [the official installation guide provided by GCE](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu).
    This guide covers crucial details essential for a successful installation, such as
    [determining the minimum supported version of NVIDIA drivers for each GPU type](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#minimum-driver)
    and
    [using the installation script to install the NVIDIA driver](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#install-script).
    It's important to note that although
    [this section](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#no-secure-boot)
    suggests that the NVIDIA driver can be installed alongside the CUDA Toolkit, I
    recommend installing the driver separately to ensure compatibility, especially when
    the pre-packaged driver version in the CUDA Toolkit might not meet the specific
    requirements of your hardware.

    After checking out Step 1, you should have pinned down the right version of the
    NVIDIA driver for your hardware. If you haven't, take another crack at it—sometimes
    it just takes a bit more digging to land on the correct driver version. Moving
    forward, we'll be using the version `535.104.05` and following the approach laid out
    in
    [this installation script provided by GCE](https://github.com/GoogleCloudPlatform/compute-gpu-installation/blob/main/linux/install_gpu_driver.py).

-   **Step 2:** Start by updating and upgrading your system packages.

    ```shell
    sudo apt update && sudo apt upgrade -y
    ```

    Next, install the kernel headers and development packages for the currently running
    kernel:

    ```shell
    sudo apt install linux-headers-$(uname -r)
    ```

-   **Step 3:** Install required system packages.

    ```shell
    sudo apt install -y software-properties-common \
    pciutils \
    make \
    dkms \
    gcc-12 \
    g++-12
    ```

    A crucial step here is to set `gcc-12` and `g++-12` as the default compilers. This
    ensures they are used when installing the NVIDIA driver.

    ```shell
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    --slave /usr/bin/g++               g++ /usr/bin/g++-12 \
    --slave /usr/bin/gcc-ar         gcc-ar /usr/bin/gcc-ar-12 \
    --slave /usr/bin/gcc-nm         gcc-nm /usr/bin/gcc-nm-12 \
    --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-12 \
    --slave /usr/bin/gcov             gcov /usr/bin/gcov-12 \
    --slave /usr/bin/gcov-dump   gcov-dump /usr/bin/gcov-dump-12 \
    --slave /usr/bin/gcov-tool   gcov-tool /usr/bin/gcov-tool-12
    ```

    To learn more about changing the default GCC/G++ compiler in Ubuntu, you might want
    to read through
    [this Stack Overflow question](https://stackoverflow.com/questions/7832892/how-to-change-the-default-gcc-compiler-in-ubuntu).

-   **Step 4:** Download and install the NVIDIA driver.

    ```shell
    curl -fSsl -O https://us.download.nvidia.com/tesla/535.104.05/NVIDIA-Linux-x86_64-535.104.05.run
    sudo sh NVIDIA-Linux-x86_64-535.104.05.run -s --dkms --no-cc-version-check
    ```

    If you encounter any errors or failures during installation, please refer to this
    [installation function
    (`compute-gpu-installation/linux/install_gpu_driver.py:install_driver_runfile`)](https://github.com/GoogleCloudPlatform/compute-gpu-installation/blob/7d1f09e414be69ece62a8024d42eba7cf90752f5/linux/install_gpu_driver.py#L365).
    You may find salvation in it.

    Once the NVIDIA driver is successfully installed, the next step might
    involve installing the CUDA Toolkit. However, it may not be necessary to
    install CUDA at the system level. Many modern frameworks, such as PyTorch,
    bundle the required CUDA libraries within their installation packages. I
    recommend first attempting to install PyTorch in a virtual environment and
    verifying GPU detection. If PyTorch can recognize your GPUs, a separate
    system-wide CUDA Toolkit installation is likely unnecessary.

## Installing CUDA Toolkit

Here are the links that might be helpful for installing CUDA Toolkit:

1. [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#)
2. [StackOverflow question on installing multiple versions of CUDA](https://stackoverflow.com/questions/75160448/install-multiple-version-of-cuda)

Now, I aim to update the CUDA Toolkit to version 12.1. The reason behind this decision
is that I discovered the official PyTorch 2.1 release seems to exclusively support CUDA
versions 12.1 (and not above?).

After removing the previous version, I will adhere to the
[NVIDIA official CUDA Toolkit installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
While this guide recommends installing CUDA Toolkit 12.3, I specifically intend to
install version 12.1. To achieve this, I will navigate to the
[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and select the
version of `CUDA Toolkit 12.1`. Following the selection of CPU architecture, OS, and
distribution, I will opt for the `runfile (local)` installation method.

**Notice:** Please exercise caution with this method when installing the CUDA Toolkit.
_If you have already installed an NVIDIA driver as described in the previous section,
**DO NOT** select the option to install the NVIDIA driver that is bundled with the CUDA
Toolkit._ This could lead to conflicts with the driver already installed on your system.

```shell
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```

**Notice:** Exercise **extreme** caution when using `sudo apt-get install cuda` or
`sudo apt-get install cuda-12-1` for installation. I have encountered instances where
the installed CUDA version did not match my intended version, even when following the
exact command provided by the guide.

After installation, it prompts the following message:

```
===========
= Summary =
===========

Driver:   Installed
Toolkit:  Installed in /usr/local/cuda-12.1/

Please make sure that
 -   PATH includes /usr/local/cuda-12.1/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-12.1/lib64, or, add /usr/local/cuda-12.1/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-12.1/bin
To uninstall the NVIDIA Driver, run nvidia-uninstall
Logfile is /var/log/cuda-installer.log
```

I can verify the installation by running:

```
$ nvidia-smi
Sun Jan 21 09:22:21 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.30.02              Driver Version: 530.30.02    CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA RTX A3000 Laptop GPU     Off| 00000000:01:00.0 Off |                  N/A |
| N/A   49C    P0               19W /  60W|      0MiB /  6144MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

We can observe that NVIDIA Driver `530.30.02` and CUDA Toolkit `12.1` are installed.
Also, let's check the location of the installed NVIDIA Driver and CUDA Toolkit.

```
$ ls /usr/bin/nvidia-*
/usr/bin/nvidia-bug-report.sh     /usr/bin/nvidia-debugdump  /usr/bin/nvidia-modprobe      /usr/bin/nvidia-powerd    /usr/bin/nvidia-smi
/usr/bin/nvidia-cuda-mps-control  /usr/bin/nvidia-detector   /usr/bin/nvidia-ngx-updater   /usr/bin/nvidia-settings  /usr/bin/nvidia-uninstall
/usr/bin/nvidia-cuda-mps-server   /usr/bin/nvidia-installer  /usr/bin/nvidia-persistenced  /usr/bin/nvidia-sleep.sh  /usr/bin/nvidia-xconfig

$ ls /usr/local/cuda*
/usr/local/cuda:
bin                DOCS      extras  gds-12.1  lib64    nsight-compute-2023.1.1  nsight-systems-2023.1.2  nvvm    share  targets  version.json
compute-sanitizer  EULA.txt  gds     include   libnvvp  nsightee_plugins         nvml                     README  src    tools

/usr/local/cuda-12.1:
bin                DOCS      extras  gds-12.1  lib64    nsight-compute-2023.1.1  nsight-systems-2023.1.2  nvvm    share  targets  version.json
compute-sanitizer  EULA.txt  gds     include   libnvvp  nsightee_plugins         nvml                     README  src    tools
```

**Post-installation Actions:**

Here are the links that can be helpful for post-installation actions:

1. [Post-installation instructions from the official guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)
2. [A blog post of installing NVIDIA Driver and CUDA Toolkit](https://shuuutin-cg.medium.com/ubuntu18-04%E5%AE%89%E8%A3%9Dcuda%E4%BB%A5%E5%8F%8A%E6%94%B9%E8%AE%8Acuda%E7%89%88%E6%9C%AC-b8ac917f880f)

To update `PATH` and `LD_LIBRARY_PATH`, adding the following two lines in `.bashrc`:

```shell
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

and source `.bashrc`, `source .bashrc`.

**Note:** I don't know what the following two commands do. I'm adding them here as a
reminder.

```shell
export CPATH=/usr/local/cuda/include:$CPATH
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH
```

Now, let's verify it.

```shell
$ which nvcc
/usr/local/cuda-12.1/bin/nvcc

$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

It appears that `nvcc -V` and `nvidia-smi` might return two different versions of the
CUDA Toolkit. To gain a better understanding of these two commands, please refer to the
following links:

-   [StackOverflow question on Different CUDA versions shown by nvcc and NVIDIA-smi](https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi)
-   [The detected CUDA version mismatches the version that was used to compile PyTorch](https://github.com/vllm-project/vllm/issues/1548)

**Note:** Some guides use the following two commands to install the CUDA Toolkit, but it
seems unnecessary to me since I successfully installed the NVIDIA driver and CUDA
Toolkit without them.

```shell
# I think the following two commands are not needed.
sudo apt-get install cuda-toolkit
sudo apt-get install nvidia-cuda-toolkit
```


## Installing Multiple Versions of the CUDA Toolkit

This section walks through installing and switching between multiple versions.
The key strategy is to install each version in its unique system path and
configure your environment variables to use the desired version during
development.

Suppose your system already has CUDA Toolkit 12.1 installed, with the following
setup:

```
/usr/local/cuda-12.1/
/usr/local/cuda -> /usr/local/cuda-12.1/
```

Now, you want to install CUDA Toolkit 11.8 alongside it.

1. Download CUDA Toolkit 11.8:

    `wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run`

2. Run the installer but ensure that **ONLY** the **CUDA Toolkit** is installed (**NOT** the **NVIDIA Driver**). Specify a different path to avoid overwriting the existing installation:

    `sudo sh cuda_11.8.0_520.61.05_linux.run --toolkitpath=/usr/local/cuda-11.8`

    Note: During installation, you might encounter errors related to
    incompatible versions of `gcc` and `g++`. If so, install the correct
    versions and prioritize them.

3. Update the Symlink

    To switch between CUDA versions, update the symlink by pointing
    `/usr/local/cuda` to the desired version:

    `sudo ln -sfn /usr/local/cuda-11.8 /usr/local/cuda`

4. Configure Environment Variables

    Whenever you need to work with CUDA 11.8, set your environment variables
    accordingly:

    ```
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
    ```

References:
- [Stack Overflow: Install multiple versions of CUDA and cuDNN](https://stackoverflow.com/questions/41330798/install-multiple-versions-of-cuda-and-cudnn)
- [Multiple versions of CUDA libraries on the same machine](https://blog.kovalevskyi.com/multiple-version-of-cuda-libraries-on-the-same-machine-b9502d50ae77)


## Installing cuDNN

Here are the links that can be helpful for installing cuDNN:

-   [Official NVIDIA cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
-   [StackOverflow question on How to verify CuDNN installation?](https://stackoverflow.com/questions/31326015/how-to-verify-cudnn-installation)

First, I log into the [NVIDIA cuDNN website](https://developer.nvidia.com/cudnn) and
select the version compatible with my setup. I opt for
`Download cuDNN v8.9.7 (December 5th, 2023), for CUDA 12.x`, and choose the version for
Ubuntu, specifically `Local Installer for Linux x86_64 (Tar)`. I prefer the tar
installation method due to its simplicity – it involves unzipping the file and copying
the files into the local directory.

Follow the steps below to install cuDNN.

-   Unzip the cuDNN package.

```shell
$ cd Downloads/
$ tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
```

-   Copy the uncompressed files into the CUDA toolkit directory. There are two CUDA
    folders after installing CUDA Toolkit, `/usr/local/cuda` and `/usr/local-12.1/cuda`,
    on my laptop. The former is symbolically linked to the latter, as shown by the
    following command:
    ```
    $ ll /usr/local/cuda
    lrwxrwxrwx 1 root root 21 Jan 29 06:32 /usr/local/cuda -> /usr/local/cuda-12.1//
    ```

```shell
sudo cp cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/cudnn*.h /usr/local/cuda-12.1/include
sudo cp -P cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/libcudnn* /usr/local/cuda-12.1/lib64
sudo chmod a+r /usr/local/cuda-12.1/include/cudnn*.h /usr/local/cuda-12.1/lib64/libcudnn*
```

-   Verify the installation by following
    [this StackOverflow question](https://stackoverflow.com/questions/31326015/how-to-verify-cudnn-installation).

```
$ cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
#define CUDNN_MAJOR 8
#define CUDNN_MINOR 9
#define CUDNN_PATCHLEVEL 7
```

We can confirm that `cuDNN v8.9.7` has been successfully installed.

# Install PyTorch in `venv` Virtual Environment

I do not utilize `conda`; if necessary, I would prefer to employ `conda` within a Docker
container environment rather than installing it on my system.

-   Create and activate a Python `venv` virtual environment:

```
python3.10 -m venv venv_gpu && source venv_gpu/bin/activate
```

-   Update `pip` and `setuptools`:

```
python3 -m pip install --upgrade pip setuptools
```

-   Install `Pytorch 2.*` by following the guide in
    [the official guide](https://pytorch.org/get-started/locally/).

    I choose `stable(2.1.2)` and `CUDA 12.1`.

```
python3 -m pip install torch torchvision torchaudio
```

-   Verify the installation

```
$ python
Python 3.10.10 (main, Mar 29 2023, 06:08:28) [GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> x = torch.rand(5, 3)
>>> print(x)
tensor([[0.5622, 0.9026, 0.5692],
        [0.4352, 0.5099, 0.2568],
        [0.4739, 0.6036, 0.8984],
        [0.0356, 0.3390, 0.2689],
        [0.6139, 0.9721, 0.4264]])
>>> torch.cuda.is_available()
True
```

**Install TensorFlow in `venv` Virtual Environment:** The process of installing
TensorFlow is similar to PyTorch's. Here are the step-by-step instructions.

-   Create, activate a Python `venv` virtual environment and update `pip` and
    `setuptools`:

```
python3.10 -m venv venv_gpu && source venv_gpu/bin/activate && python3 -m pip install --upgrade pip setuptools
```

-   Install latest stable version of TensorFlow by following
    [the official guide](https://www.tensorflow.org/install/pip).

```
# For GPU users
python3 -m pip install tensorflow[and-cuda]
# For CPU users
python3 -m pip install tensorflow
```

## To Export or Not To Export `LD_LIBRARY_PATH`

After installing and attempting to run a PyTorch program, I encountered the following
error:

```
Could not load library libcudnn_cnn_train.so.8. Error: /usr/local/cuda-12.1/lib64/libcudnn_cnn_train.so.8: undefined symbol: _ZN5cudnn3cnn34layerNormFwd_execute_internal_implERKNS_7backend11VariantPackEP11CUstream_stRNS0_18LayerNormFwdParamsERKNS1_20NormForwardOperationEmb, version libcudnn_cnn_infer.so.8
```

The strange thing is that this issue occurred after I followed the instructions in the
official guide, which include exporting `PATH` and `LD_LIBRARY_PATH` in `.bashrc`.

Upon further investigation, I found that the latest PyTorch version installs the cuDNN
library by default. If I set `LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64/` in the system
path, my PyTorch program does not search for the cuDNN path in its virtual environment,
specifically `./venv_gpu/lib/python3.10/site-packages/nvidia/cudnn/lib/`. While I'm not
sure why it doesn't search the local path, there are two solutions to resolve this:

-   Add `./venv_gpu/lib/python3.10/site-packages/nvidia/cudnn/lib/` to
    `LD_LIBRARY_PATH`.
-   Do not export `LD_LIBRARY_PATH` in `.bashrc`.

I choose the latter option, not exporting `LD_LIBRARY_PATH` in `.bashrc`, as it is
simpler. This allows each Python virtual environment to install different cuDNN
libraries while using PyTorch.

**References:**

-   https://github.com/pytorch/pytorch/issues/96595
-   https://stackoverflow.com/questions/70340812/how-to-install-pytorch-with-cuda-support-with-pip-in-visual-studio
-   https://stackoverflow.com/questions/70340812/how-to-install-pytorch-with-cuda-support-with-pip-in-visual-studio
-   https://github.com/pytorch/pytorch/issues/104591
-   https://blog.csdn.net/wangmou211/article/details/134595135
-   https://discuss.pytorch.org/t/could-not-load-library-libcudnn-cnn-train-so-8-while-training-convnet/171334/3
-   https://blog.csdn.net/qq_37700257/article/details/134312228
-   https://stackoverflow.com/questions/70340812/how-to-install-pytorch-with-cuda-support-with-pip-in-visual-studio

## Following the Exact Steps to Install PyTorch3D

After installing the NVIDIA driver, CUDA Toolkit, cuDNN, creating a `venv` virtual
environment, and installing PyTorch `2.1.*`, I can finally proceed to install
[PyTorch3D](https://github.com/facebookresearch/pytorch3d/tree/main).

**Note:**
[PyTorch3D's official guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
recommends using `conda` for installation, but I prefer not to install `conda` on my
laptop. Thus, I attempted to build/install from source within the venv virtual
environment.

**[NVIDIA Python wheels](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pip-wheels):**
NVIDIA provides Python Wheels for installing CUDA through pip, primarily for using CUDA
with Python. These packages are intended for runtime use and do not currently include
developer tools (these can be installed separately).

-   Install CUDA wheels:

```
python3 -m pip install --upgrade setuptools pip wheel
python3 -m pip install nvidia-pyindex
python3 -m pip install -r nvidia_requirements.txt
```

-   The content of `nvidia_requirements.txt` is:

```
--extra-index-url https://pypi.org/simple

nvidia-cuda-runtime-cu12
nvidia-cuda-cupti-cu12
nvidia-cuda-nvcc-cu12
nvidia-nvml-dev-cu12
nvidia-cuda-nvrtc-cu12
nvidia-nvtx-cu12
nvidia-cuda-sanitizer-api-cu12
nvidia-cublas-cu12
nvidia-cufft-cu12
nvidia-curand-cu12
nvidia-cusolver-cu12
nvidia-cusparse-cu12
nvidia-npp-cu12
nvidia-nvjpeg-cu12
nvidia-nvjitlink-cu12
nvidia-cuda-opencl-cu12
```

-   Install PyTorch3D

```
python3 -m pip install fvcore iopath
python3 -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

Building wheels for collected packages: pytorch3d
  Building wheel for pytorch3d (setup.py) ... done
  Created wheel for pytorch3d: filename=pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl size=69948490 sha256=68166755d9b2cc8643ec045b7e12f9b3809573c3b9770fb3180aaabb7d6bcd7c
  Stored in directory: /tmp/pip-ephem-wheel-cache-m_1j5r61/wheels/39/5f/20/2d3b6f3a35a60bdc0ba3c19da94340db9596637d1d1222473d
Successfully built pytorch3d
Installing collected packages: pytorch3d
Successfully installed pytorch3d-0.7.5
```

**Note:** It's extremely crucial that the NVIDIA driver and CUDA Toolkit versions align
with PyTorch 2.1 compatibility. Additionally, PyTorch3D supports specific versions of
PyTorch (not above PyTorch 2.1), as documented in the
[pytorch/install.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
file. Essentially, all versions of every software must be compatible with each other;
_otherwise, it won't work._

# Running GPU Programs within a Docker Container

-   **Step 1:** First, read through the
    [NVIDIA Container Toolkit GitHub repository](https://github.com/NVIDIA/nvidia-container-toolkit). 
    Then, follow the detailed instructions on the
    [NVIDIA Container Toolkit documentation page](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
    to install the NVIDIA Container Toolkit. This toolkit enables Docker containers to 
    access the NVIDIA GPU on your host machine.

-   **Step 2:** After installing the NVIDIA Container Toolkit, verify the installation 
    by running the following command:

    ```
    ~$ sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
    Unable to find image 'ubuntu:latest' locally
    latest: Pulling from library/ubuntu
    49b384cc7b4a: Pull complete 
    Digest: sha256:3f85b7caad41a95462cf5b787d8a04604c8262cdcdf9a472b8c52ef83375fe15
    Status: Downloaded newer image for ubuntu:latest
    Sun Jun  2 08:18:02 2024       
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 535.171.04             Driver Version: 535.171.04   CUDA Version: 12.2     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  NVIDIA RTX A3000 Laptop GPU    Off | 00000000:01:00.0 Off |                  N/A |
    | N/A   44C    P0              19W /  60W |      8MiB /  6144MiB |      0%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
                                                                                            
    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    +---------------------------------------------------------------------------------------+
    ```

# Frequently Asked Questions

-   Often, you may encounter the following error when running `nvidia-smi`:
    ```
    $ nvidia-smi
    NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
    ```
    
    This issue can arise for various reasons. After researching and trying different
    solutions, I found a blog
    [post](https://medium.com/@yt.chen/nvidia-smi-%E9%80%A3%E4%B8%8D%E5%88%B0-driver-%E7%9A%84%E8%87%AA%E6%95%91%E6%96%B9%E6%B3%95-69cbed16171d)
    that provides a viable solution. Here is asummarized version of the steps:


    - 1. Run the following command to find the previously installed NVIDIA driver version:

    ```shell
    $ ls /usr/src | grep nvidia
    nvidia-535.171.04
    ```

    - 2. Install `dkms` (if not already installed):

    ```shell
    sudo apt install dkms
    ```

    - 3. Use `dkms` to install the detected driver version:

    ```
    sudo dkms install -m nvidia -v 535.171.04
    ```
