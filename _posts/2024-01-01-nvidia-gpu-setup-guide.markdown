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

Today, I attempted to install PyTorch3D within my venv virtual environment but
encountered difficulties despite having successfully installed PyTorch, the NVIDIA
driver, CUDA Toolkit, and cuDNN using pip.

PyTorch3D recommends using conda for installation, but I prefer not to install conda on
my laptop. Consequently, my sole recourse is to leverage Docker to create an independent
system environment, run a Docker container, and install PyTorch3D with conda within it.
However, I discovered that PyTorch 2.\* mandates CUDA Toolkit version 12.1, while my
laptop currently hosts CUDA Toolkit version 12.0. As a result, I aim to meticulously
document the entire process of installing and updating the NVIDIA driver, CUDA Toolkit,
and cuDNN during the development workflow.

The objectives of this post are as follows:

-   Document the step-by-step process of installing and updating the NVIDIA driver, CUDA
    Toolkit, and cuDNN.
-   Record the procedure for creating a venv virtual environment, installing PyTorch,
    successfully running PyTorch utilities with GPU support, and installing pytorch3d
    within the venv virtual environment.
-   In the event of unsuccessful installation of pytorch3d within the venv virtual
    environment, create a Docker image to establish an environment capable of running
    pytorch3d. This environment should be compatible with the following three scenarios:
    -   Jupyter Notebook
    -   Command-line shell
    -   Visual Studio Code, ensuring it recognizes the environment.

I use Ubuntu 20.04, and my GPU is the `NVIDIA RTX A3000 Laptop GPU`, equipped with 4096
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

Here are two links that might be helpful:

1. [Official guide to remove CUDA Toolkit and driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-toolkit-and-driver)
2. [Stack Overflow question on uninstalling NVIDIA Driver and CUDA Toolkit](https://stackoverflow.com/questions/56431461/how-to-remove-cuda-completely-from-ubuntu)
3. [CUDA install discussion over NVIDIA forum](https://forums.developer.nvidia.com/t/cuda-install-unmet-dependencies-cuda-depends-cuda-10-0-10-0-130-but-it-is-not-going-to-be-installed/66488/6)

To remove CUDA Toolkit:

```shell
sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
sudo /usr/local/cuda-12.1/bin/cuda-uninstaller
```

To remove NVIDIA Drivers:

```shell
sudo /usr/bin/nvidia-uninstall
sudo apt clean; sudo apt update; sudo apt purge cuda; sudo apt purge nvidia-*; sudo apt autoremove;
sudo apt-get --purge remove "*nvidia*" "libxnvctrl*"
```

To clean up the uninstall:

```
sudo apt-get autoremove
```

To remove libraries in `/usr/local/cuda*`:

```
rm -r /usr/local/cuda*
```

## Installing NVIDIA Driver

Interestingly, it appears to me that manual installation of the NVIDIA Driver may not be
necessary. When I utilized the runfile (local) method to install the CUDA Toolkit, the
NVIDIA Driver was automatically installed along the way.

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

<!-- it seems i need to install the following two packages.

```
sudo apt-get install cuda-toolkit # probably do not do this.
sudo apt-get install nvidia-cuda-toolkit #  this is for nvcc -V. probably do not do this.
``` -->

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

## Installing cuDNN

Here are the links that can be helpful for post-installation actions:

-   [Official NVIDIA cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
-   [StackOverflow question on How to verify CuDNN installation?](https://stackoverflow.com/questions/31326015/how-to-verify-cudnn-installation)

First, I log into the [NVIDIA cuDNN website](https://developer.nvidia.com/cudnn) and
select the version compatible with my setup. I opt for
`Download cuDNN v8.9.7 (December 5th, 2023), for CUDA 12.x`, and choose the version for
Ubuntu, specifically `Local Installer for Linux x86_64 (Tar)`. I prefer the tar
installation method due to its simplicity – it involves unzipping the file and copying
the files into the local directory.

Follow the steps below to install cuDNN.

1. Unzip the cuDNN package.

```shell
$ cd Downloads/
$ tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
```

2. Copy the following files into the CUDA toolkit directory.

```shell
sudo cp cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/cudnn*.h /usr/local/cuda-12.1/include
sudo cp -P cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/libcudnn* /usr/local/cuda/lib64/
sudo cp -P cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/libcudnn* /usr/local/cuda-12.1/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
sudo chmod a+r /usr/local/cuda-12.1/include/cudnn*.h /usr/local/cuda-12.1/lib64/libcudnn*
```

There are two CUDA folders after installing CUDA Toolkit, `usr/local/cuda` and
`usr/local-12.1/cuda`, on my laptop. I don't really know why. Thus, I copy every files
to both of the folders.

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

## Following the Exact Steps to Install PyTorch3D

After installing the NVIDIA driver, CUDA Toolkit, cuDNN, creating a `venv` virtual
environment, and installing PyTorch `2.1.*`, I can finally proceed to install
[PyTorch3D](https://github.com/facebookresearch/pytorch3d/tree/main).

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

<!-- # download and build pytorch dockerimage.

https://hub.docker.com/r/pytorch/pytorch/tags?page=1

Reference: CUDA, cuDNN and Nvidia Driver.
https://medium.com/ibm-data-ai/straight-forward-way-to-update-cuda-cudnn-and-nvidia-driver-and-cudnn-80118add9e53 -->
