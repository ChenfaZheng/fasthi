# 在Linux设备上安装Anaconda3

## 下载安装包

可以在[下载页面](https://www.anaconda.com/products/distribution#Downloads)选择对应版本（通常为64-Bit (x86) Installer）进行下载，也可以直接使用`wget`下载到本地：
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
```
如果网络受限，可以从国内镜像站下载：
```bash
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.11-Linux-x86_64.sh
```

## 安装Anaconda3

给文件添加执行权限：
```bash
chmod +x Anaconda3-2021.11-Linux-x86_64.sh
```

执行安装文件：
```bash
./Anaconda3-2021.11-Linux-x86_64.sh
```
程序提示阅读license，按`Enter`继续：
```
Welcome to Anaconda3 2021.11

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>> 
```
阅读完毕后，按`q`退出，程序询问是否同意license，我们输入`yes`并`Enter`
```
...
Do you accept the license terms? [yes|no]
[no] >>> yes
```
程序询问安装地址，我们这里使用默认地址，直接`Enter`
```
...
Anaconda3 will now be installed into this location:
/home/someuser/anaconda3

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home/someuser/anaconda3] >>> 
```
程序此时会自动开始安装。等待片刻后，程序提示安装完成，并询问是否初始化anaconda，我们输入`yes`，并`Enter`
```
...
done
installation finished.
Do you wish the installer to initialize Anaconda3
by running conda init? [yes|no]
[no] >>> yes
```
初始化成功，提示需要重启shell。我们可以关闭当前的命令窗口。
```
...
==> For changes to take effect, close and re-open your current shell. <==

If you'd prefer that conda's base environment not be activated on startup, 
   set the auto_activate_base parameter to false: 

conda config --set auto_activate_base false

Thank you for installing Anaconda3!
```

## 设置不自动启动base虚拟环境

为了避免无意间将虚拟环境与系统环境冲突，我们设置conda不自动启动base环境。新开一个shell，执行：
```bash
conda config --set auto_activate_base false
```

## 一些常用命令

- 启动一个虚拟环境
    ```
    conda activate environment_name
    ```
- 退出一个虚拟环境
    ```
    conda deactivate
    ```
- 查看已有的虚拟环境
    ```
    conda env list
    ```
- 创建指定python版本的虚拟环境
    ```
    conda create -n environment_name python=3.6
    ```
- 在当前所在虚拟环境中安装包
    ```
    conda install numpy
    ```
- 在当前所处虚拟环境中安装社区包/使用国内源
    ```
    conda install -c conda-forge emcee
    conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ numpy
    conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ emcee
    ```