# 安装ubuntu

## 下载

- VMware-Workstation-Pro-12.5.6-5528349-精简绿色中文版_by_wxdjs.7z
- ubuntu-22.04.2-live-server-amd64.iso

## 安装 VMware

1. 解压 `VMware-Workstation-Pro-12.5.6-5528349-精简绿色中文版_by_wxdjs.7z` 到任意目录
2. 使用管理员权限运行 `VMware` 目录下的安装脚本 `!)安装VMware.cmd`
   ![](md-img/2023-03-21-11-53-41.png)
3. 输入 `1` 选择网络功能，然后回车
   ![](md-img/2023-03-21-11-59-24.png)
4. 输入 `1` 选择桥接，然后回车
   ![](md-img/2023-03-21-12-00-04.png)
5. 输入 `2` 启用服务，然后回车
   ![](md-img/2023-03-21-12-00-51.png)
6. 重启电脑

安装完成

## 安装 Ubuntu

流程中没有提及的部分直接点击 `下一步`

1. 点击新建虚拟机
   ![](md-img/2023-03-21-12-02-11.png)
2. 选择文件 `ubuntu-22.04.2-live-server-amd64.iso`
   ![](md-img/2023-03-21-12-02-44.png)
3. 点击自定义硬件
   ![](md-img/2023-03-21-12-04-10.png)
4. 分配内存、cpu、并将网卡设置成桥接
   ![](md-img/2023-03-21-12-05-43.png)

启动虚拟机后继续安装

> 注意，选择网络设置时，可以参考手动配置:
> ![](md-img/安装ubuntu_2023-05-08_17-41-17.png)

1. 选择第一个开始安装
   ![](md-img/2023-03-21-12-17-15.png)
2. 选择挂载硬盘，此步需要手动选择 `done`
   ![](md-img/2023-03-21-12-18-48.png)
3. 安装前的确认，选择 `continue`
   ![](md-img/2023-03-21-12-19-38.png)
4. 设置计算机基本信息，包括创建账号
   ![](md-img/2023-03-21-12-21-03.png)
5. 勾选安装 `openssh-server`
   ![](md-img/2023-03-21-12-22-36.png)
6. 手动选择 `done`
   ![](md-img/2023-03-21-12-23-20.png)
7. 安装好后选择 `reboot` 重启
   ![](md-img/2023-03-21-12-27-19.png)

安装好后，建议制作快照

![](md-img/2023-03-21-12-29-52.png)

## 配置网络

若先前安装 ubuntu 时已经配置过网络了，此处可以跳过

```shell
sudo nano /etc/netplan/00-installer-config.yaml
```

```yaml
# 参考
network:
  ethernets:
    ens33:
      # ip 地址
      addresses:
      - 192.168.38.100/24
      nameservers:
      # dns 地址
        addresses:
        - 114.114.114.114
        - 114.114.115.115
        - 8.8.8.8
        search: []
      # 网关
      routes:
      - to: default
        via: 192.168.38.2
  version: 2
```

修改完配置后，需要应用配置

```shell
sudo netplan apply
```


## 更新系统

```shell
# 更新注册表
sudo apt-get update
# 更新软件
sudo apt-get upgrade
```

## 配置hosts

```shell
sudo vim /etc/hosts
```

将自己主机和未来的两台节点机的 ip 都设置好，例如:

```
192.168.38.100 master
192.168.38.101 slave1
192.168.38.102 slave2
```