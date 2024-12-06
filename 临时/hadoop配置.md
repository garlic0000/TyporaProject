# hadoop配置

https://mirror.tuna.tsinghua.edu.cn/
https://mirror.tuna.tsinghua.edu.cn/apache/hadoop/common/hadoop-3.4.0/

**pdsh rcmd: socket: Permission denied问题解决办法**

https://blog.csdn.net/Flamewaker/article/details/116518583
https://www.vvave.net/archives/run-pdsh-script-error-rcmd-socket-permission-denied.html

https://zhuanlan.zhihu.com/p/559644885

**单节点配置**

https://cn.linux-console.net/?p=3612

**三个节点配置静态IP并互ping**

https://blog.csdn.net/wurobb/article/details/134281489

**xsync同步配置**

https://blog.csdn.net/qq_43533638/article/details/129530155

注意名称 slaver1 slaver2

**配置历史服务器**

https://blog.csdn.net/m0_46413065/article/details/116305931

**Permission denied**

检查pdsh default rcmd rsh

pdsh -q -w localhost

将pdsh的默认rcmd修改为ssh

export PDSH_RCMD_TYPE=ssh
source ~/.bashrc

执行

echo "ssh" | sudo tee /etc/pdsh/rcmd_default

**使用物理机访问虚拟机**

https://www.cnblogs.com/unirithe/p/15558493.html

**hadoop虚拟机可以访问9870，但是本机浏览器不可以访问**

https://blog.csdn.net/weixin_45419744/article/details/106945101 关闭防火墙
https://blog.csdn.net/jarvan5/article/details/114390145 永久关闭防火墙

**启动NameNode和DataNode**

start-dfs.sh

master:9870

**停止**

stop-dfs.sh

**启动历史服务器**

mapred --daemon start historyserver

**停止**

mapred --daemon stop historyserver

**在Yarn上运行MapReduce**

start-yarn.sh

slaver1:8088

**停止**

stop-yarn.sh



192.168.72.131 master
192.168.72.132 slaver1 
192.168.72.133 slaver2 