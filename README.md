


### 停止命令
ps -ef|grep python|grep -v grep|awk '{print $2}'|xargs sudo -E kill -9


### wandb使用
https://zhuanlan.zhihu.com/p/493093033