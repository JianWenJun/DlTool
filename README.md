


### 停止命令
ps -ef|grep python|grep -v grep|awk '{print $2}'|xargs sudo -E kill -9