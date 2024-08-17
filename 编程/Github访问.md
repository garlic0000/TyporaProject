# Github访问

## 1.加速github访问

每次访问github都会卡死。

1.github镜像站：<https://www.sockstack.cn/github>

- 登录个人账户可能有影响（猜测）  
- 不稳定（镜像站不知道什么时候就没了）

2.下载篡改猴浏览器插件Tampermonkey.

- 使用chrome的浏览器下载脚本无法立刻安装，需要额外配置。  
- 使用firefox浏览器可以直接安装脚本。

3.使用Greasy Fork安装脚本

- [Github 增强 - 高速下载](https://greasyfork.org/zh-CN/scripts/412245-github-%E5%A2%9E%E5%BC%BA-%E9%AB%98%E9%80%9F%E4%B8%8B%E8%BD%BD)

4.去github网站下载stream++ 2.8.6版本

- 用于访问加速，登录个人账户  
- 先用镜像站获取下载链接  
- 使用脚本下载（非常快，下载速度2Mb/s以上）

5.组合起来就可以快乐使用github了

## 2.访问github网站建立安全连接失败

主要问题:  
  打开steam++，点击开始加速，在浏览器地址中输入github网址后，浏览器访问出错并出现下列提示：  
> 建立安全连接失败  
  连接到 github.com 时发生错误。  
  对等端的证书有一个无效的签名。  
  错误代码：SEC_ERROR_BAD_SIGNATURE  

解决方法（有用）

> 参考：`https://www.jishusongshu.com/computer-software/fastgithub/`
> 1 地址栏输入：about:config  
> 2 输入首选项名称：security.enterprise_roots.enabled  
> 3 修改值为：true（本来为true，先改为false，再改为true）
