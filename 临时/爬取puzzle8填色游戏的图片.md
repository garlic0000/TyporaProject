# 爬取puzzle8填色游戏的图片

某天看到别人的画，我真的非常震惊，于是开始学画画 …… 然而没有毅力，于是退而求其次开始填色，偶然间发现了一个免费的填色游戏网站，而且其中的图片可以免费下载，让我惊喜的是这些图片都很清晰，每个页面有24张图片，总共有17页（除了最后一页是8张外），最初我是一张一张的下载，后来嫌太麻烦了，于是开始用爬虫。
对于其中的一张图片，我的目的是获取图片的名称并将其下载到指定位置。
    l 获取图片名称
    当点击这张图片时，发现图片的名称会出现在网页的<title>中，于是读取<title></title>中的内容并将其中的图片名字读取出来即可。
        a. 使用lxml中的etree对网页进行解析，使用xpath获取<title>的内容。
        b. 由于<title>中的内容不止包括图片名称，因此需要进行模式匹配，使用re库中的match
    l 下载图片
    每张图片的内容是<img>元素中src的值，因此请求src，并将其写入图片文件中即可
    a. 这个src不是一般的以jpg结尾的链接，而是图片本身，即data:image/png……这样的链接
    b. 
