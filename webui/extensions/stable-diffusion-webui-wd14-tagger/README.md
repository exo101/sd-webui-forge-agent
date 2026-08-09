# Tagger for [Automatic1111&#39;s WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)、

注意：本人已经不再研究这方面的内容了，所以如果遇到问题可能我无法提供帮助

## 相对于原版的改动

在使用wd-tagger模型的时候，会先在本地路径中搜索模型，如果搜索不到才到Huggingface中下载。


## 安装方式

1. *在webui内安装*

*Extensions* -> *Install from URL* -> Enter URL of this repository -> Press *Install* button

2. 克隆当前仓库到 `extensions/里`

在webui根目录中使用下面的命令

```shell
$ git clone https://github.com/xiaolaa2/stable-diffusion-webui-wd14-tagger.git extensions/tagger
```


## 使用方法

* 在本地使用模型
  - 从 [HuggingFace repository](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger) 中手动下载对应的模型然后放到 `${webui的model路径}/wd-tagger/${模型名字}` 例如: `/home/stable-diffusion-webui-my/models/wd-tagger/wd-v1-4-vit-tagger-v2`
  - 文件目录的格式应该如下:
    ```
     models/
     └╴wd-tagger/
       ├╴wd-v1-4-vit-tagger-v2/
       │ ├╴model.onnx
       │ ├╴selected_tags.csv
       │ └╴...
       │
       ...
    ```
如果想要使用其它的wd-tagger、模型按照同样的方法建立文件夹并把模型文件放进去就好

目前我只添加了：
- wd14-convnextv2-v2
- wd14-vit-v2
- wd14-convnext-v2
- wd14-swinv2-v2

这几个模型，如果想要使用其它模型你需要修改tagger/utils.py文件，在里面的interrogators里添加你想要的模型
