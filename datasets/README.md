---
license: Apache License 2.0
---
数据集文件元信息以及数据文件，请浏览“数据集文件”页面获取。

当前数据集卡片使用的是默认模版，数据集的贡献者未提供更加详细的数据集介绍，但是您可以通过如下GIT Clone命令，或者ModelScope SDK来下载数据集

## auto-coder.rag 测试指南

1. 解压项目中的 ./data/ouput_files_v4.tar.gz 
2. 启动 

```shell
auto-coder.rag tools count \
--tokenizer_path /Users/allwefantasy/data/tokenizer.json \
--file /Users/allwefantasy/data/ouput_files_v4
```

注意将 tokenizer_path 和 file 替换成你的目录。

#### 下载方法 
:modelscope-code[]{type="sdk"}
:modelscope-code[]{type="git"}


