# 说明

&emsp;这个Repo主要记录自己学习一些经典的语言模型源码时添加的注释，便于之后回顾复习的时候使用。主要包括**Transformer、Transformer-XL、Bert以及XLnet**。



- Transformer： 学习的源码来自[jadore801120](https://github.com/jadore801120/attention-is-all-you-need-pytorch)。分为两块，首先是模型训练部分的主要代码，建议源码的阅读顺序为：`Modules -> SubLayers -> Layers -> Constants -> Models `；然后是推理（`inference`）部分的代码，阅读顺序为`Beam -> Translator`。关于`Optim`这个相对比较简单，是原论文提到的一种加速收敛的方法，简单来说就是在前`warmup`次迭代时增大学习率，之后不断减小学习率，单独看一下就行。主要的难点在于**束搜索**代码的理解，可能需要花费一点时间。

