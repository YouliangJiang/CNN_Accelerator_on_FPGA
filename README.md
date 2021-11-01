# CNN_Accelerator_on_FPGA
A CNN Accelerator based on FPGA using HLS

hls文件夹里存放的是hls代码，结构如下：
“stream_tools.h”中定义了一些关于使用axi_stream接口的函数
“sliding_window_unit.h”中定义了滑动窗口算子的实现函数
“function.h”中目前定义了padding操作的实现函数
“pool2d.h”中定义了池化算子的实现函数
“atss_0426_round.h”中定义了神经网络所使用的权重参数
“atss_0426_round.cpp”中定义了卷积层算子的实现函数与神经网络模型的实现函数
