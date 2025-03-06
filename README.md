0 = Bass
1 = Guitar
2 = Piano
3 = String

# Return value of `__getitem__`

a tuple of multitrack_pianoroll_2channel, chord.

# Training Configs

|      | Polyffusion | Random |
|------|-------------|--------|
| 5e-5 | exp1        | exp2   |
| 1e-5 | exp5        |        |
| 1e-7 | exp3        | exp4   |


# TODO

采样一些8-bar segment听一下


流程：

0. 改网络结构，用3dconv zero init；改学习率，分不同的学习率；改lr，用warm up+decay
1. 加载数据，从val dl中随机选一首曲子
2. 每隔一定的step就验证，在验证的过程中，生成并保存结果 