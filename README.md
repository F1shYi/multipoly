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
train_valid_split每一次都是随机的，需要修改，否则在sample的时候可能会见到已有的数据集。