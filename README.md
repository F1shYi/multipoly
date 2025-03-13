0 = Bass
1 = Guitar
2 = Piano
3 = String

# Return value of `__getitem__`

a tuple of multitrack_pianoroll_2channel, chord.





# Validation
1. Validation loss
2. Chord-conditioned sample.
3. Track-conditioned sample. (Given bass, guitar, piano, string respectively.)
4. Track+chord conditioned sample.


# Coarse Musical Ideas

能量: Sum of note nums along the timestep axis.
意外程度: 有多么不符合和弦，可以在sample的时候通过对图片的每一个列向量加不同的guidance scale来实现。
选用哪些乐器？限制音高范围？通过mask来实现