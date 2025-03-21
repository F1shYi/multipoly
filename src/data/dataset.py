import os
import numpy as np
from torch.utils.data import Dataset

SEG_LGTH = 32
N_BIN = 4
SEG_LGTH_BIN = SEG_LGTH * N_BIN
TRACK_NUM = 4


class LMDDataset(Dataset):
    def __init__(self, data_samples):
        super(LMDDataset, self).__init__()
        # a list of DataSampleNpz
        self.data_samples = data_samples
        self.lgths = np.array([len(d)
                              for d in self.data_samples], dtype=np.int64)
        self.lgth_cumsum = np.cumsum(self.lgths)

    def __len__(self):
        return self.lgth_cumsum[-1]

    def __getitem__(self, index):
        # song_no is the smallest id that > dataset_item
        song_no = np.where(self.lgth_cumsum > index)[0][0]
        song_item = index - np.insert(self.lgth_cumsum, 0, 0)[song_no]

        song_data = self.data_samples[song_no]
        return song_data[song_item]


class EightBarSegmentDataset(Dataset):
    def __init__(self, data_folder):

        self.data_fpaths = [os.path.join(data_folder, path)
                            for path in os.listdir(data_folder)]
        self.datas = [np.load(self.data_fpaths[index], allow_pickle=True)
                      for index in range(len(self.data_fpaths))]

    def __len__(self):
        return len(self.data_fpaths)

    def __getitem__(self, index):

        data = self.datas[index]
        multi_prmat2c = data["multi_prmat_2c"]
        onehot_chord = data["onehot_chord"]

        return multi_prmat2c, onehot_chord


if __name__ == "__main__":
    training_folder = "/root/autodl-tmp/multipoly/data/train"
    train_ds = EightBarSegmentDataset(training_folder)
    print(len(train_ds))

    multi_prmat2c, onehot_chord = train_ds[20]
    print(multi_prmat2c.shape, onehot_chord.shape)
    from data.utils import prmat2c_to_midi_file, chd_to_midi_file

    prmat2c_to_midi_file(np.sum(multi_prmat2c, axis=0), "dataset_test.mid")
    chd_to_midi_file(onehot_chord, "dataset_test_chord.mid")
