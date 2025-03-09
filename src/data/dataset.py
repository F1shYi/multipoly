import os
import numpy as np
from torch.utils.data import Dataset
from .utils import nmat_to_multi_prmat2c

SEG_LGTH = 32
N_BIN = 4
SEG_LGTH_BIN = SEG_LGTH * N_BIN
TRACK_NUM = 4


class DataSampleNpz:
    def __init__(
        self, data_fpath
    ) -> None:
        self.fpath = data_fpath
       
        data = np.load(self.fpath, allow_pickle=True)
        self.notes = np.array(
            data["notes"]
        ) 
        self.start_table = data["start_table"]

        self.db_pos = data["db_pos"]
        self.db_pos_filter = data["db_pos_filter"]
        self.db_pos = self.db_pos[self.db_pos_filter]
        if len(self.db_pos) != 0:
            self.last_db = self.db_pos[-1]

        self.chord = data["chord"].astype(np.int32)

        self._nmat_dict = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._multitrack_prmat2c_dict = dict(zip(self.db_pos, [None] * len(self.db_pos)))

    def __len__(self):
        """Return number of complete 8-beat segments in a song"""
        return len(self.db_pos)

    def note_mat_seg_at_db(self, db):
        """
        Select rows (notes) of the note_mat which lie between beats
        [db: db + 8].
        """

        seg_mats = []
        notes = self.notes
        start_table = self.start_table.tolist()
        s_ind = start_table[db]
        if db + SEG_LGTH_BIN in start_table:
            e_ind = start_table[db + SEG_LGTH_BIN]
            note_seg = np.array(notes[s_ind:e_ind])
        else:
            note_seg = np.array(notes[s_ind:])  # NOTE: may be wrong
        seg_mats.extend(note_seg)

        seg_mats = np.array(seg_mats)
        if seg_mats.size == 0:
            seg_mats = np.zeros([0, 4])
        return seg_mats

    @staticmethod
    def reset_db_to_zeros(note_mat, db):
        note_mat[:, 0] -= db


    def store_nmat_seg(self, db):
        """
        Get note matrix (SEG_LGTH) of orchestra(x) at db position
        """
        if self._nmat_dict[db] is not None:
            return

        nmat = self.note_mat_seg_at_db(db)
        self.reset_db_to_zeros(nmat, db)
        self._nmat_dict[db] = nmat

    def store_prmat2c_seg(self, db):
        """
        Get piano roll format (SEG_LGTH) from note matrices at db position
        """
        if self._multitrack_prmat2c_dict[db] is not None:
            return

        multitrack_prmat2c = nmat_to_multi_prmat2c(self._nmat_dict[db], SEG_LGTH_BIN, TRACK_NUM)
        self._multitrack_prmat2c_dict[db] = multitrack_prmat2c

  
    def _store_seg(self, db):
        self.store_nmat_seg(db)
        self.store_prmat2c_seg(db)
        

    def _get_item_by_db(self, db):
        self._store_seg(db)
        seg_prmat2c = self._multitrack_prmat2c_dict[db]
        chord = self.chord[db // N_BIN : db // N_BIN + SEG_LGTH]
        if chord.shape[0] < SEG_LGTH:
            chord = np.append(
                chord, np.zeros([SEG_LGTH - chord.shape[0], 36], dtype=np.int32), axis=0
            )
        return seg_prmat2c, chord

    def __getitem__(self, idx):
        db = self.db_pos[idx]
        return self._get_item_by_db(db)



class LMDDataset(Dataset):
    def __init__(self, data_samples):
        super(LMDDataset, self).__init__()
        # a list of DataSampleNpz
        self.data_samples = data_samples
        self.lgths = np.array([len(d) for d in self.data_samples], dtype=np.int64)
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
    def __init__(self, data_folders):
        pass
