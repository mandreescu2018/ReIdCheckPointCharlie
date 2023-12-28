from .dataset_bases import BaseImageDataset
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17



class CombinedDataset(BaseImageDataset):
    
    def __init__(self, root='') -> None:

        super(CombinedDataset, self).__init__()

        self._datasets_list = {
            'market1501': Market1501,
            'dukemtmc': DukeMTMCreID,
            'msmt17': MSMT17,    
        }

        self.train = []
        self.query = []
        self.gallery = []
        self.num_train_pids = 0
        self.num_train_cams = 0
        self.num_train_vids = 0

        self.train = []
        for ds_key, ds_class in self._datasets_list.items():
            dataset = ds_class(root=root, pid_begin = self.num_train_pids)
            self.num_train_pids += dataset.num_train_pids
            self.num_train_cams += dataset.num_train_cams
            self.num_train_vids += dataset.num_train_vids
            self.train.extend(dataset.train)
            self.query.extend(dataset.query)
            self.gallery.extend(dataset.gallery)

    def combine_train_dirs(self, datasets):
        for ds in datasets:
            pass


    
    
