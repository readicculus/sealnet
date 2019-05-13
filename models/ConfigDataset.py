import pickle
import sys
from torch.utils.data import Dataset
import dataset.utils as utilsx
import dataset.transforms as transformsx

class ConfigDataset(Dataset):
    def __init__(self, config, type):
        try:
            sys.modules['utils'] = utilsx
            sys.modules['transforms'] = transformsx
            pickle_file = open(config, 'rb')
            config = pickle.load(pickle_file)
        except:
            raise Exception("Could not load file " + config)

        if type == "train":
            self.meta_data, _ = utilsx.get_train_test_meta_data(config)
            self.path, _ = utilsx.get_train_test_base(config)
        else:
            _, self.meta_data= utilsx.get_train_test_meta_data(config)
            _, self.path = utilsx.get_train_test_base(config)
