import os
import json
import numpy as np

import torch

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, type(bytes)):
            return str(obj, encoding='utf-8')
        elif type(obj) == int:
            return str(obj)
        elif type(obj) == float:
            return str(obj)
        elif type(obj) == bool:
            return str(obj)
        elif type(obj) == torch.device:
            return str(obj)
        else:
            print(type(obj))
            return super(MyEncoder, self).default(obj)

class ConfigClass():
    def __init__(self,
                 network_type=None,
                 root_write_path=None,
                 root_read_path=None,
                 mode=None,
                 dist_type=None,
                 device="cpu",
                 train_flag="test",
                 train_set=None,
                 query_set=None,
                 base_set=None):
        self.my_dict = {}
        self.my_dict["network_type"] = network_type
        self.my_dict["root_write_path"] = root_write_path
        self.my_dict["root_read_path"] = root_read_path
        self.my_dict["mode"] = mode
        self.my_dict["dist_type"] = dist_type
        self.my_dict["device"] = device
        self.my_dict["train_flag"] = train_flag
        self.my_dict["train_set"] = train_set
        self.my_dict["query_set"] = query_set
        self.my_dict["base_set"] = base_set
        self.write_config_to_file()
        
    def write_config_to_file(self):
        if not os.path.exists(self.my_dict["root_write_path"] + "/train_config/"):
            os.mkdir(self.my_dict["root_write_path"] + "/train_config/")
        with open(self.my_dict["root_write_path"] + "/train_config/" + self.my_dict["train_flag"] + ".json", "w", encoding = "utf-8") as f:
            json.dump(self.my_dict, f, ensure_ascii = False, cls = MyEncoder)
