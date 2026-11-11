import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import torch
import numpy

random.seed()

def set_seed(seed = -1):
    if seed == -1:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    # Basic configuration
    debug = True
    dumpfile_uniqueid = ''
    seed = random.randint(1, 9999)
    debug = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.abspath(__file__)[:-10] # dont use os.getcwd()
    checkpoint_dir = root_dir + '/exp/snapshots'

    dataset = 'germany'
    dataset_prefix = ''
    dataset_file = ''
    dataset_cell_file = ''
    dataset_embs_file = ''

    min_lon = 0.0
    min_lat = 0.0
    max_lon = 0.0
    max_lat = 0.0
    max_traj_len = 200
    min_traj_len = 20
    cell_size = 1000.0
    cellspace_buffer = 500.0


    #===========SemMovCL=============
    semmovcl_batch_size = 128 
    cell_embedding_dim = 256
    seq_embedding_dim = 256
    moco_proj_dim =  seq_embedding_dim // 2
    moco_nqueue = 2048 
    moco_temperature = 0.05

    semmovcl_training_epochs = 20
    semmovcl_training_bad_patience = 5
    semmovcl_training_lr = 0.001
    semmovcl_training_lr_degrade_gamma = 0.5
    semmovcl_training_lr_degrade_step = 5
    semmovcl_aug1 = 'adaptive_mask'
    semmovcl_aug2 = 'adaptive_mask'
    semmovcl_local_mask_sidelen = cell_size * 11
    
    # Transformer configuration
    trans_attention_head = 4
    trans_attention_dropout = 0.1
    trans_attention_layer = 2
    trans_pos_encoder_dropout = 0.1
    trans_hidden_dim = 2048

    # Trajectory processing configuration
    traj_simp_dist = 100
    traj_shift_dist = 200
    traj_mask_ratio = 0.3
    traj_add_ratio = 0.3
    traj_subset_ratio = 0.7 # preserved ratio
    
    # Adaptive mask parameters
    adaptive_mask_base_weight = 0.5
    adaptive_mask_direction_factor = 0.4
    adaptive_mask_endpoint_weight = 0.2

    test_exp1_lcss_edr_epsilon = 0.25

    #===========trajsimi=============
    trajsimi_encoder_name = 'SemMovCL'
    trajsimi_encoder_mode = 'finetune_all'
    trajsimi_measure_fn_name = 'edr'
    trajsimi_batch_size = 128
    trajsimi_epoch = 30
    trajsimi_training_bad_patience = 10
    trajsimi_learning_rate = 0.0001
    trajsimi_learning_weight_decay = 0.0001
    trajsimi_finetune_lr_rescale = 0.5


    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(Config, k)) == type(v)
            setattr(Config, k, v)
        cls.post_value_updates()


    @classmethod
    def post_value_updates(cls):
        if 'porto' == cls.dataset:
            cls.dataset_prefix = 'porto_20200'
            cls.min_lon = -8.7005
            cls.min_lat = 41.1001
            cls.max_lon = -8.5192
            cls.max_lat = 41.2086
        else:
            cls.dataset_prefix = 'germany_20200'
            cls.min_lon = 5.866
            cls.min_lat = 47.270
            cls.max_lon = 15.042
            cls.max_lat = 55.058
        
        cls.dataset_file = cls.root_dir + '/data/' + cls.dataset_prefix
        cls.dataset_cell_file = cls.dataset_file + '_cell' + str(int(cls.cell_size)) + '_cellspace.pkl'
        cls.dataset_embs_file = cls.dataset_file + '_cell' + str(int(cls.cell_size)) + '_embdim' + str(cls.cell_embedding_dim) + '_embs.pkl'
        set_seed(cls.seed)

        cls.moco_proj_dim =  cls.seq_embedding_dim // 2

    @classmethod
    def to_str(cls): # __str__, self
        dic = cls.__dict__.copy()
        lst = list(filter( \
                        lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                        dic.items() \
                        ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])

