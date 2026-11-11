import os
import math
import time
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from ast import literal_eval
import traj_dist.distance as tdist

from utils.cellspace import CellSpace
from utils.tools import lonlat2meters


def inrange(lon, lat):
    if lon <= min_lon or lon >= max_lon or lat <= min_lat or lat >= max_lat:
        return False
    return True


def clean_and_output_data():
    _time = time.time()
    dfraw = pd.read_csv(raw_data_path)
    dfraw = dfraw.rename(columns={"POLYLINE": "wgs_seq"})

    dfraw = dfraw[dfraw.MISSING_DATA == False]

    # length requirement
    dfraw.wgs_seq = dfraw.wgs_seq.apply(literal_eval)
    dfraw['trajlen'] = dfraw.wgs_seq.apply(lambda traj: len(traj))
    dfraw = dfraw[(dfraw.trajlen >= min_traj_len) & (dfraw.trajlen <= max_traj_len)]
    print('Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))

    # range requirement
    dfraw['inrange'] = dfraw.wgs_seq.map(lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj))
    dfraw = dfraw[dfraw.inrange == True]
    print('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))

    # convert to Mercator
    dfraw['merc_seq'] = dfraw.wgs_seq.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])

    print('Preprocessed-output. #traj={}'.format(dfraw.shape[0]))
    dfraw = dfraw[['trajlen', 'wgs_seq', 'merc_seq']].reset_index(drop=True)  # 1372725

    dfraw.to_pickle(clean_data_path)
    print('Preprocess end. @={:.0f}'.format(time.time() - _time))
    return


def split_data():
    clean_data = pickle.load(open(clean_data_path, 'rb'))
    l = clean_data.shape[0]
    train_data = clean_data.iloc[:int(l * 0.7)]
    eval_data = clean_data.iloc[int(l * 0.7):int(l * 0.8)]
    test_data = clean_data.iloc[int(l * 0.8):]
    train_data.to_pickle(train_data_path)
    eval_data.to_pickle(eval_data_path)
    test_data.to_pickle(test_data_path)

    traj_simi_data = clean_data.iloc[int(l * 0.7):int(l * 0.7) + 10000]
    traj_simi_data.to_pickle(traj_simi_data_path)


###########################################################################

# ===calculate trajsimi distance matrix for trajsimi learning===
def traj_simi_computation(fn_name='hausdorff'):
    print("traj_simi_computation starts. fn={}".format(fn_name))
    _time = time.time()

    data_1w = pickle.load(open(traj_simi_data_path, 'rb'))
    data_1w.reset_index()

    # 2.
    fn = _get_simi_fn(fn_name)
    data_simi = _simi_matrix(fn, data_1w)

    _output_file = '{}/traj_simi_dict_{}.pkl'.format(root_path, fn_name)
    with open(_output_file, 'wb') as fh:
        tup = data_simi
        pickle.dump(tup, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print("traj_simi_computation ends. @={:.3f}".format(time.time() - _time))
    return tup


def _get_simi_fn(fn_name):
    fn = {'dfret': tdist.discret_frechet, 'sspd': tdist.sspd, 'haus': tdist.hausdorff, "dtw": tdist.dtw,
          "edr": tdist.edr, "lcss": tdist.lcss}.get(fn_name, None)
    if fn_name == 'lcss' or fn_name == 'edr':
        fn = partial(fn, eps=0.25)
    return fn


def _simi_matrix(fn, df):
    _time = time.time()

    l = df.shape[0]
    batch_size = 50
    assert l % batch_size == 0

    # parallel init
    tasks = []
    for i in range(math.ceil(l / batch_size)):
        if i < math.ceil(l / batch_size) - 1:
            tasks.append((fn, df, list(range(batch_size * i, batch_size * (i + 1)))))
        else:
            tasks.append((fn, df, list(range(batch_size * i, l))))

    num_cores = int(mp.cpu_count()) // 2
    assert num_cores > 0
    print("pool.size={}".format(num_cores))
    pool = mp.Pool(num_cores)
    lst_simi = pool.starmap(_simi_comp_operator, tasks)
    pool.close()

    # extend lst_simi to matrix simi and pad 0s
    lst_simi = sum(lst_simi, [])
    for i, row_simi in enumerate(lst_simi):
        lst_simi[i] = [0] * (i + 1) + row_simi
    assert sum(map(len, lst_simi)) == l ** 2
    print('simi_matrix computation done., @={}, #={}'.format(time.time() - _time, len(lst_simi)))

    return lst_simi


# async operator
def _simi_comp_operator(fn, df_trajs, sub_idx):
    simi = []
    l = df_trajs.shape[0]
    for _i in sub_idx:
        t_i = np.array(df_trajs.iloc[_i].wgs_seq)
        simi_row = []
        for _j in range(_i + 1, l):
            t_j = np.array(df_trajs.iloc[_j].wgs_seq)
            simi_row.append(float(fn(t_i, t_j)))
        simi.append(simi_row)
    print('simi_comp_operator ends. sub_idx=[{}:{}], pid={}' \
          .format(sub_idx[0], sub_idx[-1], os.getpid()))
    return simi


if __name__ == "__main__":
    min_lon = -8.7005
    min_lat = 41.1001
    max_lon = -8.5192
    max_lat = 41.2086
    cell_size = 100

    min_traj_len = 20
    max_traj_len = 200

    root_path = os.getcwd()
    raw_data_path = root_path + '/train(1).csv'
    clean_data_path = root_path + '/clean_porto.pkl'

    train_data_path = root_path + "/porto_train.pkl"
    eval_data_path = root_path + "/porto_eval.pkl"
    test_data_path = root_path + "/porto_test.pkl"
    traj_simi_data_path = root_path + "/porto_1w.pkl"

    # 1.
    clean_and_output_data()
    """
    Preprocessed-rm length. #traj=1499510
    Preprocessed-rm range. #traj=1372725
    Preprocessed-output. #traj=1372725
    Preprocess end. @=785
    """
    split_data()
    traj_simi_computation('lcss')  # ['hausdorff','sspd','discret_frechet']


