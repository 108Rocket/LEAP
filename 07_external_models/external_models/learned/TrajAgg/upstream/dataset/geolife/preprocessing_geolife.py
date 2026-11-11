import os
import math
import time
import pickle
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
import traj_dist.distance as tdist

from AAAI.utils.tools import lonlat2meters
from AAAI.utils.cellspace import CellSpace


def inrange(lon, lat):
    if lon <= min_lon or lon >= max_lon or lat <= min_lat or lat >= max_lat:
        return False
    return True

############################################################################

def get_all_trajs_path(data_path):
    traj_paths = []
    for i in range(0, 182):
    # for i in range(0, 2):
        user_data_path = data_path + '/' + str(i).zfill(3)
        if os.path.exists(user_data_path):
            traj_data_path = user_data_path + '/Trajectory'
            if os.path.exists(traj_data_path):
                trajs_name = os.listdir(traj_data_path)
                traj_paths.extend([traj_data_path + '/' + traj_name for traj_name in trajs_name])
    return traj_paths


def read_traj(traj_path):
    df = pd.read_csv(traj_path, header=None, sep=',', skiprows=6, names=['lat', 'lon', 'zero', 'alt', 'days', 'date', 'time'])
    lats = df["lat"].to_list()
    lons = df["lon"].to_list()
    trajs = []
    for lat, lon in zip(lats, lons):
        record = [lon, lat]
        trajs.append(record)
    return trajs


def batch_read_traj(traj_paths):
    all_trajs = []
    for i, traj_path in enumerate(traj_paths):
        traj = read_traj(traj_path)
        all_trajs.append(traj)
        if i % 1000 == 0:
            print('read {} trajs'.format(i))
    print(f'{len(all_trajs)}done!')
    return all_trajs

def filter_data(src, cs):
    merc_seq_ = [list(lonlat2meters(p[0], p[1])) for p in src]
    tgt = [[cs.get_cellid_by_point(merc[0], merc[1]), wgs, merc] for wgs, merc in zip(src, merc_seq_)]
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i - 1][0]]
    tgt, wgs_seq, merc_seq = zip(*tgt)
    return list(wgs_seq), list(merc_seq)


def clean_and_output_data(data):
    _time = time.time()
    cellspace = CellSpace(cell_size, cell_size, min_lon, min_lat, max_lon, max_lat)
    dfraw = pd.DataFrame({'wgs_seq': [traj for traj in data]})

    # 1.range filter
    dfraw['inrange'] = dfraw.wgs_seq.map(lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj))
    dfraw = dfraw[dfraw.inrange == True]
    print('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))     #

    # 2.
    dfraw['wgs_seq'], dfraw['merc_seq'] = zip(* dfraw.wgs_seq.apply(lambda traj: filter_data(traj, cellspace)))

    # 3.len filter
    dfraw['traj_len'] = dfraw.wgs_seq.apply(lambda traj: len(traj))
    dfraw = dfraw[(dfraw.traj_len >= min_traj_len) & (dfraw.traj_len <= max_traj_len)]
    print('Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))        #

    # 4.output
    dfraw = dfraw[['wgs_seq', 'merc_seq', "traj_len"]].reset_index(drop=True)

    dfraw.to_pickle(clean_data_path)
    print('Preprocess end. @={:.0f}'.format(time.time() - _time))
    return

def sample_data():
    clean_data = pickle.load(open(clean_data_path, 'rb'))
    idx = random.sample(range(clean_data.shape[0]), 10000)
    data_1w = clean_data.iloc[idx]
    data_1w.to_pickle(fine_tuning_data_path)
    return

###########################################################################
# ===calculate trajsimi distance matrix for trajsimi learning===
def traj_simi_computation(fn_name='hausdorff'):
    print("traj_simi_computation starts. fn={}".format(fn_name))
    _time = time.time()

    data_1w = pickle.load(open(fine_tuning_data_path, 'rb'))
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
    min_lon = 116.25  # √
    max_lon = 116.5  # √
    min_lat = 39.8  # √
    max_lat = 40.1  # √
    cell_size = 100
    min_traj_len = 20
    max_traj_len = 300

    root_path = os.getcwd()
    raw_data_path = root_path + "/Geolife Trajectories 1.3/Data/"
    clean_data_path = root_path + '/clean_geolife.pkl'
    fine_tuning_data_path = root_path + "/geolife_1w.pkl"

    # 1
    traj_paths = get_all_trajs_path(raw_data_path)      # 18670
    trajs = batch_read_traj(traj_paths)                 #
    clean_and_output_data(trajs)
    """
    18670done!
    Preprocessed-rm range. #traj=14988
    Preprocessed-rm length. #traj=10940
    Preprocess end. @=32
    """
    # 3.
    sample_data()
    traj_simi_computation('dtw')            # ['haus','sspd','dfret',"dtw"]
