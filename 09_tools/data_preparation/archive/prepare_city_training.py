"""Freeze disjoint SDD development splits without changing published tests."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from audit_cities import reader, digest
from paths import load_paths


def pack(paths,prefix):
    lens=np.array([len(p) for p in paths],np.int64)
    return {prefix+'_points':np.concatenate(paths).astype(np.float64),
            prefix+'_offsets':np.r_[0,lens.cumsum()]}


def main(argv=None):
    paths_config = load_paths("prepare_city_training", argv)
    if paths_config is None:
        return
    DEST = paths_config["output_root"]
    all_splits={}
    for city in ('Chengdu','Geolife','Porto'):
        source=paths_config["source_dir"]/city
        out=DEST/city
        out.mkdir(parents=True,exist_ok=False)
        paths=reader.numeric_load(source/'train_list')
        queries=reader.numeric_load(source/'query_list')
        hashes=digest(paths)
        query_hashes=set(digest(queries))
        groups={}
        excluded=[]
        for i,h in enumerate(hashes):
            if h in query_hashes:
                excluded.append(i)
            else:
                groups.setdefault(h,[]).append(i)
        rng=np.random.default_rng(2000)
        order=rng.permutation(len(groups))
        group_list=list(groups.values())
        count=int(round(.2*len(group_list)))
        va=sorted(i for g in order[:count] for i in group_list[g])
        tr=sorted(i for g in order[count:] for i in group_list[g])
        assert not (set(hashes[i] for i in tr)&set(hashes[i] for i in va))
        assert not (set(hashes[i] for i in tr+va)&query_hashes)
        assert len(tr)+len(va)+len(excluded)==len(paths)
        train=[np.asarray(paths[i],np.float64) for i in tr]
        val=[np.asarray(paths[i],np.float64) for i in va]
        flat=np.concatenate(train)
        mean=flat.mean(0)
        scale=float(np.sqrt(np.mean((flat-mean)**2)))
        assert scale>0
        packed={**pack(train,'train'),**pack(val,'validation'),
                'geometry_mean':mean,'geometry_std':np.full(2,scale)}
        np.savez(out/'raw_inputs.npz',**packed)
        protocol=dict(city=city,seed=2000,published_training_rows=len(paths),train_rows=len(tr),validation_rows=len(va),
            train_ids=tr,validation_ids=va,excluded_query_overlap_train_ids=excluded,
            excluded_geometry_groups=len(set(hashes[i] for i in excluded)),
            grouping='identical original FP64 coordinate sequences stay in one split',
            validation_rule='20 percent of unique training geometry groups, seed2000',
            query_and_database_unchanged=True,test_labels_used_for_split=False,
            test_geometry_used_only_to_exclude_exact_training_overlap=True,
            geometry_mean=mean.tolist(),geometry_std=[scale,scale],
            geometry_normalization='train-only translation and isotropic scale; preserves SDD distances up to one factor',
            train_geometry_hash=hashlib.sha256(''.join(hashes[i] for i in tr).encode()).hexdigest(),
            validation_geometry_hash=hashlib.sha256(''.join(hashes[i] for i in va).encode()).hexdigest(),metrics={})
        for metric,folder in [('haus','Haus'),('dfd','DFD'),('dtw','DTW')]:
            matrix=reader.numeric_load(source/folder/(metric+'_train_distance_matrix_result'))
            if not np.allclose(matrix,matrix.T,rtol=0,atol=1e-12):
                assert not np.count_nonzero(np.tril(matrix,-1)) or not np.count_nonzero(np.triu(matrix,1))
                matrix=matrix+matrix.T
            tm=matrix[np.ix_(tr,tr)]
            vm=matrix[np.ix_(va,va)]
            assert (np.diag(tm)==0).all() and np.isfinite(tm).all() and tm.min()>=0
            np.save(out/(metric+'_train.npy'),tm)
            np.save(out/(metric+'_validation.npy'),vm)
            np.save(out/(metric+'_validation_to_train.npy'),matrix[np.ix_(va,tr)])
            protocol['metrics'][metric]=dict(train_maximum=float(tm.max()),
                teacher_unit='original longitude/latitude Euclidean distance',
                triGeo_teacher_scale=scale,train_matrix_shape=list(tm.shape),validation_matrix_shape=list(vm.shape))
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        all_splits[city]=protocol
        print(json.dumps(dict(city=city,train=len(tr),validation=len(va),excluded=len(excluded))),flush=True)
    reader.save(DEST/'TRAIN_SPLITS_DONE.json',
                dict(time=time.time(),cities=all_splits,training_started=False))


if __name__=='__main__':
    main()
