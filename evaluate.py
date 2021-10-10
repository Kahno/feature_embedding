import tensorflow as tf
import pandas as pd
import numpy as np
import lightgbm as lgb

import time
import gc 

from base_model import BaseModel
from embedding_modules import *


# Space of hyperparameters
scale = [1/10, 1/3, 3/4, 2, 5, 10]
num_hidden = [1, 2, 3]
num_factors = [4, 6, 8, 10, 15]
num_meta_factors = [4, 8, 15]
squeeze = [2, 3, 10]
trees = [10, 20, 30]
leaves = [10, 20, 30, 45, 90]


def hyper_FM_Scale():
    result = []
    for s in scale:
        for h in num_hidden:
            for k in num_factors:
                result.append([s, h, k])
    return result


def hyper_LR_Scale():
    result = []
    for s in scale:
        for h in num_hidden:
            result.append([s, h])
    return result


def hyper_LR_GBDT():
    result = []
    for t in trees:
        for l in leaves:
            result.append([t, l])
    return result


def hyper_FM_GBDT():
    result = []
    for k in num_factors:
        for t in trees:
            for l in leaves:
                result.append([t, l, k])
    return result


def hyper_FM():
    result = []
    for f in num_factors:
        result.append(f)
    return result


def hyper_FM_Meta():
    result = []
    for f in num_factors:
        for f2 in num_meta_factors:
            result.append([f, f2])
    return result


def hyper_FM_Autoencoder():
    result = []
    for f in num_factors:
        for s in squeeze:
            result.append([f, s])
    return result


def hyper_FM_Net():
    result = []
    for f in num_factors:
        for h in num_hidden:
            result.append([f, h])
    return result


def hyper_FM_Weight():
    result = []
    for f in num_factors:
        result.append(f)
    return result


def extract_model_performance(model, test_x, test_y, BATCH_SIZE):
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
    total_time = 0

    for i in range(19):
        cur_df = pd.read_csv(f"{path}/samo0000000000{str(i).zfill(2)}")
        
        start = time.time()
        model.fit(cur_df[feats].values, cur_df["click"].values, epochs=1, batch_size=BATCH_SIZE)
        end = time.time()

        total_time += (end - start)
        
        del cur_df

    start = time.time()    
    result = model.evaluate(test_x, test_y, batch_size=BATCH_SIZE)
    end = time.time()

    pred_time = (end - start)
    
    print(model.name)
    print(result[1])
    print("###############################################################")

    return (result[1], total_time, pred_time)


def extract_model_performance_GBDT(model, test_x, test_y, BATCH_SIZE, 
                                   num_trees, num_leaves):
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    params = {
        "objective": "binary", 
        "num_leaves": num_leaves, 
        "num_trees": num_trees, 
        "min_data_in_leaf": 50,
        
        "learning_rate": 0.2,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": 5,
    }
    df = pd.read_csv(f"{path}/samo000000000000")

    total_time = 0

    gbdt_model = lgb.LGBMClassifier(**params)

    start = time.time()
    gbdt_model.fit(df[feats].values, df["click"].values)
    end = time.time()

    total_time += (end - start)

    del df
    gc.collect()
    
    index_correction = np.cumsum([0] + [num_leaves] * (num_trees - 1))

    for i in range(19):
        cur_df = pd.read_csv(f"{path}/samo0000000000{str(i).zfill(2)}")

        start = time.time()
        tree_data = gbdt_model.predict(cur_df[feats].values, pred_leaf=True) + index_correction
        model.fit(np.concatenate((tree_data, cur_df[feats].values), axis=1), cur_df["click"].values, epochs=1, batch_size=BATCH_SIZE)
        end = time.time()

        total_time += (end - start)

        del cur_df
        del tree_data
        gc.collect()
        
    start = time.time()
    tree_data = gbdt_model.predict(test_x, pred_leaf=True) + index_correction
    result = model.evaluate(np.concatenate((tree_data, test_x), axis=1), test_y, batch_size=BATCH_SIZE)
    end = time.time()

    pred_time = (end - start)

    del tree_data
    gc.collect()

    print(model.name)
    print(result[1])
    print("###############################################################")

    return (result[1], total_time, pred_time)


path = "Samo_Mag"
df = pd.read_csv(f"{path}/samo000000000000")
feats = [c for c in df.columns if c != "click"]
del df
gc.collect()

dim = len(feats)
BATCH_SIZE = 8192

# Set approximate model size
param_bound = 30_000_000

# Generate the test set
test_dfs = []
for i in range(19, 23):
    test_dfs.append(pd.read_csv(f"{path}/samo0000000000{str(i).zfill(2)}"))
test_df = pd.concat(test_dfs, ignore_index=True)
test_x = test_df[feats].values
test_y = test_df["click"].values
del test_df
gc.collect()

# LR-Scale
for hp in hyper_LR_Scale():
    cur_model = BaseModel(
        num_feats=dim, 
        num_bins=param_bound, 
        lin_modules=[EScaling(dim, hp[0], hp[1])],
    )
    extract_model_performance(cur_model, test_x, test_y, BATCH_SIZE)
    del cur_model
    gc.collect()

# FM
for hp in hyper_FM():
    nbins = round(param_bound / (hp+1))
    cur_model = BaseModel(
        num_feats=dim, 
        num_bins=nbins, 
        num_factors=hp,
    )
    extract_model_performance(cur_model, test_x, test_y, BATCH_SIZE)
    del cur_model
    gc.collect()

# FM-Meta
for hp in hyper_FM_Meta()[:2]:
    K, C = hp
    nbins = round(param_bound / (K+1))
    cur_model = BaseModel(
        num_feats=dim, 
        num_bins=nbins, 
        num_factors=K,
        int_modules=[EMeta(dim, K, C)],
    )
    extract_model_performance(cur_model, test_x, test_y, BATCH_SIZE)
    del cur_model
    gc.collect()    

# FM-Autoencoder
for hp in hyper_FM_Autoencoder():
    K, s = hp
    nbins = round(param_bound / (K+1))
    cur_model = BaseModel(
        num_feats=dim, 
        num_bins=nbins, 
        num_factors=K,
        int_modules=[EAutoencoder(dim, s, K),],
        lin_modules=[EAutoencoder(dim, s),],
    )
    extract_model_performance(cur_model, test_x, test_y, BATCH_SIZE)
    del cur_model
    gc.collect()

# FM-Net
for hp in hyper_FM_Net():
    K, h = hp
    nbins = round(param_bound / (K+1))
    cur_model = BaseModel(
        num_feats=dim, 
        num_bins=nbins, 
        num_factors=K,
        both_modules=[ENet(dim, K, h), ],
    )
    extract_model_performance(cur_model, test_x, test_y, BATCH_SIZE)
    del cur_model
    gc.collect()

# FM-Weight
for hp in hyper_FM_Weight():
    nbins = round(param_bound / (hp+1))
    cur_model = BaseModel(
        num_feats=dim, 
        num_bins=nbins, 
        num_factors=hp,
        int_modules=[EWeighting(dim, hp)],
        lin_modules=[EWeighting(dim, 1)],
    )
    extract_model_performance(cur_model, test_x, test_y, BATCH_SIZE)
    del cur_model
    gc.collect()

# LR
cur_model = BaseModel(
    num_feats=dim, 
    num_bins=param_bound, 
)
extract_model_performance(cur_model, test_x, test_y, BATCH_SIZE)
del cur_model
gc.collect()

# LR-Autoencoder
for hp in [2, 3, 10]:
    cur_model = BaseModel(
        num_feats=dim, 
        num_bins=param_bound, 
        lin_modules=[EAutoencoder(dim, hp),],
    )
    extract_model_performance(cur_model, test_x, test_y, BATCH_SIZE)
    del cur_model
    gc.collect()

# LR-Weight
cur_model = BaseModel(
    num_feats=dim, 
    num_bins=param_bound, 
    lin_modules=[EWeighting(dim, 1)],
)
extract_model_performance(cur_model, test_x, test_y, BATCH_SIZE)
del cur_model
gc.collect()

# FM-Scale
for hp in hyper_FM_Scale():
    nbins = round(param_bound / (hp[2]+1))
    cur_model = BaseModel(
        num_feats=dim, 
        num_bins=nbins, 
        num_factors=hp[2],
        lin_modules=[EScaling(dim, hp[0], hp[1])],
        int_modules=[EScaling(dim, hp[2], hp[0], hp[1])],
    )
    extract_model_performance(cur_model, test_x, test_y, BATCH_SIZE)
    del cur_model
    gc.collect()

# LR-GBDT
for hp in hyper_LR_GBDT():
    cur_model = BaseModel(
        gbdt_enhance={"num_trees": hp[0], "num_leaves": hp[1]},
        num_feats=dim, 
        num_bins=param_bound, 
    )
    extract_model_performance_GBDT(cur_model, test_x, test_y, BATCH_SIZE, hp[0], hp[1])
    del cur_model
    gc.collect()

# FM-GBDT
for hp in hyper_FM_GBDT():
    nbins = round(param_bound / (hp[2]+1))
    cur_model = BaseModel(
        gbdt_enhance={"num_trees": hp[0], "num_leaves": hp[1]},
        num_feats=dim, 
        num_bins=nbins, 
        num_factors=hp[2],
    )
    extract_model_performance_GBDT(cur_model, test_x, test_y, BATCH_SIZE, hp[0], hp[1])
    del cur_model
    gc.collect()