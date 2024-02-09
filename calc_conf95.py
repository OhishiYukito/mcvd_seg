
from config import dict2namespace
from tools.functions_with_config import FuncsWithConfig
from models.fvd.fvd import frechet_distance

import scipy.stats as st
import os
import argparse
import pickle
import yaml
import numpy as np

 

# get args
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help="path of config (.yaml)", default='bair_fp_deeper_1_0_5.yaml')
args = parser.parse_args()

# load config
with open('config/'+args.config_path) as f:
   dict_config = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(dict_config)


funcs = FuncsWithConfig(config)
tags = funcs.get_tags()

folder_path = os.path.join('results', config.data.dataset.upper(), args.config_path.replace(".yaml", ""))
# load result
with open(os.path.join(folder_path, f'[{args.config_path.replace(".yaml", "")}]_'+'-'.join(tags)+'_test_results.pkl'), "rb") as f:
    result = pickle.load(f)
    

task = "future_prediction"
key = "embeddings"
target_embeddings = np.concatenate(np.array(result[task][key]["target"]))
pred_embeddings = np.concatenate(np.array(result[task][key]["pred"]))
fvd = round(frechet_distance(pred_embeddings, target_embeddings),3)
if config.eval.preds_per_test > 1:
    fvds_list = []
    # calc FVD for each trajectory and its average
    trajs = np.random.choice(np.arange(config.eval.preds_per_test), (config.eval.preds_per_test), replace=False)
    for traj in trajs:
        fvds_list.append(frechet_distance(pred_embeddings[traj::config.eval.preds_per_test], target_embeddings))
    fvd_traj_avg, fvd_traj_std = round(float(np.mean(fvds_list)),3), round(float(np.std(fvds_list)),3)
    fvd_traj_conf95 = fvd_traj_avg - float(st.norm.interval(alpha=0.95, loc=fvd_traj_avg, scale=st.sem(fvds_list))[0])
    
    #calc_result[task]["fvd_traj"] = {"avg":fvd_traj_avg, "std":fvd_traj_std}
    print(f"fvd_traj:\t{fvd_traj_avg}±{fvd_traj_std}, conf95={fvd_traj_conf95}")
    
#calc_result[task]["fvd"] = {"avg":fvd}
print(f"fvd:\t{fvd}")
    