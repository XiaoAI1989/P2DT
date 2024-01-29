import logging
import os
import re

import hydra
import torch

from core import final_eval

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')


def get_matching_time_folder(date):
    # 正则表达式来匹配时间格式的文件夹
    time_pattern = re.compile(r'\d{2}-\d{2}-\d{2}_')
    # 列出给定日期的所有文件夹
    root_dir=os.getcwd()
    base_path = root_dir+f"/runs/{date}/"

    if not os.path.exists(base_path):
        print(f"No such directory: {base_path}")
        return []
    time_folders = [f for f in os.listdir(base_path) if time_pattern.match(f)]
    return time_folders

date = ""

time_folders = get_matching_time_folder(date)
time = time_folders[0]
filepath = f"runs/{date}/{time}/"


@hydra.main(config_path=filepath+".hydra", config_name="config", version_base="1.3")

def main(cfg):

    seed=42
    root_dir = hydra.utils.get_original_cwd()

    exp_path = root_dir +"/"+ filepath

    final_eval(cfg.train.env_name,cfg.rtg_target,logger,cfg,seed,root_dir,exp_path)




if __name__ == "__main__":

    main()