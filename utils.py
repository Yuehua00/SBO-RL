import os
import json
import datetime
import random
import string

from config import args


# 獲得輸出的資料夾路徑
def get_output_folder():
    if (not args.save_result):
        return
    random_suffix = "".join(random.choices(string.ascii_uppercase, k = 8))
    env_names = args.env_names[0]
    for i in range(1, len(args.env_names)):
        env_names += "_" + args.env_names[i]
    output_folder = os.path.join(args.output_path, f"[{args.algorithm}][{env_names}][{args.seed}][{datetime.date.today()}][Learning Curve][{random_suffix}]")
    args.output_folder = output_folder

    if (not os.path.exists(args.output_path)):
        os.makedirs(args.output_path)
    os.makedirs(args.output_folder)

    # 將args寫到檔案中
    with open(os.path.join(args.output_folder, "config.json"), "w") as file:
        json.dump(vars(args), file, indent = 4)
        file.close()