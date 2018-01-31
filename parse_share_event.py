import pickle
import numpy as np
import pandas as pd
import glob
import datetime
import tqdm

"""
EvDef(type=0, note=None, gid=18, host='kitami-dc-l2s1')
24_tokyo-dc-gm1_20120415.dump
"""

def parse_evdef_text(evdef_text):
    evdef_text = evdef_text.strip()
    lt_id = evdef_text.split("gid=")[1].split(",")[0]
    host = evdef_text.split("host='")[1].split("'")[0]
    return lt_id + "_" + host


def open_dump(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f, encoding="bytes")

if __name__ == "__main__":
    print("make def_event dict...")
    def_dict = {}
    for file_path in tqdm.tqdm(glob.glob("share_event_rplinear/def_*")):
        def_file = [i.strip() for i in open(file_path, "r").readlines()]
        tmp_def_dict = [i.split(":") for i in def_file]
        tmp_def_dict = {int(i[0]): parse_evdef_text(i[1]) for i in tmp_def_dict}
        # def_dict[file_path.split("def_")[-1]] = tmp_def_dict
        
        event_obj = open_dump("share_event_rplinear/event_"+file_path.split("def_")[-1])
        for k, v in event_obj.items():
            dump_name = tmp_def_dict[k]
            date = file_path.split("_")[-1]
            with open("share_event_rplinear_tmp/{0}_{1}.dump".format(dump_name, date), "wb") as f:
                pickle.dump(v, f)

    print("make one day long dumps...")

    for i in tqdm.tqdm(set(["_".join(i.split("/")[-1].split("_")[:2]) for i in glob.glob("share_event_rplinear_tmp/*")])):
        time_series = []
        for j in glob.glob("share_event_rplinear_tmp/"+i+"*"):
            obj = open_dump(j)
            time_series.extend(obj)
        with open("rplinear_oneday/"+i+".dump", "wb") as f:
            pickle.dump(time_series, f)
