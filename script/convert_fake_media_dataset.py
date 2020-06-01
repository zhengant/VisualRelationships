import json
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm

IMAGE_DIR = "/home/jzda/images"
DATASET_DIR = "/home/jzda/dataset"
OUTPUT_DIR = "/home/jzda/VisualRelationships/dataset/fake_media"

INPUT_FNAMES = ["dataset_dev.csv", "dataset_train.csv", "dataset_test.csv"]

OUTPUT_FNAMES = ["valid.json", "train.json", "test.json"]

CAPTION_HEADERS = [
    "intent_2",
    "disinfo_2",
    "disinfo_1",
    "mp2_feel_1",
    "mp1_mislead_1",
    "mp3_mislead_1",
    "mp2_feel_0",
    "mp1_feel_1",
    "implications_2",
    "mp1_feel_2",
    "mp3_mislead_0",
    "mp1_mislead_2",
    "mp2_mislead_0",
    "mp1_mislead_0",
    "intent_0",
    "mp2_mislead_1",
    "intent_1",
    "implications_0",
    "mp2_mislead_2",
    "mp3_feel_2",
    "disinfo_0",
    "implications_1",
    "mp3_mislead_2",
    "mp1_feel_0",
    "mp3_feel_0",
    "mp3_feel_1",
    "mp2_feel_2",
]


def convert_data(csv_fname):
    output = []
    orig_data = pd.read_csv(csv_fname)
    for row in tqdm(orig_data.itertuples(), total=len(orig_data)):
        index = row.index
        source_fname = str(index) + "_o.jpg"
        edit_fname = str(index) + "_e.jpg"
        source_fpath = os.path.join(IMAGE_DIR, source_fname)
        edit_fpath = os.path.join(IMAGE_DIR, edit_fname)

        # test images
        try:
            img = Image.open(source_fpath)
            img.load()
            img = Image.open(edit_fpath)
            img.load()
        except (OSError, IOError):
            continue

        entry = {
            "img0": source_fpath,
            "img1": edit_fpath,
            "sents": [],
            "uid": str(index)
        }

        for cap in CAPTION_HEADERS:
            sentence = getattr(row, cap)
            if isinstance(sentence, str) and not sentence == "{}":
                entry["sents"].append(sentence)

        output.append(entry)
    
    return output

for csv_fname, json_fname in zip(INPUT_FNAMES, OUTPUT_FNAMES):
    csv_path = os.path.join(DATASET_DIR, csv_fname)
    json_path = os.path.join(OUTPUT_DIR, json_fname)

    output_data = convert_data(csv_path)
    with open(json_path, 'w') as out_fp:
        json.dump(output_data, out_fp, indent=4)

