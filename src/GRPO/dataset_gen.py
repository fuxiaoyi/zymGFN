
import numpy as np
import os
import random
import argparse
import pandas as pd
import statistics
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)

args = parser.parse_args()

def append_to_csv(name, sequence, TM, TM_norm_que, algn, iteration_num, output_file):
    file_exists = os.path.exists(output_file) and os.stat(output_file).st_size > 0
    with open(output_file, "a", newline="") as csvfile:
        fieldnames = [
            "name",
            "sequence",
            "TM",
            "TM_norm_que",
            "algn",
            "iteration_num",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "name" : name,
            "sequence" : sequence,
            "TM" : TM,
            "TM_norm_que" : TM_norm_que,
            "algn" : algn,
            "iteration_num" : iteration_num,
        })

def generate_dataset(iteration_num, ec_label):

    output_file = "logs.csv"
     
    with open(f"seq_gen_{ec_label}_iteration{iteration_num}.fasta", "r") as f:
        rep_seq = f.readlines()

    with open(f"alpha_{ec_label}_TM_iteration{iteration_num}", "r") as f:
        alpha_TM_score = f.readlines()
        
    sequences_rep = dict()
     
    for line in rep_seq:
            if ">" in line:
                name = line.split("\t")[0].replace(">", "").replace("\n", "")
                emb_identifier = line.replace(">", "").replace("\n", "")
            else:
                aa = line.strip()
                sequences_rep[name] = {
                              "sequence" : aa,
                                      }
     
    for entry in alpha_TM_score:

            name = entry.split("\t")[0]
            TM = entry.split("\t")[2]
            TM_norm_que = entry.split("\t")[4]
            algn = int(entry.split("\t")[5])
            sequence = sequences_rep[str(name)]['sequence']

            append_to_csv(name, sequence, TM, TM_norm_que, algn, iteration_num, output_file)

print("iteration number",args.iteration_num)
generate_dataset(args.iteration_num, args.label)