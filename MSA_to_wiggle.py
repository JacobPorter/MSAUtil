#!/usr/bin/env python3
"""
Produces an MSA file consisting of columns of a given reference id
and a wiggle file of entropy for the reference id.
Removes gaps in the reference sequence.
TODO: this could be done in parallel.
TODO: this could be done with seek on a binary file to reduce memory requirements.
Authors: Jacob S. Porter <jsporter@virginia.edu>
"""
import argparse
import datetime
import math
import os
import pdb
import sys
import tracemalloc
from collections import Counter

import numpy as np
from guppy import hpy
from tqdm import tqdm

# from memory_profiler import profile

BASE = 2


# @profile(precision=8)
def get_wiggle_from_MSA(
    MSA_file,
    seq_id,
    wig_output="entropy.wig",
    msa_output="msa.afa",
    debug=False,
):
    """
    Produce an MSA file and a wiggle entropy file from a given MSA file
    and reference sequence

    Parameters
    ----------
    MSA_file: str
        The location of the MSA input file.
    seq_id: str
        The reference id for the sequence in the MSA file.
    wig_output: str
        The location to write the output WIG file to.
    msa_output: str
        The location to write the output MSA file to.
    debug: bool
        Print extra info (columns in wiggle file) if true.


    Returns
    -------
    count: int
        A count of the columns in the final MSA
        after removing the gaps in the reference.

    """
    count = 0
    order = []
    curr_id = ""
    seq = ""
    total_size = 0
    with open(MSA_file) as msa:
        mb_size = 10 ** 6
        for line in msa:
            if line.startswith(";") or line.startswith("#"):
                continue
            if line.startswith(">"):
                if curr_id:
                    total_size += sys.getsizeof(curr_id) / mb_size
                    total_size += sys.getsizeof(seq) / mb_size
                    order.append((curr_id, seq))
                curr_id = line[1:].strip()
                seq = ""
                continue
            seq += line.strip()
    if curr_id:
        total_size += sys.getsizeof(curr_id) / mb_size
        total_size += sys.getsizeof(seq) / mb_size
        order.append((curr_id, seq))
    ref = [(i, rec) for i, rec in enumerate(order) if rec[0].startswith(seq_id)]
    if len(ref) <= 0:
        return 0
    p, query = ref[0]
    ref = None
    query_id, query_seq = query
    # output_d = {q_id: "" for q_id, _ in order}
    print(
        "The total size of the loaded file is {}MB.".format(total_size),
        file=sys.stderr,
    )
    del order[p]
    print("Creating wiggle entropy file.", file=sys.stderr, flush=True)
    with open(wig_output, "w") as wig:
        print("variableStep chrom={}".format(query_id), file=wig)
        skip = 0
        print(
            "Length of reference without gaps: {}\nLength of reference with gaps: {}".format(
                len(query_seq.replace("-", "")),
                len(query_seq),
            ),
            file=sys.stderr,
            flush=True,
        )
        for i, char in enumerate(tqdm(query_seq)):
            if i % 5000 == 1:
                wig.flush()
            if isgap(char):
                skip += 1
                continue
            count += 1
            col = [char]
            for curr_id, curr_seq in order:
                if curr_seq[i].upper() != "N":
                    col.append(curr_seq[i])
            if debug:
                print("{} {} {}".format(i - skip, entropy2(col), col), file=wig)
            else:
                print("{} {}".format(i - skip, entropy2(col)), file=wig)
    print("Creating afa file.", file=sys.stderr, flush=True)
    with open(msa_output, "w") as msa:
        out_seq = ""
        for i, char in enumerate(tqdm(query_seq)):
            if isgap(char):
                skip += 1
                continue
            out_seq += char
        print(">{}".format(query_id), file=msa)
        print(out_seq, file=msa)
        for curr_id, curr_seq in tqdm(order):
            skip = 0
            out_seq = ""
            for i, char in enumerate(tqdm(query_seq)):
                if isgap(char):
                    skip += 1
                    continue
                out_seq += curr_seq[i]
            print(">{}".format(curr_id), file=msa)
            print(out_seq, file=msa, flush=True)
    if debug:
        h = hpy()
        pdb.set_trace()
        print(h.heap())
    return count


def isgap(char):
    return char == "_" or char == "." or char == "~" or char == "-"


# def convertFasta(
#     msa_output="msa.afa", ids_temp=".ids.temp", msa_temp=".msa.temp", remove=True
# ):
#     with open(ids_temp) as ids_file, open(msa_output, "w") as msa:
#         ids = []
#         for line in ids_file:
#             if line.startswith(">"):
#                 ids.append(line.strip())
#         for i, record_id in tqdm(enumerate(ids)):
#             print(record_id, file=msa)
#             with open(msa_temp) as msa_file:
#                 j = 0
#                 while True:
#                     char = msa_file.read(1)
#                     if not char:
#                         break
#                     if j % len(ids) == i:
#                         msa.write(char)
#                     j += 1
#                 msa.write("\n")
#     if remove:
#         remove_file(ids_temp)
#         remove_file(msa_temp)


# def remove_file(file_location):
#     if os.path.exists(file_location):
#         os.remove(file_location)


def entropy1(seq, base=BASE, adj=100):
    """
    Calculate adjusted Shannon entropy from a sequence.

    Parameters
    ----------
    seq: str
        A sequence
    adj: int
        A constant to rescale Shannon entropy.

    Returns
    -------
    ent: int
        The adjusted entropy of the input.
        S = -100 * Sum (Pi * log2Pi)

    """
    counts = Counter(seq)
    base = BASE if base is None else base
    ent = -adj * sum(
        [
            ((counts[char] / len(seq)) * math.log(counts[char] / len(seq), base))
            for char in counts
        ]
    )
    return ent if ent > 0.0 else 0.0


def entropy2(labels, base=BASE, adj=100):
    """
    Computes entropy of label distribution.
    Modified from:
    https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
    """
    labels = list(labels)
    n_labels = len(labels)
    if n_labels <= 1:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0.0
    ent = 0.0
    # Compute entropy
    base = BASE if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)
    return adj * ent


def main():
    """Parse the arguments."""
    tick = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("MSA_file", help="Location of an MSA file.", type=str)
    parser.add_argument("seq_id", help="The sequence id of the reference.", type=str)
    parser.add_argument(
        "--msa_output",
        "-m",
        help="The location of the MSA output file.",
        type=str,
        default="msa.afa",
    )
    parser.add_argument(
        "--wig_output",
        "-w",
        help="The location of the WIG output file.",
        type=str,
        default="entropy.wig",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Use a breakpoint.", default=False
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Print extra memory usage info.",
        default=False,
    )
    map_args = parser.parse_args()
    print("Started MSA_to_wiggle at time {}.".format(tick), file=sys.stderr)
    print(map_args, file=sys.stderr, flush=True)
    if map_args.memory:
        tracemalloc.start()
    count = get_wiggle_from_MSA(
        map_args.MSA_file,
        map_args.seq_id,
        map_args.wig_output,
        map_args.msa_output,
        debug=map_args.debug,
    )
    if map_args.memory:
        current, peak = tracemalloc.get_traced_memory()
        print(
            "Current memory usage is {}MB; Peak was {}MB".format(
                current / 10 ** 6, peak / 10 ** 6
            ),
            file=sys.stderr,
        )
        tracemalloc.stop()
    print("There were {} columns extracted.".format(count), file=sys.stderr)
    print(
        "The process took {} time.".format(datetime.datetime.now() - tick),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
