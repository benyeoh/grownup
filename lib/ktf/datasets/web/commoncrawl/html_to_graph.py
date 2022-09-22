#!/usr/bin/env python
import os
import glob
import sys
import multiprocessing
import optparse
import csv
import random
import shutil

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../..'))

import ktf.datasets
import ktf.datasets.web.html_to_graph
from ktf.datasets.web.html_to_graph import FAIL, WARNING, ENDC


def read_html_processed(text_path):
    html_processed = []
    if os.path.isfile(text_path):
        with open(os.path.join(text_path), "r") as fd:
            content = fd.readlines()
            for cur_item in content:
                html_processed.append(cur_item[:-1])
        fd.close()
    return html_processed


def parse_cdx_csv(out_dir, csv_path, max_groups=None):
    # Parses the generated cdx-xxxxx.csv file and returns the filename, url and group
    # from a random selection of groups up to `max_groups`

    # Introduce html_processed to save tuple (filename, url, group) of processed htmls
    if os.path.isfile(os.path.join(out_dir, "processed.txt")):
        html_processed = read_html_processed(os.path.join(out_dir, "processed.txt"))
    else:
        html_processed = []

    group_table = {}
    with open(csv_path, "r") as fd:
        reader = csv.reader(fd, delimiter="|")
        for filename, url, group in reader:
            if not group in group_table:
                group_table[group] = []
            group_table[group].append((filename, url))

        groups = list(group_table.keys())
        random.shuffle(groups)
        num_htmls_processed = 0
        for group in groups:
            # Divide num_htmls_processed by 2 to get the num_groups_processed
            if max_groups is None or int(num_htmls_processed / 2) < max_groups:
                for filename, url in group_table[group]:
                    if not (filename + " " + url + " " + group) in html_processed:
                        num_htmls_processed += 1
                        yield filename, url, group
            else:
                break


def initialize_options(opt_parser):
    opt_parser.add_option('--cdx-start', dest='cdx_start',
                          help='Start CDX index in the directory to process (according to `ls`)',
                          type='int', metavar='NUM', default=0)
    opt_parser.add_option('--cdx-end', dest='cdx_end',
                          help='End CDX index in the directory to process (according to `ls`)',
                          type='int', metavar='NUM', default=10000)
    opt_parser.add_option('--max-groups', dest='max_groups',
                          help='Maximum number of groups within a CDX to process',
                          type='int', metavar='NUM', default=100)
    opt_args, args = ktf.datasets.web.html_to_graph.initialize_options(opt_parser)
    return opt_args, args


def process_csv(process_html_fn, opt_args, **kwargs):
    all_cdx_csv = glob.glob(os.path.join(opt_args.input_dir, "*.csv"))
    total_count = 0
    orig_out_dir = opt_args.out_dir
    for i, csv_path in enumerate(all_cdx_csv[opt_args.cdx_start:opt_args.cdx_end + 1]):

        print("Parsing %d of %d: %s" % (i, len(all_cdx_csv[opt_args.cdx_start:opt_args.cdx_end + 1]), csv_path))
        csv_name = os.path.basename(csv_path).split(".")[0]

        # If all htmls in current csv have been processed
        # HACK: Modify the opt_args to output where I want
        opt_args.out_dir = os.path.join(orig_out_dir, csv_name)
        if os.path.exists(os.path.join(opt_args.out_dir, "complete.txt")):
            print(WARNING + "Skipping. `complete.txt` exists." + ENDC)
            print()
            continue

        # If there are remaining htmls in current csv have NOT been processed
        print("Processing CDX...%s" % csv_name)

        # Workaround: save TFRecord generated in running task into a new dir 'out_dir_tmp'.
        # This is to prevent overwriting TFRecord bcos they are named from idx 0 always.
        # TFRecord saved in out_dir_tmp, will be moved to out_dir upon completion of current run.
        # HACK: Modify the opt_args by adding out_dir_tmp
        opt_args.out_dir_tmp = os.path.join(orig_out_dir, csv_name + "_tmp")
        os.makedirs(opt_args.out_dir_tmp, exist_ok=True)
        os.makedirs(opt_args.out_dir, exist_ok=True)

        all_html_inputs = []
        for filename, url, group in parse_cdx_csv(opt_args.out_dir, csv_path, max_groups=opt_args.max_groups):
            html_path = os.path.join(opt_args.input_dir, csv_name, filename)
            all_html_inputs.append((html_path, url, group))

        if len(all_html_inputs) > 0:
            process_html_fn(opt_args, all_html_inputs, **kwargs)
        else:
            print(WARNING + "Empty CDX or all HTMLs have been processed." + ENDC)
            with open(os.path.join(opt_args.out_dir, "processed.txt"), "w") as fd:
                fd.close()

        # Upon completion of processing htmls, align .tfrecord filename between out_dir_tmp and out_dir
        cnt_tfrecords = len(glob.glob(os.path.join(opt_args.out_dir, "*.tfrecord")))
        cur_tfrecords_list = sorted(glob.glob(os.path.join(opt_args.out_dir_tmp, "*.tfrecord")))
        for filename in cur_tfrecords_list:
            # Rename FROM filename index in list (cur_tfrecords_list) TO "index + count" of tfrecords in out_dir
            # This will prevent overwriting due to the same name, regardless of how many past tasks and multi processing
            dst_idx = cur_tfrecords_list.index(filename) + cnt_tfrecords
            shutil.move(os.path.join(opt_args.out_dir_tmp, filename), os.path.join(
                opt_args.out_dir, str(dst_idx) + ".tfrecord"))
        os.rmdir(opt_args.out_dir_tmp)

        total_count += len(read_html_processed(os.path.join(opt_args.out_dir, "processed.txt")))
        print("Total: %d" % total_count)
        print()

        # Rename text file to complete.txt if all htmls have been processed
        if (len(read_html_processed(os.path.join(opt_args.out_dir, "processed.txt"))) ==
                len(glob.glob(os.path.join(opt_args.input_dir, csv_name, "*.html")))):
            os.rename(os.path.join(opt_args.out_dir, "processed.txt"),
                      os.path.join(opt_args.out_dir, "complete.txt"))

    print("Done!")


if __name__ == "__main__":
    # Should be self-explanatory, but yeah cos' TF uses multi-threading semaphores internally and
    # would deadlock and cause all sorts of issues if we "fork" processes that already import
    # tensorflow. So we set it to spawn instead.
    print()
    print(WARNING + "WARNING: Forcing python multiprocessing to 'spawn' to ensure TF does not hang." + ENDC)
    print()
    multiprocessing.set_start_method("spawn")

    opt_parser = optparse.OptionParser()
    opt_args, args = initialize_options(opt_parser)
    tag_feats_model, graph_params = ktf.datasets.web.html_to_graph.initialize_config(opt_args)

    # This scripts prepare dataset for single pre-train objective (without same website prediction)
    # Function below has not been updated to enable multiple runs with more html to be converted
    # Because it is also used in other dataset conversion (i.e cleaneval), which encourage convert all htmls in one run
    def _write_to_html_graph(opt_args, html_inputs, **kwargs):
        ktf.datasets.web.html_to_graph.write_html_to_graph(opt_args,
                                                           [path for path, _, _ in html_inputs],
                                                           **kwargs)
    process_csv(_write_to_html_graph,
                opt_args,
                tag_feats_model=tag_feats_model,
                graph_params=graph_params)
