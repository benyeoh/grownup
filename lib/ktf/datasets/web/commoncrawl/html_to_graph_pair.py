#!/usr/bin/env python
import os
import sys
import multiprocessing
import optparse
import random
import hashlib
import ctypes

# Hack to allow this script to either be run independently or imported as a module
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../..'))

import ktf.datasets
import ktf.datasets.web.html_to_graph
from ktf.datasets.web.html_to_graph import FAIL, WARNING, ENDC
import ktf.datasets.web.commoncrawl
import ktf.datasets.web.commoncrawl.html_to_graph


def _worker_init(tag_features_args):
    """Internal function to initialize workers for multiprocessing
    """
    print("Loading tag features per worker...")
    ktf.datasets.web.html_to_graph.tag_features_model = ktf.datasets.web.TagFeatures(**tag_features_args)


def write_to_tfrecord_pairs(args):
    """Multiprocessing-friendly function to convert several html files to multiple tfrecord files
    given some arguments.

    Args:
        args: An argument list of:
            1. An integer ID for this process. Usually 0 if multiprocessing is not used.
            2. A list of tuples of (group, url, html file).
            3 & 4. Directories to (3) record processed files and TFRecords upon task completion, 
                (4) temporary save TFRecords for running task. 
            5. An integer number of tfrecord files to generate for the converted data.
            6. A boolean to indicate whether graph nodes should be sampled exclusively
                from tags containing text strings. If False, then graph nodes are sampled from all the
                tags in the DOM.
            7. A boolean to indicate whether inline css added to tag. If False, skip
            8. Number of iterations to expand the nodes in the graph.
            9. Maximum number of nodes the graph can contain
            10. The maximum number of neighbouring nodes that a node in the graph can have.    
    """
    id, all_inputs, out_dir, out_dir_tmp, num_splits, sample_text_nodes, inline_css, depth, max_num_nodes, max_neighbours = args

    print("%d: Creating %d record writers ..." % (id, num_splits))
    record_writers = [ktf.datasets.RecordWriter(os.path.join(out_dir_tmp, "%d_%d.tfrecord" % (i, id)),
                                                ktf.datasets.web.commoncrawl.FEATURE_DESC_GRAPH,
                                                use_compression=True) for i in range(num_splits)]

    total_count = 0
    num_errors = 0
    # Introduce html_processed list to save filename already been processed
    html_processed = []

    for group, urls, files in all_inputs:
        # Get a 64-bit deterministic hash from the group string for identifying each group later
        group_hash = int.from_bytes(hashlib.shake_256(group.encode("utf-8")).digest(8), byteorder="little")
        # Convert to signed 64-bit int
        group_hash = ctypes.c_long(group_hash).value
        print("%d: Group %s (%d)" % (id, group, group_hash))

        # Gather all tensors into 2 lists per file, maintaining similar length as much as possible
        l1 = []
        l2 = []
        cur_list = l1
        for i, orig_file in enumerate(files):
            print("%d: Parsing %s, URL: %s ..." % (id, orig_file, urls[i]))
            try:
                for j, (adj, feats, node_list) in enumerate(ktf.datasets.web.html_to_graph.html_to_graph_tensor(orig_file,
                                                                                                                sample_text_nodes,
                                                                                                                depth,
                                                                                                                max_num_nodes,
                                                                                                                max_neighbours,
                                                                                                                inline_css)):
                    cur_list.append((adj, feats, node_list))
                    if j == 10:
                        print("Already converted %d subgraphs. Skipping the rest ..." % j)
                        break

                cur_list = l1 if len(l1) < len(l2) else l2
                # Save a tuple (filename, url, group) to html_processed list
                html_processed.append((orig_file.split("/")[-1], urls[i], group))

            # Add FileNotFoundError in case extra file path are written in cdx-xxxxx.csv
            except (AssertionError, FileNotFoundError):
                num_errors += 1

        if len(l1) == 0 or len(l2) == 0:
            print((FAIL + "%d: Unable to form a pair. A list was empty. Skipping ..." + ENDC) % id)
            continue

        count = 0
        less_list, more_list = (l1, l2) if len(l1) < len(l2) else (l2, l1)

        for i, (adj1, feats1, node_list1) in enumerate(less_list):
            adj2, feats2, node_list2 = more_list[i]
            record_writers[total_count % num_splits].write((adj1.flatten().tolist(),
                                                            feats1.flatten().tolist(),
                                                            node_list1.tolist(),
                                                            group_hash,
                                                            adj2.flatten().tolist(),
                                                            feats2.flatten().tolist(),
                                                            node_list2.tolist(),
                                                            group_hash))
            count += 1
            total_count += 1

        # Now for the remaining data that has not been paired, we have to match with some repeat data
        for adj2, feats2, node_list2 in more_list[len(less_list):]:
            # Randomly sample feats from the first list
            adj1, feats1, node_list1 = random.choice(less_list)
            record_writers[total_count % num_splits].write((adj1.flatten().tolist(),
                                                            feats1.flatten().tolist(),
                                                            node_list1.tolist(),
                                                            group_hash,
                                                            adj2.flatten().tolist(),
                                                            feats2.flatten().tolist(),
                                                            node_list2.tolist(),
                                                            group_hash))
            count += 1
            total_count += 1

        print("%d: Written %s - %d records ..." % (id, orig_file, count))
        print("%d: Adj shape: %s, feat shape: %s, nodes shape: %s" % (id, adj.shape, feats.shape, node_list.shape))

    for record_writer in record_writers:
        record_writer.close()

    print()
    print((WARNING + "%d: Finished %d files. Written %d records ..." + ENDC) % (id, len(all_inputs), total_count))
    if num_errors > 0:
        print((FAIL + "%d: WARNING. %d errors encountered ..." + ENDC) % (id, num_errors))
    print()

    # Upon completion the task, write tuple (filename, url, group) into text file, each record is a line
    with open(os.path.join(out_dir, "processed.txt"), "a") as fd:
        for t in html_processed:
            line = ' '.join(str(x) for x in t)
            fd.write(line + '\n')
        fd.close


def write_html_to_graph_pairs(opt_args, html_inputs, tag_feats_model, graph_params):
    """Top level utility function to serializes html to a graph for unsupervised learning tasks
    (using multiprocessing if required)

    Args:
        opt_args: optparse arguments object
        html_inputs: A list of HTML file paths to serialize
        tag_feats_model: A `ktf.datasets.web.TagFeatures` object to convert tags into features
        graph_params: A tuple of `(depth, max_nodes, max_neighbours)` parameters for graph building
    """

    # This parameter is a global because we might want to use multi-processing
    # but we can never pickle this. Each process will have to initialize its own
    # and this is a convenient way to have the same codepath for both multi and single
    # process
    ktf.datasets.web.html_to_graph.tag_features_model = tag_feats_model
    tag_features_args = tag_feats_model.get_config()

    # Rearrange all input tuples into groups for easier consumption downstream
    all_orig = []
    for filepath, url, group in html_inputs:
        if len(all_orig) > 0 and group == all_orig[-1][0]:
            all_orig[-1][-2].append(url)
            all_orig[-1][-1].append(filepath)
        else:
            all_orig.append((group, [url], [filepath]))

    assert opt_args.num_tfrecords > 0, ".pkl output not supported. Only .tfrecord output is supported."
    if opt_args.num_proc > 1:
        to_process = []
        num_files_per_task = int(len(all_orig) / opt_args.num_proc)
        num_files_so_far = 0
        for i in range(opt_args.num_proc - 1):
            to_process.append(tuple([i,
                                     all_orig[num_files_so_far:num_files_so_far + num_files_per_task],
                                     opt_args.out_dir,
                                     opt_args.out_dir_tmp,
                                     opt_args.num_tfrecords,
                                     opt_args.sample_text_nodes,
                                     opt_args.inline_css] + graph_params))
            num_files_so_far += num_files_per_task

        to_process.append(tuple([opt_args.num_proc - 1,
                                 all_orig[num_files_so_far:],
                                 opt_args.out_dir,
                                 opt_args.out_dir_tmp,
                                 opt_args.num_tfrecords,
                                 opt_args.sample_text_nodes,
                                 opt_args.inline_css] + graph_params))

        with multiprocessing.Pool(opt_args.num_proc, initializer=_worker_init, initargs=(tag_features_args,)) as p:
            res = p.map(write_to_tfrecord_pairs, to_process)
            p.close()
            p.join()
    else:
        write_to_tfrecord_pairs(tuple([0,
                                       all_orig,
                                       opt_args.out_dir,
                                       opt_args.out_dir_tmp,
                                       opt_args.num_tfrecords,
                                       opt_args.sample_text_nodes,
                                       opt_args.inline_css] + graph_params))


if __name__ == "__main__":
    # Should be self-explanatory, but yeah cos' TF uses multi-threading semaphores internally and
    # would deadlock and cause all sorts of issues if we "fork" processes that already import
    # tensorflow. So we set it to spawn instead.
    print()
    print(WARNING + "WARNING: Forcing python multiprocessing to 'spawn' to ensure TF does not hang." + ENDC)
    print()
    multiprocessing.set_start_method("spawn")

    opt_parser = optparse.OptionParser()
    opt_args, args = ktf.datasets.web.commoncrawl.html_to_graph.initialize_options(opt_parser)
    tag_feats_model, graph_params = ktf.datasets.web.html_to_graph.initialize_config(opt_args)

    ktf.datasets.web.commoncrawl.html_to_graph.process_csv(write_html_to_graph_pairs,
                                                           opt_args,
                                                           tag_feats_model=tag_feats_model,
                                                           graph_params=graph_params)
