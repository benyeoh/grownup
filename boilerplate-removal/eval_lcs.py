#!/usr/bin/env python
import os
import glob
import argparse
import difflib
import multiprocessing


_all_tp = None
_all_fp = None
_all_fn = None

_all_prec = None
_all_recall = None
_all_total = None


def _worker_init(all_tp, all_fp, all_fn, all_prec, all_recall, all_total):
    global _all_tp
    global _all_fp
    global _all_fn

    _all_tp = all_tp
    _all_fp = all_fp
    _all_fn = all_fn

    global _all_prec
    global _all_recall
    global _all_total
    _all_prec = all_prec
    _all_recall = all_recall
    _all_total = all_total


def _get_token_text(filepath):
    with open(filepath, "r", encoding="utf-8") as fd:
        text = fd.read()
        return text.split()


def _get_confusion_matrix(filepath_pred, filepath_gt):
    pred_text = _get_token_text(filepath_pred)
    gt_text = _get_token_text(filepath_gt)
    tp = 0
    fp = 0
    fn = 0

    seq_matcher = difflib.SequenceMatcher(None, pred_text, gt_text, autojunk=False)
    for tag, i1, i2, j1, j2 in seq_matcher.get_opcodes():
        if tag == "delete":
            fp += i2 - i1
        elif tag == "equal":
            tp += i2 - i1
            assert (i2 - i1) == (j2 - j1)
        elif tag == "replace":
            fp += i2 - i1
            fn += j2 - j1
        elif tag == "insert":
            fn += j2 - j1
        else:
            raise ValueError("Unknown op: %s" % tag)

    return tp, fp, fn


def _compute_precision_recall_f1(num_tp, num_fp, num_fn):
    if num_tp == 0:
        prec = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        prec = float(num_tp) / (num_tp + num_fp)
        recall = float(num_tp) / (num_tp + num_fn)
        f1 = 2.0 * (prec * recall) / (prec + recall)
    return prec, recall, f1


def _worker_compute_lcs(args):
    i, to_process = args

    _all_tp[i] = 0
    _all_fp[i] = 0
    _all_fn[i] = 0

    for extracted_path, gt_path in to_process:
        tp, fp, fn = _get_confusion_matrix(extracted_path, gt_path)
        _all_tp[i] += tp
        _all_fp[i] += fp
        _all_fn[i] += fn


# @numba.jit(nopython=True, nogil=True)
# def lcs_length(a, b):
#     """ Longest common subsequence """
#     if len(a) == 0 or len(b) == 0:
#         return 0

#     res = max(
#         lcs_length(a[:-1], b),
#         lcs_length(a, b[:-1])
#     )

#     if a[-1] == b[-1]:
#         res = max(res, lcs_length(a[:-1], b[:-1]) + 1)

#     return res


def _get_prec_recall(filepath_pred, filepath_gt):
    pred_text = _get_token_text(filepath_pred)
    gt_text = _get_token_text(filepath_gt)

    # def lcs_length(a, b):
    #     table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    #     for i, ca in enumerate(a, 1):
    #         for j, cb in enumerate(b, 1):
    #             table[i][j] = (
    #                 table[i - 1][j - 1] + 1 if ca == cb else
    #                 max(table[i][j - 1], table[i - 1][j]))
    #     return table[-1][-1]

    # def longest_common_subsequence_recursive_memoized_mit(A, B):
    #     N = len(A)
    #     M = len(B)
    #     if N <= 0 or M <= 0:
    #         return []
    #     res_matrix = [[None] * M for i in range(N)]

    #     def lcs(i, j):
    #         if i <= -1 or j <= -1:
    #             return []
    #         if res_matrix[i][j] == None:
    #             if A[i] == B[j]:
    #                 res_matrix[i][j] = lcs(i - 1, j - 1) + [A[i]]
    #             else:
    #                 prev1 = lcs(i - 1, j)
    #                 prev2 = lcs(i, j - 1)
    #                 res_matrix[i][j] = prev1 if (
    #                     len(prev1)
    #                     >
    #                     len(prev2)
    #                 ) else prev2
    #         return res_matrix[i][j]

    #     return lcs(N - 1, M - 1)

    seq_matcher = difflib.SequenceMatcher(None, pred_text, gt_text, autojunk=False)
    num_match = 0
    for block in seq_matcher.get_matching_blocks():
        num_match += block.size

    # num_match = lcs_length(pred_text, gt_text)

    # num_match = max(len(pred_text), len(gt_text)) - editdistance.eval(pred_text, gt_text)
    # assert num_match >= 0

    # for tag, i1, i2, j1, j2 in seq_matcher.get_opcodes():
    #     if tag == "equal":
    #         num_match += i2 - i1
    #         assert (i2 - i1) == (j2 - j1)

    prec = 1.0
    if len(pred_text) > 0:
        prec = float(num_match) / float(len(pred_text))

    recall = 1.0
    if len(gt_text) > 0:
        recall = float(num_match) / float(len(gt_text))
    return prec, recall


def _worker_compute_lcs_per_page(args):
    i, to_process = args

    _all_prec[i] = 0.0
    _all_recall[i] = 0.0
    _all_total[i] = 0

    for extracted_path, gt_path in to_process:
        prec, recall = _get_prec_recall(extracted_path, gt_path)
        _all_prec[i] += prec
        _all_recall[i] += recall
        _all_total[i] += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', dest='extracted_dir',
                        help='Directory containing extracted text', metavar='DIR',
                        default=None)
    parser.add_argument('-g', dest='gt_dir', help='Directory to ground-truth text', metavar='DIR',
                        default=None)
    # parser.add_argument('--per-webpage', dest='per_webpage', help='Directory to ground-truth text', metavar='DIR',
    #                     default=None)

    args = parser.parse_args()

    all_gt = glob.glob(os.path.join(args.gt_dir, "*.txt"))
    all_extracted = sorted(glob.glob(os.path.join(args.extracted_dir, "*.txt")))

    gt_map = {}
    for gt in all_gt:
        gt_map[os.path.basename(gt).split(".")[0]] = gt

    all_extracted_set = set()

    computed = set()
    skipped = set()
    to_process = []
    for extracted in all_extracted:
        name = os.path.basename(extracted).split(".")[0]
        all_extracted_set.add(name)
        if name in gt_map:
            assert extracted not in computed
            computed.add(extracted)
            to_process.append((extracted, gt_map[name]))
        else:
            assert extracted not in skipped
            print("Skipped: %s" % extracted)
            skipped.add(extracted)

    num_cores = multiprocessing.cpu_count() // 2
    num_per_process = len(to_process) // num_cores
    num_processes = num_cores + 1 if len(to_process) > num_cores * num_per_process else num_cores

    all_process = [(i, to_process[i * num_per_process:(i + 1) * num_per_process])
                   for i in range(num_processes)]

    all_tp = multiprocessing.Array("i", num_processes)
    all_fp = multiprocessing.Array("i", num_processes)
    all_fn = multiprocessing.Array("i", num_processes)

    all_prec = multiprocessing.Array("f", num_processes)
    all_recall = multiprocessing.Array("f", num_processes)
    all_total = multiprocessing.Array("i", num_processes)

    with multiprocessing.Pool(num_cores, initializer=_worker_init, initargs=(all_tp, all_fp, all_fn, all_prec, all_recall, all_total)) as p:
        res = p.map(_worker_compute_lcs_per_page, all_process)
        p.close()
        p.join()

    print("Processed: %d, Skipped: %d" % (len(computed), len(skipped)))
    not_in_extracted = set(gt_map.keys()) - all_extracted_set
    print("Not in extracted: %d, %s" % (len(not_in_extracted), list(not_in_extracted)))

    prec = 0.0
    recall = 0.0
    total = 0
    for i in range(num_processes):
        prec += all_prec[i]
        recall += all_recall[i]
        total += all_total[i]

    prec = prec / total
    recall = recall / total
    f1 = 2 * prec * recall / (prec + recall)
    print("Precision: %.5f, Recall: %.5f, F1: %.5f" % (prec, recall, f1))

    # # Compute micro-f1
    # num_tp = 0
    # num_fp = 0
    # num_fn = 0

    # for i in range(num_processes):
    #     num_tp += all_tp[i]
    #     num_fp += all_fp[i]
    #     num_fn += all_fn[i]

    # prec, recall, f1 = _compute_precision_recall_f1(num_tp, num_fp, num_fn)
    # print("Precision: %.5f, Recall: %.5f, F1: %.5f" % (prec, recall, f1))
