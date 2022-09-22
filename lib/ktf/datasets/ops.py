def process_chain(input, **kwargs):
    """Runs processes on input

    Args:
        input: A tuple
        **kwargs: A dictionary with indices of input tuple as key and processes to run on indexed input as value

    Returns:
        A tuple of datasets that have been processed
    """

    def _get_slice_or_index(index_str):
        if ":" in index_str:
            index = slice(*map(lambda x: int(x.strip())
                               if x.strip() else None, index_str.split(':')))
        else:
            index = int(index_str.strip())
        return index

    output = list(input)
    for key, value in kwargs.items():
        for k in key.split(","):

            track_idx = range(len(output))

            slice_indices = _get_slice_or_index(k)

            slice_inputs = output[slice_indices]

            if type(slice_inputs) is not list:
                slice_inputs = [slice_inputs]

            track_idx = track_idx[slice_indices]

            if type(track_idx) is not range:
                track_idx = [track_idx]

            for idx, slice_input in zip(track_idx, slice_inputs):
                for proc in value:
                    slice_input = proc(slice_input)
                output[idx] = slice_input

    return tuple(output)


def select(input, indices):
    """Selects input datasets into another list of output datasets

    Args:
        input: A list/tuple of datasets
        indices: A list of indices into the input dataset list

    Returns:
        A tuple of datasets
    """
    return tuple([input[i] for i in indices])


def join(dataset_tuples, filter_none=True):
    """Gets each element in each tuple and puts them all in a tuple following the seq of the input list of tuples.
    Ignores None objects.

    Args:
        dataset_tuples: A list of tuples

    Returns:
        A tuple of datasets
    """
    dataset_list = []
    for dataset_tuple in dataset_tuples:
        for dataset in dataset_tuple:
            if dataset is None:
                if not filter_none:
                    dataset_list.append(None)
            else:
                dataset_list.append(dataset)
    return tuple(dataset_list)


def concatenate(dataset_tuples_list, batch_size, repeat=False, shuffle_size=None, drop_remainder=False):
    """Concatenates multiple datasets. Note: For each dataset to concatenate in the dataset_tuples_list, DO NOT repeat
    else it will never reach the other datasets other than the 1st one. Repeat should be done here.

    Args:
        dataset_tuples_list: List of TF Dataset objects tuple (train,val) to concatenate.
        batch_size: Size of each batch of elements
        repeat : (Optional) If concatenated ataset should repeat.
        shuffle_size : (Optional) To enable shuffling thougout the concatenated dataset, not within each dataset
        drop_remainder: (Optional) If True, the final batch of the dataset will be dropped if it has fewer than
                `batch_size` elements.
    Returns:
        A tuple of concatenated datasets (train,val)
    """
    concatenated_train = None
    concatenated_val = None

    for train_ds, val_ds in dataset_tuples_list:
        if train_ds is not None:
            concatenated_train = train_ds if concatenated_train is None else concatenated_train.concatenate(train_ds)
        if val_ds is not None:
            concatenated_val = val_ds if concatenated_val is None else concatenated_val.concatenate(val_ds)

    concatenated_train = concatenated_train.unbatch()
    concatenated_val = concatenated_val.unbatch()

    if shuffle_size is not None:
        concatenated_train = concatenated_train.shuffle(shuffle_size, reshuffle_each_iteration=True)
        concatenated_val = concatenated_val.shuffle(shuffle_size, reshuffle_each_iteration=True)

    if repeat:
        concatenated_train = concatenated_train.repeat()
        concatenated_val = concatenated_val.repeat()

    concatenated_train = concatenated_train.batch(batch_size, drop_remainder=drop_remainder)
    concatenated_val = concatenated_val.batch(batch_size, drop_remainder=drop_remainder)

    return concatenated_train, concatenated_val
