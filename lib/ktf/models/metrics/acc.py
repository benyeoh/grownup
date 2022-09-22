import tensorflow as tf
from ...datasets.actions.classification.auxiliary_word2vec import classes2embedding


def get_class_embeddings_and_num_classes(label_file, dataset_name, wv_model):
    """ A function to retrieve class embeddings and number of classes for action classification dataset

    This function was written for datasets in ktf.datasets.actions.classification
    1. Kinetics700
    2. UCF101
    3. HMDB51

    Args:
        label_file: Path to label file of corresponding dataset.
                    1. Kinetics700: /hpc-datasets/action_recognition/kinetics_700_2020/tfrecord/labels.txt
                    2. UCF101: /hpc-datasets/action_recognition/UCF-101/tfrecord/labels.txt
                    3. HMDB51: /hpc-datasets/action_recognition/hmdb51_sta/tfrecord/labels.txt
        dataset_name: Name of dataset. Supported options are 'kinetics', 'ucf101', 'hmdb51'.
        wv_model: A word2vec model loaded from ktf.datasets.actions.classification.auxiliary_word2vec.load_word2vec()
    """
    # Get labels first
    f = open(label_file, "r")
    labels_idx_w_labels = f.read().split('\n')
    # File might have an extra '\n' at the end thus creating an empty element
    while(len(labels_idx_w_labels[-1]) == 0):
        labels_idx_w_labels = labels_idx_w_labels[:-1]
    labels = []
    for label_idx_w_label in labels_idx_w_labels:
        label_splits = label_idx_w_label.split(',')
        label = label_splits[1]
        labels.append(label)

    # Get label_embeddings
    embeddings = classes2embedding(dataset_name, labels, wv_model)

    return embeddings, len(labels)


class FeatureVectorBasedSparseTopKCategoricalAccuracy(tf.keras.metrics.SparseTopKCategoricalAccuracy):
    """ Sparse Top K Categorical Accuracy measure for zero shot classification

    This class is a "wrapper" around tf.keras.metrics.SparseTopKCategoricalAccuracy.

    The difference btw this class and the parent lies in the
    1. pre-process of matching network output embedding with the respective dataset class
    embeddings to retrieve the top k predicted classes and then
    2. updating the state of the parent metric with the predicted and true classes

    How this class works:
        Init Input:
            1. List of tuples of dataset properties
                e.g. [(dataset_0_class_embeddings, dataset_0_num_classes),
                      ...,
                      (dataset_n_class_embeddings, dataset_n_num_classes)]
            2. ground_truth_label_idx
                e.g. y_true is expected to be of shape (batch_size, num_label in target, label_emb_len). an example
                would be y_true containing
                    1. label index
                    2. label embedding
                    3. dataset index indicating which dataset this data belongs to
                Specify the index int in `num_label in target` that corresponds to label index. In the example, its 0

            3. ground_truth_dataset_idx
                e.g. y_true is expected to be of shape (batch_size, num_label in target, label_emb_len). an example
                would be y_true containing
                    1. label index
                    2. label embedding
                    3. dataset index indicating which dataset this data belongs to
                Specify the index int in `num_label in target` that corresponds to dataset index. In the example, its 2

            4. Value of k

        Update State Process:
            1. Receive y_pred i.e. network output embedding

            2. Receive y_true with shape (batch_size, num_label in target, word_emb_len) which contains among other
               label details:
               1. Label index
               2. Dataset index indicating which dataset this data belongs to ----- <idx>
               for each data point target in the batch

               In the case of zero shot video action classification, we have y_true containing
               1. label index
               2. label embedding
               3. dataset index indicating which dataset this data belongs to

            3. Compare y_pred with dataset_<idx>_class_embeddings and retrieve top k closest classes

            4. update_state using parent class (which is the usual sparse top k categorical accuracy) with predicted and
               true classes

            5. Retrieve result from parent class (which is the usual sparse top k categorical accuracy)
    """

    def __init__(self,
                 datasets_properties,
                 ground_truth_label_idx,
                 ground_truth_dataset_idx,
                 k=5,
                 name='zero_shot_sparse_top_k_categorical_accuracy',
                 **kwargs):
        """ Init the class

        Args:
            datasets_properties: List of tuples of dataset class embeddings and num classes. The ordering in this list
                                 matters as the position will be used to match the network output embedding to the right
                                 dataset class embeddings to compare against

                                 e.g.

                                 datasets_properties = [(kinetics_class_embeddings, kinetics_num_classes),
                                                        (ucf101_class_embeddings, ucf101_num_classes),
                                                        (hmdb51_class_embeddings, hmdb51_num_classes)]

                                 Assume y_true is of shape (batch_size, num_label in target, word_emb_len) and contains
                                 1. label index
                                 2. label embedding
                                 3. dataset index indicating which dataset this data belongs to
                                 for each data point target in the batch

                                 Assume y_pred is network output embedding of shape (batch_size, word_emb_len)

                                 If dataset index in y_true is 0, then y_pred will be compared against
                                 kinetics_class_embeddings which is in the tuple at position 0 of the dataset_properties
                                 list

                                 Similarly, if dataset index in y_true is 1, then y_pred will be compared against
                                 ucf101_class_embeddings which is in the tuple at position 1 of the dataset_properties
                                 list

                                 Similarly, if dataset index in y_true is 2, then y_pred will be compared against
                                 hmdb51_class_embeddings which is in the tuple at position 2 of the dataset_properties
                                 list

                                 The same applies for the dataset_num_classes. The corresponding dataset num_classes
                                 will be used to generate probability vectors for update_state process
            ground_truth_label_idx: y_true is expected to be of shape (batch_size, num_label in target, label_emb_len).
                                     Specify the index int in `num_label in target` that corresponds to label index
            ground_truth_dataset_idx: y_true is expected to be of shape
                                      (batch_size, num_label in target, label_emb_len). Specify the index int in
                                      `num_label in target` that corresponds to dataset index
            k: (Optional) Number of top matches to be used for checking top k accuracy
            name: (Optional) Metric name
        """
        super(FeatureVectorBasedSparseTopKCategoricalAccuracy, self).__init__(k=k, name=name, **kwargs)
        self._embeddings = tf.ragged.constant([dataset_properties[0] for dataset_properties in datasets_properties])
        self._num_classes = tf.convert_to_tensor([dataset_properties[1] for dataset_properties in datasets_properties])
        self._gt_label_idx = ground_truth_label_idx
        self._gt_ds_idx = ground_truth_dataset_idx
        self._k = k

    def _cosine_similarity_btw_2_batched_normalized_vectors(self, a, b):
        """Calculates cosine distances between batch of inferred feature vectors and dataset class embeddings

        Args:
            a: Batch of inferred feature vectors of shape [batch_size, embedding length]
            b: Dataset class embeddings [num_classes, embedding_length]
        """
        b = tf.transpose(b)
        dot = tf.tensordot(a, b, axes=1)

        return dot

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Args:
            y_true: Tensor of shape (batch_size, num_label in target, label_emb_len)
            y_pred: Tensor of shape (batch_size, label_emb_len)
            sample_weight: (Optional) Weighting to apply for each sample.
        """
        # Example:
        # In zero shot video action classification, we have num_label in target is 3
        # 1. label index
        # 2. label embedding
        # 3. dataset index indicating which dataset this data belongs to
        #
        # Assume we have word_emb_len = 300
        #
        # So... For each data point target in batch:
        # 1. First elem is label index (replicated 300 times)
        # 2. Second elem is label embedding vector of length 300
        # 3. Third elem is dataset index indicating which dataset this data belong to (replicated 300 times)

        # We want label index here
        # Get label index from label "tuple"
        #
        # And since label index is duplicated, we only need first elem for each element in batch
        #
        # Then cast label index to int
        true_class_idx = tf.cast(y_true[:, self._gt_label_idx][:, 0], tf.int32)

        # We want dataset index here
        # Get dataset index from label "tuple"
        #
        # And since dataset index is duplicated for each element in the batch,
        # and all the elements in the batch are from the same dataset
        # we only need first elem
        #
        # Then cast label index to int
        dataset_idx = tf.cast(y_true[:, self._gt_ds_idx][0, 0], tf.int32)

        # Get the relevant dataset class embeddings as a tensor
        embeddings = self._embeddings[dataset_idx].to_tensor()

        # Retrieve top k closest classes
        _, pred_class_idxs = tf.math.top_k(
            self._cosine_similarity_btw_2_batched_normalized_vectors(y_pred, embeddings), k=self._k)

        # Get the relevant num classes
        num_classes = self._num_classes[dataset_idx]

        # Generate probability output to be used in update_state
        if self._k == 1:
            pred_class_probabilities = tf.one_hot(indices=pred_class_idxs, depth=num_classes)
            pred_class_probabilities = tf.squeeze(pred_class_probabilities, axis=1)
        else:
            pred_class_probabilities = tf.one_hot(indices=pred_class_idxs, depth=num_classes)
            pred_class_probabilities = tf.math.reduce_sum(pred_class_probabilities, axis=1)
            pred_class_probabilities = tf.math.truediv(pred_class_probabilities, self._k)

        super(FeatureVectorBasedSparseTopKCategoricalAccuracy, self).update_state(
            true_class_idx, pred_class_probabilities, sample_weight=sample_weight)

    def result(self):
        return super(FeatureVectorBasedSparseTopKCategoricalAccuracy, self).result()


class FeatureVectorBasedSparseCategoricalAccuracy(FeatureVectorBasedSparseTopKCategoricalAccuracy):
    """ Sparse Categorical Accuracy measure for zero shot classification

    This class is a "wrapper" around FeatureVectorBasedSparseTopKCategoricalAccuracy with k = 1.

    For more details, refer to docstring for FeatureVectorBasedSparseTopKCategoricalAccuracy
    """

    def __init__(self,
                 datasets_properties,
                 ground_truth_label_idx,
                 ground_truth_dataset_idx,
                 name='feature_vector_sparse_categorical_accuracy',
                 **kwargs):
        """ Init the class

        Args:
            datasets_properties: List of tuples of dataset class embeddings and num classes. The ordering in this list
                                 matters as the position will be used to match the network output embedding to the right
                                 dataset class embeddings to compare against

                                 e.g.

                                 datasets_properties = [(kinetics_class_embeddings, kinetics_num_classes),
                                                        (ucf101_class_embeddings, ucf101_num_classes),
                                                        (hmdb51_class_embeddings, hmdb51_num_classes)]

                                 Assume y_true is of shape (batch_size, num_label in target, word_emb_len) and contains
                                 1. label index
                                 2. label embedding
                                 3. dataset index indicating which dataset this data belongs to
                                 for each data point target in the batch

                                 Assume y_pred is network output embedding of shape (batch_size, word_emb_len)

                                 If dataset index in y_true is 0, then y_pred will be compared against
                                 kinetics_class_embeddings which is in the tuple at position 0 of the dataset_properties
                                 list

                                 Similarly, if dataset index in y_true is 1, then y_pred will be compared against
                                 ucf101_class_embeddings which is in the tuple at position 1 of the dataset_properties
                                 list

                                 Similarly, if dataset index in y_true is 2, then y_pred will be compared against
                                 hmdb51_class_embeddings which is in the tuple at position 2 of the dataset_properties
                                 list

                                 The same applies for the dataset_num_classes. The corresponding dataset num_classes
                                 will be used to generate probability vectors for update_state process
             ground_truth_label_idx: y_true is expected to be of shape (batch_size, num_label in target, label_emb_len).
                                     Specify the index int in `num_label in target` that corresponds to label index
             ground_truth_dataset_idx: y_true is expected to be of shape
                                       (batch_size, num_label in target, label_emb_len). Specify the index int in
                                       `num_label in target` that corresponds to dataset index
            name: (Optional) Metric name
        """
        super(FeatureVectorBasedSparseCategoricalAccuracy, self).__init__(datasets_properties=datasets_properties,
                                                                          ground_truth_label_idx=ground_truth_label_idx,
                                                                          ground_truth_dataset_idx=ground_truth_dataset_idx,
                                                                          k=1,
                                                                          name=name,
                                                                          **kwargs)
