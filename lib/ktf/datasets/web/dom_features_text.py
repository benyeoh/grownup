import numpy as np
import bs4
import nltk

import fasttext
import fasttext.util
import tensorflow_hub as hub
from scipy.sparse.csgraph import connected_components
from sentence_transformers import SentenceTransformer
import re


class Text2VecFasttext:
    def __init__(self,
                 ft_model_path,
                 truncate_wordvec=None,
                 **kwargs):
        """Wrapper for Fasttext word model

        Args:
            ft_model_path: Path to the Fasttext model
            truncate_wordvec: The size of the truncated feature vector. Default is None,
                which is the full sized feature vector
        """
        self._model_path = ft_model_path
        self._ft = fasttext.load_model(ft_model_path)
        self._truncate_wordvec = truncate_wordvec
        if truncate_wordvec is not None:
            fasttext.util.reduce_model(self._ft, truncate_wordvec)

    def get_vector_size(self):
        """Get size of vector embedding.

        Returns:
            An integer representing the size.
        """

        return self._ft.get_dimension()

    def _text_to_feature(self, tag):
        all_str = []
        for c in tag.find_all(string=True, recursive=False):
            if type(c) == bs4.element.NavigableString:
                all_str += str(c).split()
        sentence = " ".join(all_str)
        return {"embedding": self._get_sentence_vector(sentence).tolist(), "length": [len(sentence)]}

    def _get_sentence_vector(self, sentence):
        return self._ft.get_sentence_vector(sentence)

    def get_mean_word_vector(self, word_list_list):
        """Get mean word vector from a list of words.

        Args:
            word_list_list: A list of list of words.

        Returns:
            A list of mean word vectors
        """

        feat_list = []
        for word_list in word_list_list:
            feat = np.zeros(self._ft.get_dimension(), dtype=np.float32)
            for word in word_list:
                feat = feat + self._ft.get_word_vector(word)
            feat /= len(word_list)
            feat_list.append(feat)
        return feat_list

    def set_feats(self, tag_list, feats_list):
        """Extracts features from a list of HTML tags and sets it.

        Args:
            tag_list: A list of tags to extract features.
            feats_list: A list of extracted features to be modified. Each entry is a dictionary
                of features corresponding to each tag in `tag_list`.
        """

        for tag, feats in zip(tag_list, feats_list):
            feats["text"] = self._text_to_feature(tag)

    def get_config(self):
        return {
            "ft_model_path": self._model_path,
            "truncate_wordvec": self._truncate_wordvec
        }


class Text2VecUSE:
    def __init__(self,
                 use_model_path,
                 **kwargs):
        """Wrapper for Universal Sentence Encoder model

        Args:
            use_model_path: Path to the USE model
        """
        self._model_path = use_model_path
        self._embed = hub.load(use_model_path)

    def get_vector_size(self):
        """Get size of vector embedding.

        Returns:
            An integer representing the size.
        """

        return 512

    def _text_to_feature(self, tag):
        all_str = []
        for c in tag.find_all(string=True, recursive=False):
            # A try-except to handle ??
            try:
                if type(c) == bs4.element.NavigableString:
                    all_str += str(c).split()
            except ValueError as e:
                print(e)
                raise
        sentence = " ".join(all_str)
        return {"embedding": self._get_sentence_vector(sentence).tolist(), "length": [len(sentence)]}

    def _get_sentence_vector(self, sentence):
        # A try-except to handle unicode conversion error
        '''
        ***Error description: if surrogate pair appears in html_doc, it can parsed by bs4 and generate a soup object.
        However, when USE attempts to vectorize the sentence with surrogate pair, it is not convertible to tensor.
        Raise ValueError and then in function html_to_graph_tensor, skip this html with filename highlighted

        ***Effort to fix incl.:
        - in html_doc, identify surrogate pair using regex (not reporting)
        - workaround to encode by utf-16, decode with error handler "surrogatepass" (not working)

        ***Next Break,
        - able to convert surrogate pair to convertable utf-8 string (?)
        '''

        try:
            res = self._embed(sentence)
        except ValueError as e:
            print(e)
            raise
        if isinstance(sentence, list):
            return np.array(res)
        else:
            return np.array(res[0])

    def get_mean_word_vector(self, word_list_list):
        """Get size of vector embedding.

        Returns:
            An integer representing the size.
        """

        feat_list = []
        for word_list in word_list_list:
            feat_list.append(self._get_sentence_vector(" ".join(word_list)).tolist())
        return feat_list

    def set_feats(self, tag_list, feats_list):
        """Extracts features from a list of HTML tags and sets it.

        Args:
            tag_list: A list of tags to extract features.
            feats_list: A list of extracted features to be modified. Each entry is a dictionary
                of features corresponding to each tag in `tag_list`.
        """

        for tag, feats in zip(tag_list, feats_list):
            feats["text"] = self._text_to_feature(tag)

    def get_config(self):
        return {
            "use_model_path": self._model_path
        }


class _LexRank:
    """Taken from https://github.com/crabcamp/lexrank.
    Internal class used to determine the connectivity of sentences with other sentences in a text
    """

    @classmethod
    def _power_method(cls, transition_matrix, increase_power=True):
        eigenvector = np.ones(len(transition_matrix))

        if len(eigenvector) == 1:
            return eigenvector
        transition = transition_matrix.transpose()
        while True:
            eigenvector_next = np.dot(transition, eigenvector)
            if np.allclose(eigenvector_next, eigenvector):
                return eigenvector_next
            eigenvector = eigenvector_next
            if increase_power:
                transition = np.dot(transition, transition)

    @classmethod
    def connected_nodes(cls, matrix):
        _, labels = connected_components(matrix)

        groups = []

        for tag in np.unique(labels):
            group = np.where(labels == tag)[0]
            groups.append(group)

        return groups

    @classmethod
    def create_markov_matrix(cls, weights_matrix):
        n_1, n_2 = weights_matrix.shape
        if n_1 != n_2:
            raise ValueError('\'weights_matrix\' should be square')

        row_sum = weights_matrix.sum(axis=1, keepdims=True)

        return weights_matrix / row_sum

    @classmethod
    def create_markov_matrix_discrete(cls, weights_matrix, threshold):
        discrete_weights_matrix = np.zeros(weights_matrix.shape)
        ixs = np.where(weights_matrix >= threshold)
        discrete_weights_matrix[ixs] = 1

        return _LexRank.create_markov_matrix(discrete_weights_matrix)

    @classmethod
    def graph_nodes_clusters(cls, transition_matrix, increase_power=True):
        clusters = _LexRank.connected_nodes(transition_matrix)
        clusters.sort(key=len, reverse=True)

        centroid_scores = []

        for group in clusters:
            t_matrix = transition_matrix[np.ix_(group, group)]
            eigenvector = _LexRank._power_method(t_matrix, increase_power=increase_power)
            centroid_scores.append(eigenvector / len(group))

        return clusters, centroid_scores

    @classmethod
    def stationary_distribution(cls, transition_matrix, increase_power=True, normalized=True):
        n_1, n_2 = transition_matrix.shape
        if n_1 != n_2:
            raise ValueError('\'transition_matrix\' should be square')

        distribution = np.zeros(n_1)

        grouped_indices = _LexRank.connected_nodes(transition_matrix)

        for group in grouped_indices:
            t_matrix = transition_matrix[np.ix_(group, group)]
            eigenvector = _LexRank._power_method(t_matrix, increase_power=increase_power)
            distribution[group] = eigenvector

        if normalized:
            distribution /= n_1

        return distribution

    @classmethod
    def degree_centrality_scores(cls, similarity_matrix, threshold=None, increase_power=True):
        if not (threshold is None or isinstance(threshold, float) and 0 <= threshold < 1):
            raise ValueError(
                '\'threshold\' should be a floating-point number '
                'from the interval [0, 1) or None',
            )

        if threshold is None:
            markov_matrix = _LexRank.create_markov_matrix(similarity_matrix)
        else:
            markov_matrix = _LexRank.create_markov_matrix_discrete(similarity_matrix, threshold)

        scores = _LexRank.stationary_distribution(markov_matrix, increase_power=increase_power, normalized=False)

        return scores


class _SentenceTokenizer:
    """Tokenizes sentences using nltk. Tokenization uses only the English language features
    with some basic support for Chinese punctuation
    """

    def __init__(self, nltk_data="/hpc-datasets/ext_models/nltk"):
        # Load from internal resource preferably
        if nltk_data not in nltk.data.path:
            nltk.data.path.append(nltk_data)

    def tokenize(self, text):
        # Make this support other languages?
        sentences = nltk.tokenize.sent_tokenize(text)
        # Somewhat support Chinese punctuation
        return [s for sentence in sentences for s in re.findall(u'[^！？。]+[!？。]?', sentence, flags=re.U)]


class TextSentenceTransformer:
    """SentenceTransformer model wrapper. See: https://www.sbert.net/
    """

    def __init__(self, model_path, nltk_data_path, max_sequence_length=128, **kwargs):
        self._model_path = model_path
        self._nltk_data_path = nltk_data_path
        self._model = SentenceTransformer(model_path)
        self._sentence_tokenizer = _SentenceTokenizer(nltk_data=nltk_data_path)
        self._max_sequence_length = max_sequence_length

    def _get_text_from_tag(self, tag):
        all_str = []
        for c in tag.find_all(string=True, recursive=False):
            if type(c) == bs4.element.NavigableString:
                all_str += str(c).split()
        text = " ".join(all_str)
        return text

    def _summarize_text(self, text, num_sentences=3):
        # We split the text into individual sentences
        sentences = self._sentence_tokenizer.tokenize(text)
        # Then we encode each sentence
        encoded = self._model.encode(sentences)
        if str(list(self._model.children())[-1]) != "Normalize()":
            # Normalize the embeddings if they are not already normalized
            encoded = encoded / np.expand_dims(np.linalg.norm(encoded, axis=-1), axis=-1)
        # Get the similarity matrix of each sentence normalized from 0-1
        similarity_mat = np.dot(encoded, encoded.T) * 0.5 + 0.5
        # Compute the centrality of each sentence (see _LexRank)
        centrality = _LexRank.degree_centrality_scores(similarity_mat)
        # Select the top k sentences for summary and encode the summary
        summary = ""
        selected_indices = np.argsort(centrality)[::-1][0:num_sentences]
        for i, s in enumerate(sentences):
            if i in selected_indices:
                summary += s + " "
        return summary.strip()

    def _text_to_feature(self, texts):
        token_lengths = [len(self._model.tokenize(texts=[text])["input_ids"][0]) for text in texts]
        start = 0
        encoded_texts = []
        for i, token_length in enumerate(token_lengths):
            # If it exceeds the user specified max length or the model max length
            if (token_length >= self._model.get_max_seq_length() or
                    token_length >= self._max_sequence_length):
                # Then encode all the texts up till now
                if i > start:
                    encoded_texts.extend(self._model.encode(texts[start:i]))
                # And encode the current summarized text separately
                encoded_texts.append(self._model.encode(self._summarize_text(texts[i])))
                start = i + 1

        if start <= len(token_lengths) - 1:
            encoded_texts.extend(self._model.encode(texts[start:]))

        res = []
        for text, encoded in zip(texts, encoded_texts):
            res.append({"embedding": encoded.tolist(), "length": [len(text)]})
        return res

    def get_vector_size(self):
        """Get size of vector embedding.

        Returns:
            An integer representing the size.
        """

        return self._model.get_sentence_embedding_dimension()

    def get_mean_word_vector(self, word_list_list):
        """Get the mean word vector from a list of words.

        Args:
            word_list_list: A list of list of words to get feature embeddings.

        Returns:
            A list of feature embeddings
        """

        sentence_list = []
        for word_list in word_list_list:
            sentence_list.append(" ".join(word_list))

        return [f["embedding"] for f in self._text_to_feature(sentence_list)]

    def set_feats(self, tag_list, feat_list):
        """Extracts features from a list of HTML tags and sets it.

        Args:
            tag_list: A list of tags to extract features.
            feat_list: A list of extracted features to be modified. Each entry is a dictionary
                of features corresponding to each tag in `tag_list`.
        """

        sentence_list = []
        for tag in tag_list:
            sentence_list.append(self._get_text_from_tag(tag))

        for text_feat, feats in zip(self._text_to_feature(sentence_list), feat_list):
            feats["text"] = text_feat

    def get_config(self):
        return {
            "model_path": self._model_path,
            "nltk_data_path": self._nltk_data_path,
            "max_sequence_length": self._max_sequence_length
        }
