import sys
import traceback

import hjson
import numpy as np
from bs4 import BeautifulSoup
import scipy.sparse.linalg
import networkx.linalg.laplacianmatrix
import networkx.generators.lattice
import tensorflow_text

from .dom_parser import to_graph
from .dom_features_text import Text2VecUSE, TextSentenceTransformer
from .dom_features_font import FontSimple, FontCompStyles
from .dom_features_visual import VisualSimple


class TagFeatures:
    """Utility class for extracting features from HTML DOM tags.

    Features include:

    1. Tag class
    2. Tag width/height
    3. Tag text/length
    4. Tag font style/size/weight
    """

    # All HTML tags that we consider for feature extraction. Length 152
    # There are previous options for tag list, i.e. dragnet_tags.txt (length 84) for custom tasks
    HTML_TAGS = [
        "a",
        "abbr",
        "acronym",
        "address",
        "applet",
        "area",
        "article",
        "aside",
        "audio",
        "b",
        "base",
        "basefont",
        "bdi",
        "bdo",
        "big",
        "blockquote",
        "body",
        "br",
        "button",
        "canvas",
        "caption",
        "center",
        "cite",
        "code",
        "col",
        "colgroup",
        "command",
        "data",
        "datagrid",
        "datalist",
        "dd",
        "del",
        "details",
        "dfn",
        "dialog",
        "dir",
        "div",
        "dl",
        "dt",
        "em",
        "embed",
        "fieldset",
        "figcaption",
        "figure",
        "font",
        "footer",
        "form",
        "frame",
        "frameset",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "head",
        "header",
        "hr",
        "html",
        "i",
        "iframe",
        "img",
        "input",
        "ins",
        "kbd",
        "label",
        "legend",
        "li",
        "link",
        "main",
        "map",
        "mark",
        "meta",
        "meter",
        "nav",
        "nobr",
        "noframes",
        "noscript",
        "object",
        "ol",
        "optgroup",
        "option",
        "output",
        "p",
        "param",
        "picture",
        "pre",
        "progress",
        "q",
        "rp",
        "rt",
        "ruby",
        "s",
        "samp",
        "script",
        "section",
        "select",
        "small",
        "source",
        "spacer",
        "span",
        "strike",
        "strong",
        "style",
        "sub",
        "summary",
        "sup",
        "svg",
        "table",
        "tbody",
        "td",
        "template",
        "text",
        "textarea",
        "tfoot",
        "th",
        "thead",
        "time",
        "title",
        "tr",
        "track",
        "tt",
        "u",
        "ul",
        "var",
        "video",
        "wbr",

        # New
        "path",
        "g",
        "circle",
        "use",
        "defs",
        "rect",
        "symbol",
        "polygon",
        "stop",
        "line",
        "lineargradient",
        "i-amphtml-sizer",
        "amp-img",
        "amp-analytics",
        "mask",
        "amp-pixel",
        "amp-ad",
        "amp-fit-text",
        "polyline",
        "filter",
        "clippath",
        "ellipse",
        "desc",
        "image"
    ]

    def __init__(self,
                 text_model=None,
                 font_model=None,
                 visual_model=None,
                 graph_num_eigvec=32,
                 html_tags=[],
                 tag_num_child_pos=48,
                 **kwargs):
        """Initializer

        Args:
            text_model: (Optional) A text features model instance. See `dom_features_text.py`. Default is None.
            font_model: (Optional) A font features model instance. See `dom_features_font.py`. Default is None.
            visual_model: (Optional) A visual features model instance. See `dom_features_visual.py`. Default is None.
            html_tags: (Optional) Supported HTML tags. Default is all possible tags (ie, empty list).
            graph_num_eigvec: (Optional) Number of graph Laplacian eigenvectors to use for positional encoding. Default is 32.
            tag_num_child_pos: (Optional) Size of one-hot vector used to represent the child position. Default is 48.
            **kwargs: (Optional) Additional variable parameters (for back-compatibility)
        """

        def _load_model(model):
            if isinstance(model, dict):
                classname, params = next(iter(model.items()))
                return eval(classname)(tag_features=self, **params)
            elif isinstance(model, str):
                model = hjson.loads(model)
                classname, params = next(iter(model.items()))
                return eval(classname)(tag_features=self, **params)
            else:
                return model

        self._text_model = _load_model(text_model)

        # Back-compatbility chunk
        if font_model is None and visual_model is None and kwargs.get("include_visual", False):
            font_model = FontSimple(tag_features=self)
            visual_model = None
            if kwargs.get("debug_no_wh", False):
                visual_model = VisualSimple(tag_features=self)

        self._font_model = _load_model(font_model)
        self._visual_model = _load_model(visual_model)

        if len(html_tags) == 0:
            html_tags = TagFeatures.HTML_TAGS
        self._html_tags_indices = {a: i for i, a in enumerate(html_tags)}
        self._html_tags = html_tags
        self._graph_num_eigvec = graph_num_eigvec
        self._tag_num_child_pos = tag_num_child_pos

        self._config = {
            "text_model": {self._text_model.__class__.__name__: self._text_model.get_config()},
            "font_model": {self._font_model.__class__.__name__: self._font_model.get_config()},
            "visual_model": {self._visual_model.__class__.__name__: self._visual_model.get_config()},
            "graph_num_eigvec": graph_num_eigvec,
            "tag_num_child_pos": tag_num_child_pos,
            "html_tags": html_tags
        }

    def get_config(self):
        return self._config

    def get_feature_size_and_offsets(self):
        soup = BeautifulSoup("<html><body>test</body></html>", "html.parser")
        self.set_tag_features(soup)

        def _find_offsets_for_feats(feats, count=0):
            if isinstance(feats, dict):
                flattened_feat_offsets = {}
                for k, v in feats.items():
                    offsets, new_count = _find_offsets_for_feats(v, count)
                    flattened_feat_offsets[k] = offsets if offsets else {"idx": count, "len": new_count - count}
                    count = new_count
                return flattened_feat_offsets, count
            else:
                return None, count + len(feats)
        return {
            "feature_size": len(soup.body.flattened_feats),
            "feature_offsets": _find_offsets_for_feats(soup.body.orig_feats)[0]
        }

    def _tag_pos_to_feature(self, tag):
        parent = tag.parent
        one_hot = np.zeros((self._tag_num_child_pos,), dtype=np.float32)
        if parent is not None:
            for i, c in enumerate(parent.find_all(recursive=False)):
                if c == tag:
                    one_hot[min(i, len(one_hot) - 1)] = 1.0
                    return one_hot.tolist()
            return None
        else:
            one_hot[0] = 1.0
            return one_hot.tolist()

    def _tag_type_to_feature(self, tag):
        all_tags = self._html_tags_indices
        tag_index = all_tags.get(tag.name, len(all_tags))
        # if tag_index == len(all_tags):
        #     print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>> tag.name: %s\n\n" % tag.name)
        one_hot = np.zeros(len(all_tags) + 1, dtype=np.float32)
        one_hot[tag_index] = 1.0
        return one_hot.tolist()

    def _flatten_feats(self, feats):
        flattened_feats = []
        if isinstance(feats, dict):
            for k, v in feats.items():
                flattened_feats = flattened_feats + self._flatten_feats(v)
        else:
            flattened_feats = flattened_feats + feats
        return flattened_feats

    def set_tag_features(self, soup, recover_failures=True, elem_props=None):
        """Extracts tag features from a HTML DOM and sets it as attributes in the tag

        Args:
            soup: The BeautifulSoup DOM object to extract features for graph input feature
            recover_failures: (Optional) If failed to extract features, will ignore
                and use default values. Otherwise will raise Exception. Default is True
            elem_props: (Optional) A nested dict of element `attr_names` (a list) and `props_vals ` (a dict).
                Default is None
        """
        graph_nodes = None
        eigvec = None
        if self._graph_num_eigvec > 0:
            graph = to_graph(soup).to_undirected(as_view=True)
            num_nodes = len(graph.nodes)
            k = self._graph_num_eigvec + 1
            num_tries = 1
            while True:
                try:
                    # # Eigenvectors with numpy
                    # EigVal, EigVec = np.linalg.eig(L.toarray())
                    # idx = EigVal.argsort() # increasing order
                    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
                    # g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()

                    laplacian = networkx.linalg.laplacianmatrix.normalized_laplacian_matrix(graph)
                    # laplacian = nxlaplace.laplacian_matrix(graph_undir).asfptype()
                    _, eigvec = scipy.sparse.linalg.eigsh(laplacian,
                                                          k=k,
                                                          which="SM",
                                                          maxiter=None,
                                                          ncv=max(4 * k + 1, 32))
                except Exception as e:
                    _, _, tb = sys.exc_info()
                    traceback.print_tb(tb)  # Fixed format
                    print(e)
                    print()
                    if isinstance(e, scipy.sparse.linalg.ArpackNoConvergence) and num_tries < 3:
                        print("Retrying ...")
                        print()
                        num_tries += 1
                        continue
                    if not recover_failures:
                        raise AssertionError("Failed to get eigenvectors. "
                                             "Min nodes expected vs actual: %d, %d" % (k, num_nodes))
                    print("Using zero eigenvectors. Min nodes expected vs actual: %d, %d" % (k, num_nodes))
                    print()
                    eigvec = np.zeros((num_nodes, k), dtype=np.float32)
                break

            graph_nodes = list(graph.nodes)

        tag_list = [t for t in soup.find_all()]
        feats_list = [{} for _ in tag_list]

        # Be consistent for the sequence of input features, in flatten_feats
        # Update Font, Text and Visual features
        if self._font_model:
            self._font_model.set_feats(tag_list, feats_list, elem_props=elem_props)
        if self._text_model:
            self._text_model.set_feats(tag_list, feats_list)
        if self._visual_model:
            self._visual_model.set_feats(tag_list, feats_list, elem_props=elem_props)

        for tag, feats in zip(tag_list, feats_list):
            feats["tag_type"] = self._tag_type_to_feature(tag)
            feats["num_child"] = [len(tag.find_all(recursive=False))]
            if self._tag_num_child_pos > 0:
                feats["tag_child_pos"] = self._tag_pos_to_feature(tag)

            if self._graph_num_eigvec > 0:
                idx = graph_nodes.index(tag.tag_id)
                feats["tag_graph_eigen"] = eigvec[idx, 1:].tolist()

            tag.orig_feats = feats
            tag.flattened_feats = self._flatten_feats(feats)
