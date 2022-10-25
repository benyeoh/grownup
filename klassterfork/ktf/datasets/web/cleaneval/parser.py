import os
import glob
import magic
import re
import difflib
import sys
import traceback

import bs4
from bs4 import BeautifulSoup

import ktf.datasets.web


class ParserHTML:
    """Parses raw CleanEval dataset described here: https://sigwac.org.uk/cleaneval/devset.html

    The parser looks through each raw input pair of (raw html, cleaned results) and:

    1. Extracts the DOM from the HTML
    2. Performs an LCS (longest-common-subsequence) mapping of the cleaned results to html tags and flags those
    that appear in the cleaned results as "content"
    3. Extracts features for each tag
    4. Converts the DOM to a graph
    5. Generates positive / negative subgraphs from the graph
    """

    def __init__(self, orig_path, cleaned_path, inline_css, tag_features_model, include_tag=False):
        """Initializer

        Args:
            orig_path: The file path to the raw html
            cleaned_path: The file path to the cleaned results
            inline_css: Inline CSS styles into tags
            tag_features_model: The feature extractor for each DOM tag
            include_tag: (Optional) Includes the DOM tag as an "tag" attribute in the graph node. Default is False
        """
        self._orig_path = orig_path
        self._cleaned_path = cleaned_path
        self._tag_features = tag_features_model
        self._inline_css = inline_css
        self._include_tag = include_tag
        self._content_sampler = None
        self._non_content_sampler = None

    def _read_file(self, path):
        with open(path, "rb") as fd:
            m = magic.Magic(mime_encoding=True)
            doc = fd.read()
            try:
                return doc.decode(m.from_buffer(doc))
            except:
                try:
                    return doc.decode("cp1252")
                except UnicodeDecodeError:
                    return doc.decode("iso-8859-1")
        return None

    def _gold_std_to_list(self, txt):
        txt = re.sub("(^|\n)(URL:.*)", "", txt)
        txt = re.sub("(^|\n)[ \t]*(<.*?>)", "\n", txt)
        txt_list = txt.split()
        return txt_list

    def _soup_txt_to_list(self, soup):
        blk_elems = set(["p", "h1", "h2", "h3", "h4", "h5", "h6",
                         "ul", "ol", "td", "dl", "pre", "hr", "blockquote", "address"])

        """
        def _find_blk_parent(tag, max_lvl=5):
            parent = tag.parent
            for _ in range(max_lvl):
                if tag.parent.name in blk_elems:
                    return tag.parent
                parent = tag.parent
            return None
        """

        filtered_parent = []
        filtered_str = []
        is_whitespace_last = True
        #last_blk_parent = None
        doc = soup.html if soup.html else soup.contents[0]
        for s in doc.next_elements:
            if type(s) == bs4.element.Tag:
                if s.name == "br" or s.name in blk_elems:
                    is_whitespace_last = True
                elif s.tag_style:
                    if s.tag_style["display"] == "block":
                        is_whitespace_last = True

            elif type(s) == bs4.element.NavigableString:
                text_str = str(s)
                split_str = str(s).split()
                #blk_parent = _find_blk_parent(s)
                if len(split_str) > 0:
                    # We split strings into separate word tokens based on whether or not
                    # there is a whitespace either in the beginning of this string or the end of the last string
                    # or if the string starts on a new HTML block
                    if (is_whitespace_last or text_str[0].isspace()):  # or blk_parent != last_blk_parent):
                        filtered_str.extend(split_str)
                        # We also store owner tags for each word token that this string belongs to
                        filtered_parent.extend([s.parent for _ in range(len(split_str))])
                    else:
                        # We assume that there is no whitespace between this string and the previous
                        filtered_str[-1] += split_str[0]
                        if not isinstance(filtered_parent[-1], list):
                            filtered_parent[-1] = [filtered_parent[-1]]
                        filtered_parent[-1].append(s.parent)

                        if len(split_str) > 1:
                            filtered_str.extend(split_str[1:])
                            filtered_parent.extend([s.parent for _ in range(len(split_str[1:]))])

                #last_blk_parent = blk_parent
                if len(text_str) > 0:
                    is_whitespace_last = text_str[-1].isspace()

        return filtered_str, filtered_parent

    def _get_lcs_counts(self, soup, toks, tok_tags, gold_std_toks):
        """Get the longest common subsequence (LCS) comparing word tokens from `gold_std_toks` to `toks`, then
        count tokens owned by HMTL tags that appear in the LCS.
        """
        seq_matcher = difflib.SequenceMatcher(None, toks, gold_std_toks, autojunk=False)

        # Get the LCS assemble the word tokens in a list
        lcs = [tok for block in seq_matcher.get_matching_blocks() for tok in toks[block.a:(block.a + block.size)]]

        # Get the same LCS but assemble the relevant owner tags of each token in a list
        lcs_tags = [tag for block in seq_matcher.get_matching_blocks()
                    for tag in tok_tags[block.a:(block.a + block.size)]]

        # Now we count the total number of tokens that each tag owns, and how many of these tokens belong in the LCS
        tag_tok_count = {}
        tag_tok_lcs_count = {}
        doc = soup.html if soup.html else soup.contents[0]
        for s in doc.find_all_next(string=True):
            if type(s) == bs4.element.NavigableString:
                if len(str(s).split()) > 0:
                    if s.parent.tag_id not in tag_tok_count:
                        tag_tok_count[s.parent.tag_id] = 0
                        tag_tok_lcs_count[s.parent.tag_id] = 0
                    tag_tok_count[s.parent.tag_id] += len(str(s).split())

        for tag in lcs_tags:
            if isinstance(tag, list):
                for t in tag:
                    tag_tok_lcs_count[t.tag_id] += 1
            else:
                tag_tok_lcs_count[tag.tag_id] += 1

        return tag_tok_count, tag_tok_lcs_count, lcs

    def init_graph_samplers(self):
        prop_filepath = os.path.join(os.path.dirname(self._orig_path), os.path.basename(self._orig_path) + ".br")
        (soup, elem_props) = ktf.datasets.web.to_soup_from_file(
            self._orig_path, self._inline_css, prop_filepath=prop_filepath)

        self._tag_features.set_tag_features(soup, recover_failures=True, elem_props=elem_props)

        graph = ktf.datasets.web.to_graph(soup)

        if self._include_tag:
            for tag in soup.find_all():
                try:
                    graph.nodes[tag.tag_id]["tag"] = tag
                except:
                    pass

        gold_std = self._read_file(self._cleaned_path)
        gold_std_toks = self._gold_std_to_list(gold_std)
        toks, tok_tags = self._soup_txt_to_list(soup)
        tag_tok_count, tag_tok_lcs_count, lcs = self._get_lcs_counts(soup, toks, tok_tags, gold_std_toks)

        """
        print()
        print()
        print("raw: %s" % toks)
        print()
        print("gold_std: %s" % gold_std_toks)
        print()
        print("lcs: %s" % lcs)
        print()
        print()
        """

        if len(gold_std_toks) > 0:
            assert (float(len(lcs)) / len(gold_std_toks)) > 0.75

        content_nodes = []
        non_content_nodes = []
        other_nodes = []
        for tag in soup.find_all():
            if tag.tag_id in tag_tok_lcs_count:
                lcs_ratio = float(tag_tok_lcs_count[tag.tag_id]) / tag_tok_count[tag.tag_id]
                if lcs_ratio > 0:
                    content_nodes.append(tag.tag_id)
                else:
                    non_content_nodes.append(tag.tag_id)
            else:
                other_nodes.append(tag.tag_id)

        self._graph = graph
        self._content_sampler = ktf.datasets.web.GraphNodeSampler(graph, content_nodes)
        self._non_content_sampler = ktf.datasets.web.GraphNodeSampler(graph, non_content_nodes)

    def sample_content(self, max_depth, max_neighbours, max_nodes):
        return self._content_sampler.sample(max_depth, max_neighbours, max_nodes)

    def sample_non_content(self, max_depth, max_neighbours, max_nodes):
        return self._non_content_sampler.sample(max_depth, max_neighbours, max_nodes)

    def parse(self, max_depth, max_neighbours, max_nodes):
        if self._content_sampler is None:
            self.init_graph_samplers()

        for sample in self._content_sampler.iter_nodes(max_depth, max_neighbours, max_nodes):
            yield sample + (self._graph, 1)

        for sample in self._non_content_sampler.iter_nodes(max_depth, max_neighbours, max_nodes):
            yield sample + (self._graph, 0)


class Parser:
    def __init__(self, orig_dir, cleaned_dir, inline_css, tag_features_model, include_tag=False, exclude_files=[]):
        """Initializer

        Args:
            orig_dir: The dir path to the raw html
            cleaned_di: The dir path to the cleaned results
            inline_css: Inline CSS styles into tags
            tag_features_model: The feature extractor for each DOM tag
            include_tag: (Optional) Includes the DOM tag as an "tag" attribute in the graph node. Default is False
        """

        self._tag_features = tag_features_model
        self._include_tag = include_tag
        self._inline_css = inline_css
        all_orig = glob.glob(os.path.join(orig_dir, "*.html"))
        all_cleaned = glob.glob(os.path.join(cleaned_dir, "*.txt"))

        all_cleaned_map = {}
        for file in all_cleaned:
            filename = os.path.basename(file).split(".")[0].split("-")[0]
            all_cleaned_map[filename] = file

        self._all_pairs = []
        for file in all_orig:
            filename = os.path.basename(file).split(".")[0]
            if filename in all_cleaned_map and os.path.basename(file) not in exclude_files:
                self._all_pairs.append((file, all_cleaned_map[filename]))
            else:
                print("Skipping parsing %s" % file)

    def parse(self, max_depth, max_neighbours, max_nodes):
        """Parses all HTML / cleaned results in the folder and returns a positive or negative
        sample subgraph

        Args:
            max_depth: The maximum depth of the subgraph from the originating node
            max_neighbours: The maximum number of neighbours per node
            max_nodes: The maximum total number of nodes in the subgraph

        Returns:
            An iterator of tuples of 
            (node adjacency tensor, node features tensor, start node ID, old node ID->new node ID map)
        """

        total_content = 0
        total_non_content = 0
        for orig_file, cleaned_file in self._all_pairs:
            num_content = 0
            num_non_content = 0
            print("Parsing: %s, %s" % (orig_file, cleaned_file))
            try:
                html_parser = ParserHTML(orig_file, cleaned_file, self._inline_css,
                                         self._tag_features, include_tag=self._include_tag)
                for sample in html_parser.parse(max_depth, max_neighbours, max_nodes):
                    if sample[-1] > 0:
                        num_content += 1
                    else:
                        num_non_content += 1
                    yield sample + (orig_file,)
            except AssertionError as e:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)  # Fixed format
                print()
                print("Skipping file %s..." % orig_file)
                print()

            print("Content vs non-content: %d vs %d" % (num_content, num_non_content))
            total_content += num_content
            total_non_content += num_non_content
        print("Total content vs non-content: %d vs %d" % (total_content, total_non_content))
