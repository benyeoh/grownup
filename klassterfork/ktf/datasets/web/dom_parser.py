import logging
import os

import cssutils
cssutils.log.setLevel(logging.CRITICAL)

import magic
import networkx as nx
import premailer
import bs4
from bs4 import BeautifulSoup
import json
import brotli

import ktf.datasets.web


def to_soup_from_file(html_path, inline_css, filter_tags=["script", "noscript", "meta"], prop_filepath=None):
    """Converts a html text file to a BeautifulSoup DOM.

    Args:
        html_path: Path of input html.
        inline_css: Inline CSS styles into tags.
        filter_tags: (Optional) A list of tags to remove from the DOM. Default is ["script", "noscript", "meta"].
        prop_filepath: (Optional) Path of element property file corresponding to its HTML

    Return:
        - if prop_filepath is a file, return a tuple of a BeautifulSoup DOM and element properties
        - if prop_filepath isnt a file or not exist, return a BeautifulSoup DOM
    """
    with open(html_path, "rb") as fd:
        m = magic.Magic(mime_encoding=True)
        doc = fd.read()
        try:
            html_doc = doc.decode(m.from_buffer(doc))
        except:
            try:
                html_doc = doc.decode("cp1252")
            except UnicodeDecodeError:
                html_doc = doc.decode("iso-8859-1")
        soup = to_soup(html_doc, inline_css, filter_tags)

    try:
        return (soup, _to_elem_props_from_file(prop_filepath)) if os.path.isfile(prop_filepath) else (soup, None)
    except TypeError as e:
        print(e)
        return soup


def to_soup(html_doc, inline_css, filter_tags=["script", "noscript", "meta"]):
    """Converts a html text document to a BeautifulSoup DOM and initializes each tag with an ID.

    Args:
        inline_css: Inline CSS styles into tags. Default is True.
        filter_tags: (Optional) A list of tags to remove from the DOM. Default is ["script", "noscript", "meta"].
    Return:
        A BeautifulSoup DOM
    """

    if inline_css:
        import requests

        # We need this to force a timeout for premailer
        _send = requests.Session.send

        def _send_w_timeout(*args, **kwargs):
            DEFAULT_TIMEOUT = 10
            if kwargs.get("timeout", None) is None:
                kwargs["timeout"] = DEFAULT_TIMEOUT
            return _send(*args, **kwargs)
        requests.Session.send = _send_w_timeout

        try:
            html_doc = premailer.transform(html_doc)
        except Exception as e:
            print()
            print(e)
            print("Warning: Failed to fetch CSS")
            print()
            try:
                html_doc = premailer.transform(html_doc, allow_network=False)
            except Exception as e:
                print()
                print(e)
                print("Error: Failed to fetch CSS. Skipping")
                print()
                pass

        # Reset to previous
        requests.Session.send = _send

    soup = BeautifulSoup(html_doc, "html.parser")

    # Filter out comments
    for node in soup.find_all(text=lambda x: isinstance(x, bs4.Comment)):
        node.extract()

    # Filter out "useless" tags
    for filt in filter_tags:
        for s in soup.select(filt):
            s.extract()

    # Parse styles and set IDs
    for i, tag in enumerate(soup.find_all()):
        try:
            style = cssutils.parseStyle(tag["style"])
        except KeyError:
            style = None
        tag.tag_style = style
        tag.tag_id = i

    return soup


def to_graph(soup):
    """Converts a BeautifulSoup DOM with tag IDs to a networkx graph

    Args:
        soup: The BeautifulSoup DOM object.

    Return:
        A networkx graph
    """
    g = nx.DiGraph()
    index_to_tag = {}
    for tag in soup.find_all():
        index_to_tag[tag.tag_id] = tag
        try:
            feat = tag.flattened_feats
        except:
            feat = None
        g.add_node(tag.tag_id, feat=feat)

    for n in g.nodes:
        tag = index_to_tag[n]
        if type(tag.parent) == bs4.element.Tag:
            g.add_edge(n, tag.parent.tag_id, type_id=0)
        for c in tag.children:
            if type(c) == bs4.element.Tag:
                g.add_edge(n, c.tag_id, type_id=1)
    return g


def from_file_count_dom_tags(html_files):
    all_tags = {}
    for i, file in enumerate(html_files):
        print("Processing: %s" % file)
        tags_set = set()
        soup = to_soup_from_file(file, inline_css=False)
        for tag in soup.find_all():
            tags_set.add(tag.name)

            # Accumulate stats
            if tag.name not in all_tags:
                all_tags[tag.name] = {}
                all_tags[tag.name]["total"] = 1
                all_tags[tag.name]["per_file"] = 0
            else:
                all_tags[tag.name]["total"] += 1

        # Accumulate stats per file
        for tag_name in tags_set:
            all_tags[tag_name]["per_file"] += 1

        if (i % 100) == 0:
            print("Processed: %d" % i)

    print("Total processed: %d" % i)
    return all_tags


def from_file_filter_dom_tags(html_files, threshold=0):
    all_tag_counts = from_file_count_dom_tags(html_files)
    supported_tags = set(ktf.datasets.web.TagFeatures.HTML_TAGS) & set(all_tag_counts.keys())
    supported_tags_counts = [(all_tag_counts[t]["per_file"], t)
                             for t in supported_tags if all_tag_counts[t]["per_file"] >= threshold]
    supported_tags_counts.sort(reverse=True)

    unsupported_tags = set(all_tag_counts.keys()) - set(ktf.datasets.web.TagFeatures.HTML_TAGS)
    unsupported_tags_counts = [(all_tag_counts[t]["per_file"], t)
                               for t in unsupported_tags if all_tag_counts[t]["per_file"] >= threshold]
    unsupported_tags_counts.sort(reverse=True)

    return ([tag_pair[1] for tag_pair in supported_tags_counts], [tag_pair[0] for tag_pair in supported_tags_counts],
            [tag_pair[1] for tag_pair in unsupported_tags_counts], [tag_pair[0] for tag_pair in unsupported_tags_counts])


def _to_elem_props_from_file(file):
    """Parse element property file which comprises bytes object with all computed styles and absolute position and size
    of each element.

    Args:
        file: Path to element property file
    Return:
        elem_props: A nested dict of element `attr_names`  (a list) and `props_vals ` (a dict). In `props_value`:
        index - key; and property incl. computed style and position/size - values
    """

    def _flatten_list(elem_prop):
        flat_list = []
        for elem in elem_prop:
            if isinstance(elem, list):
                for str in elem:
                    flat_list.append(str)
            else:
                flat_list.append(elem)
        return flat_list

    # Decompress element property file
    with open(file, "rb") as f:
        orig_elem_props = json.loads(brotli.decompress(f.read()))

    elem_props = {}
    # Get attribute names: for computed styles directly load, for position/size flatten and rename
    elem_props["attr_names"] = []
    for attr_name in orig_elem_props["attributeNames"]:
        if isinstance(attr_name, list):  # original position/size [top, left, width, height]
            for elem in attr_name:
                elem_props["attr_names"].append("BoundingClientRect_" + elem)
        elif isinstance(attr_name, str):
            elem_props["attr_names"].append(attr_name)

    # Get element properties with _klass_elem_id as the key
    elem_props["props_vals"] = {}
    for elem_id in list(orig_elem_props["elements"].keys()):
        elem_props["props_vals"][elem_id] = _flatten_list(orig_elem_props["elements"][elem_id])

    return elem_props
