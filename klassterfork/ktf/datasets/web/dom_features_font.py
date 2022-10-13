
from .dom_features_utils import _size_to_feature, _exact_match


def get_font_size_one_hot(font_size):
    val_indices = ["medium",    # 16px
                   "xx-small",  # 9px
                   "x-small",   # 10px
                   "small",     # 13px
                   "large",     # 18px
                   "x-large",   # 24px
                   "xx-large",  # 32px
                   "smaller",   # 0.835em
                   "larger",    # 1.2em
                   # "initial",
                   "inherit"]

    one_hot = [0.0] * (len(val_indices))
    if font_size in val_indices:
        one_hot[val_indices.index(font_size)] = 1.0
    else:
        size = _size_to_feature(font_size)
        if size["class"][-1] == 1.0:
            # Invalid values
            one_hot[val_indices.index("inherit")] = 1.0
            if font_size is not None:
                print("Font invalid values: %s, %s" % (font_size, one_hot))
        elif size["class"][0] == 1.0 or size["class"][2] == 1.0:
            # Relative values
            if size["val"][0] >= 1.01:
                one_hot[val_indices.index("larger")] = 1.0
            elif size["val"][0] <= 0.99:
                one_hot[val_indices.index("smaller")] = 1.0
            else:
                one_hot[val_indices.index("inherit")] = 1.0
        elif size["class"][1] == 1.0:
            # Abs values
            if size["val"][0] <= 9.0:
                one_hot[val_indices.index("xx-small")] = 1.0
            elif size["val"][0] <= (10.0 + 1):
                one_hot[val_indices.index("x-small")] = 1.0
            elif size["val"][0] <= (13.0 + 1.0):
                one_hot[val_indices.index("small")] = 1.0
            elif size["val"][0] <= (16.0 + 1.0):
                one_hot[val_indices.index("medium")] = 1.0
            elif size["val"][0] <= (18.0 + 4.0):
                one_hot[val_indices.index("large")] = 1.0
            elif size["val"][0] <= (24.0 + 6.0):
                one_hot[val_indices.index("x-large")] = 1.0
            else:
                one_hot[val_indices.index("xx-large")] = 1.0
    return one_hot


def get_font_weight_one_hot(font_weight):
    font_weight = {
        "100": "normal",
        "200": "normal",
        "300": "normal",
        "400": "normal",
        "500": "normal",
        "600": "bold",
        "700": "bold",
        "800": "bold",
        "900": "bold",
        "lighter": "normal",
        "bolder": "bold",
        "initial": "normal"
    }.get(font_weight, font_weight)

    weight_indices = ["normal",
                      "bold",
                      "inherit"]
    one_hot = [0.0] * (len(weight_indices))
    if font_weight in weight_indices:
        one_hot[weight_indices.index(font_weight)] = 1.0
    else:
        one_hot[weight_indices.index("inherit")] = 1.0

    return one_hot


def get_font_style_one_hot(font_style):
    font_style = {
        "initial": "normal"
    }.get(font_style, font_style)

    style_indices = ["normal",
                     "italic",
                     "oblique",
                     "inherit"]
    one_hot = [0.0] * (len(style_indices))
    if font_style in style_indices:
        one_hot[style_indices.index(font_style)] = 1.0
    else:
        one_hot[style_indices.index("inherit")] = 1.0
    return one_hot


class FontSimple:
    """Default model for extracting DOM font features (ie, size/style/weight etc)
    of a tag.
    """

    def __init__(self, tag_features):
        self._tag_features = tag_features

    def _font_weight_to_feature(self, tag):
        style = tag.tag_style

        font_weight = None
        if style:
            if style["font-weight"] != "":
                font_weight = style["font-weight"]
            elif style["font"] != "":
                try:
                    font_weight = style["font"].split(",")[0].split()[-3]
                except IndexError as e:
                    print("font-weight: %s" % e)
                    print(style["font"])
                    print()

        elif tag.name == "b" or tag.name == "strong":
            # Support for HTML 4
            font_weight = "bold"

        return get_font_weight_one_hot(font_weight)

    def _font_style_to_feature(self, tag):
        style = tag.tag_style

        font_style = None
        if style:
            if style["font-style"] != "":
                font_style = style["font-style"]
            elif style["font"] != "":
                try:
                    font_style = style["font"].split()[0]
                except IndexError as e:
                    print("font-style: %s" % e)
                    print(style["font"])
                    print()

        elif tag.name == "i" or tag.name == "em":
            # Support for HTML 4
            font_style = "italic"

        return get_font_style_one_hot(font_style)

    def _font_size_to_feature(self, tag):
        style = tag.tag_style

        font_size = None
        if style:
            if style["font-size"] != "":
                font_size = style["font-size"]
            elif style["font"] != "":
                try:
                    font_size = style["font"].split(",")[0].split()[-2].split("/")[0]
                except IndexError as e:
                    print("font-size: %s" % e)
                    print(style["font"])
                    print()
        elif tag.name == "font":
            # Support for HTML 4
            try:
                font_size = tag["size"]
                font_size = {
                    "1": "x-small",
                    "2": "small",
                    "3": "medium",
                    "4": "large",
                    "5": "large",
                    "6": "x-large",
                    "7": "xx-large",
                    "+0": "medium",
                    "+1": "large",
                    "+2": "x-large",
                    "+3": "xx-large",
                    "+4": "xx-large",
                    "+5": "xx-large",
                    "-0": "medium",
                    "-1": "small",
                    "-2": "x-small",
                    "-3": "xx-small",
                    "-4": "xx-small",
                    "-5": "xx-small",
                }.get(font_size, font_size)
            except KeyError:
                pass

        font_size = {
            # "xx-small": "x-small",
            # "xx-large": "x-large",
            "initial": "medium",
        }.get(font_size, font_size)

        return get_font_size_one_hot(font_size)

    def set_feats(self, tag_list, feats_list, **kwargs):
        """Extracts features from a list of HTML tags and sets it.

        Args:
            tag_list: A list of tags to extract features.
            feat_list: A list of extracted features to be modified. Each entry is a dictionary
                of features corresponding to each tag in `tag_list`.
            **kwargs: Additional variable parameters like a nested dict comprising all computed styles
        """

        for tag, feats in zip(tag_list, feats_list):
            feats["font_size"] = self._font_size_to_feature(tag)
            feats["font_weight"] = self._font_weight_to_feature(tag)
            feats["font_style"] = self._font_style_to_feature(tag)

    def get_config(self):
        return {}


class FontCompStyles:
    """Optional model for extracting font features (ie, size/style/weight etc) from computed
    styles saved.
    """

    def __init__(self, tag_features):
        self._tag_features = tag_features

    def _font_size_to_feature(self, tag, elem_props):
        try:
            indice = _exact_match("font-size", elem_props)
            font_size = elem_props["props_vals"][tag["_klass_elem_id"]][indice]
        except (KeyError, TypeError):
            font_size = "medium"  # Initial value given

        return get_font_size_one_hot(font_size)

    def _font_weight_to_feature(self, tag, elem_props):
        try:
            indice = _exact_match("font-weight", elem_props)
            font_weight = elem_props["props_vals"][tag["_klass_elem_id"]][indice]
        except (KeyError, TypeError):
            font_weight = "normal"

        return get_font_weight_one_hot(font_weight)

    def _font_style_to_feature(self, tag, elem_props):
        try:
            indice = _exact_match("font-weight", elem_props)
            font_style = elem_props["props_vals"][tag["_klass_elem_id"]][indice]
        except (KeyError, TypeError):
            font_style = "normal"

        return get_font_style_one_hot(font_style)

    def set_feats(self, tag_list, feats_list, **kwargs):
        """Extracts features from a list of HTML tags and sets it.

        Args:
            tag_list: A list of tags to extract features.
            feats_list: A list of extracted features to be modified. Each entry is a dictionary
                of features corresponding to each tag in `tag_list`.
            **kwargs: Additional variable parameter like a nested dict comprising all computed style.
        """

        elem_props = kwargs.get("elem_props")
        for tag, feats in zip(tag_list, feats_list):
            feats["font_size"] = self._font_size_to_feature(tag, elem_props)
            feats["font_weight"] = self._font_weight_to_feature(tag, elem_props)
            feats["font_style"] = self._font_style_to_feature(tag, elem_props)

    def get_config(self):
        return {}
