from .dom_features_utils import _size_to_feature, _color_to_feature, _exact_match


class VisualSimple:
    """Default model for extracting DOM visual (ie, width/height/color etc)
    of a tag. This implementation uses the class and id attributes of a tag which
    has an indirect relationship with visual layouts through css styles.
    """

    def __init__(self, tag_features, debug_no_wh=True):
        self._tag_features = tag_features
        self._debug_no_wh = debug_no_wh

    def _height_to_feature(self, tag):
        try:
            height = tag["height"]
        except KeyError:
            height = None
            style = tag.tag_style
            if style and style["height"] != "":
                height = style["height"]

        return _size_to_feature(height)

    def _width_to_feature(self, tag):
        try:
            width = tag["width"]
        except KeyError:
            width = None
            style = tag.tag_style
            if style and style["width"] != "":
                width = style["width"]

        return _size_to_feature(width)

    def _class_to_feature(self, tag_list):
        word_list_list = []
        for tag in tag_list:
            try:
                word_list = tag["class"]
                # print(word_list)
            except KeyError:
                word_list = []
            word_list_list.append(word_list)
        return self._tag_features._text_model.get_mean_word_vector(word_list_list)

    def _id_to_feature(self, tag_list):
        word_list_list = []
        for tag in tag_list:
            try:
                word_list = tag["id"]
            except KeyError:
                word_list = []
            word_list_list.append(word_list)
        return self._tag_features._text_model.get_mean_word_vector(word_list_list)

    def set_feats(self, tag_list, feat_list, **kwargs):
        """Extracts features from a list of HTML tags and sets it.

        Args:
            tag_list: A list of tags to extract features.
            feat_list: A list of extracted features to be modified. Each entry is a dictionary
                of features corresponding to each tag in `tag_list`.
            **kwargs: Additional variable parameters like A nested dict comprising all computed style.
        """
        class_feat_list = self._class_to_feature(tag_list)
        id_feat_list = self._id_to_feature(tag_list)

        for tag, feat, class_feat, id_feat in zip(tag_list, feat_list, class_feat_list, id_feat_list):
            if not self._debug_no_wh:
                feat["height"] = self._height_to_feature(tag)
                feat["width"] = self._width_to_feature(tag)
            feat["class_attr"] = class_feat
            feat["id_attr"] = id_feat

    def get_config(self):
        return {
            "debug_no_wh": self._debug_no_wh
        }


class VisualCompStyles:
    """Optional model for extracting visual features of a tag. This implementation uses a list of selected computed
    style and element position and size, which can be customized/extended based on requirement
    """

    def __init__(self, tag_features):
        self._tag_features = tag_features

    def _elem_size_to_feature(self, tag, elem_props):
        try:
            indice_width = _exact_match("BoundingClientRect_width", elem_props)
            indice_height = _exact_match("BoundingClientRect_height", elem_props)
            width = float(elem_props["props_vals"][tag["_klass_elem_id"]][indice_width] / 2048.0)
            height = float(elem_props["props_vals"][tag["_klass_elem_id"]][indice_height] / 2048.0)

            if elem_props["props_vals"][tag["_klass_elem_id"]][_exact_match("display", elem_props)] == "none":
                if width != 0.0 or height != 0.0:
                    print("Size of %s is %.3f and %.3f" % (tag["_klass_elem_id"], width, height))
                    return [0.0, 0.0]
            elif width > 4.0 or width < 0.0:
                return [0.0, 0.0]
            elif height > 4.0 or height < 0.0:
                return [0.0, 0.0]
            return [width, height]

        except (KeyError, TypeError) as e:
            # KeyError refers to missing _klass_elem_id, not able to retrieve size info
            # print(e)
            return [0.0, 0.0]

    def _elem_position_to_feature(self, tag, elem_props):
        try:
            indice_top = _exact_match("BoundingClientRect_top", elem_props)
            indice_left = _exact_match("BoundingClientRect_left", elem_props)
            top = float(elem_props["props_vals"][tag["_klass_elem_id"]][indice_top] / 2048.0)
            left = float(elem_props["props_vals"][tag["_klass_elem_id"]][indice_left] / 2048.0)

            if elem_props["props_vals"][tag["_klass_elem_id"]][_exact_match("display", elem_props)] == "none":
                if top != 0.0 or left != 0.0:
                    # print("Position of %s is %.3f and %.3f" % (tag["_klass_elem_id"], top, left))
                    return [0.0, 0.0]
            elif top > 4.0 or top < 0.0:
                return [0.0, 0.0]
            elif left > 4.0 or left < 0.0:
                return [0.0, 0.0]
            return [top, left]

        except (KeyError, TypeError) as e:
            # print(e)
            # KeyError refers to missing _klass_elem_id, not able to retrieve size info
            return [0.0, 0.0]

    def _fg_color_to_feature(self, tag, elem_props):
        try:
            indice = _exact_match("color", elem_props)
            fg_color = elem_props["props_vals"][tag["_klass_elem_id"]][indice]
            return _color_to_feature(fg_color)

        except (KeyError, TypeError) as e:
            # KeyError refers to missing _klass_elem_id, not able to retrieve size info
            # print(e)
            return _color_to_feature("rgba(0, 0, 0, 0.0)")

    def _bg_color_to_feature(self, tag, elem_props):
        try:
            indice = _exact_match("background-color", elem_props)
            bg_color = elem_props["props_vals"][tag["_klass_elem_id"]][indice]
            return [0.0, 0.0, 0.0, 0.0] if (bg_color == "transparent") else _color_to_feature(bg_color)

        except (KeyError, TypeError) as e:
            # KeyError refers to missing _klass_elem_id, not able to retrieve size info
            # TypeError occured during initiation, while  elem_props = None
            # print(e)
            return [0.0, 0.0, 0.0, 0.0]  # The official initial value for background color is "transparent"

    def set_feats(self, tag_list, feats_list, **kwargs):
        """Extracts features from a list of HTML tags and sets it.

        Args:
            tag_list: A list of tags to extract features.
            feats_list: A list of extracted features to be modified. Each entry is a dictionary
                of features corresponding to each tag in `tag_list`.
            **kwargs: Additional variable parameters like A nested dict comprising all computed style.
        """

        elem_props = kwargs.get("elem_props")
        for tag, feats in zip(tag_list, feats_list):
            feats["bg_color"] = self._bg_color_to_feature(tag, elem_props)
            feats["fg_color"] = self._fg_color_to_feature(tag, elem_props)
            feats["elem_pos"] = self._elem_position_to_feature(tag, elem_props)
            feats["elem_size"] = self._elem_size_to_feature(tag, elem_props)

    def get_config(self):
        return {}
