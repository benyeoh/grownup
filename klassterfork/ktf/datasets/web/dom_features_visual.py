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
