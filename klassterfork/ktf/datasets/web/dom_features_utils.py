
def _size_to_feature(size_str):
    """Convert from various units of size (ie, px, em, pt and so on) to a
    representative binary format.

    Args:
        size_str: A text string describing a size unit. Example, "100px".

    Returns:
        A dictionary with the one-hot class and float value representing the size
    """

    size = {}
    if size_str:
        try:
            if size_str[-1:] == "%":
                size["class"] = [1.0, 0.0, 0.0, 0.0]
                size["val"] = [float(size_str[:-1]) / 100.0]
                return size
            elif size_str[-2:] == "em" and size_str[-3:] != "rem":  # implementing rem as em
                size["class"] = [0.0, 0.0, 1.0, 0.0]
                size["val"] = [float(size_str[:-2])]
                return size
            elif size_str[-3:] == "rem":   # implementing rem as em
                size["class"] = [0.0, 0.0, 1.0, 0.0]
                size["val"] = [float(size_str[:-3])]  # rem vs em
                return size
            elif size_str[-2:] == "px":
                size["class"] = [0.0, 1.0, 0.0, 0.0]
                size["val"] = [float(size_str[:-2])]
                return size
            elif size_str[-2:] == "in":
                units = float(size_str[:-2]) * 96.0
                size["class"] = [0.0, 1.0, 0.0, 0.0]
                size["val"] = [units]
                return size
            elif size_str[-2:] == "pt":
                units = float(size_str[:-2]) * (1.0 / 72.0) * 96.0
                size["class"] = [0.0, 1.0, 0.0, 0.0]
                size["val"] = [units]
                return size
            elif size_str[-2:] == "pc":
                units = float(size_str[:-2]) * (12.0 / 72.0) * 96.0
                size["class"] = [0.0, 1.0, 0.0, 0.0]
                size["val"] = [units]
                return size
            else:
                try:
                    size["class"] = [0.0, 1.0, 0.0, 0.0]
                    size["val"] = [float(size_str)]
                    return size
                except:
                    pass
        except ValueError as e:
            print(e)
            print("size_str: %s" % size_str)

    size["class"] = [0.0, 0.0, 0.0, 1.0]
    size["val"] = [0.0]
    return size


def _color_to_feature(color_str):
    """Convert from various color models (hex, rgb(), hsl(), hsla() and named color) to rgba(). Default alpha = 0.0
    Args:
        color_str: A text string describing a color value. Example, "red", "rgb(255,0,0)", "#FFFFFF" and
        "hsl(0, 100%, 50%)"

    Returns:
        A list of int/float representing the value of color channels R, G, B and Alpha
    """
    def _hsl_to_rgb(color_list):
        """Ref: https://www.rapidtables.com/convert/color/hsl-to-rgb.html
        """
        try:
            assert len(color_list) == 3
            # Get Hue, Saturation, Lightness
            h = int(color_list[0])
            if color_list[1][-1] == "%":
                s = float(color_list[1][:-1]) / 100.0
            if color_list[2][-1] == "%":
                l = float(color_list[2][:-1]) / 100.0
            # Compute C, X and m
            if h < 360 and s <= 1 and l <= 1:
                C = (1 - abs(2 * l - 1)) * s
                X = C * (1 - abs(((h / 60) % 2) - 1))
                m = l - C / 2
                # Get intermediate rgb
                rev_h = int(h / 60)
                temp_color = {
                    0: [C, X, 0],
                    1: [X, C, 0],
                    2: [0, C, X],
                    3: [0, X, C],
                    4: [X, 0, C],
                    5: [C, 0, X]
                }
                # Convert to rgb
                return [round((col + m) * 255) for col in temp_color.get(rev_h, None)]
            else:
                raise ValueError("Invalid HSL color detected")
        except AssertionError:
            print("Missing information in color string")

    def _named_color_to_rgb(color_str):
        color_palette = {
            "black": [0, 0, 0],
            "white": [255, 255, 255],
            "red": [255, 0, 0],
            "lime": [0, 255, 0],
            "blue": [0, 0, 255],
            "yellow": [255, 255, 0],
            "cyan": [0, 255, 255],
            "magenta": [255, 0, 255],
            "silver": [191, 191, 191],
            "gray": [128, 128, 128],
            "maroon": [128, 0, 0],
            "olive": [128, 128, 0],
            "green": [0, 128, 0],
            "purple": [128, 0, 128],
            "teal": [0, 128, 128],
            "navy": [0, 0, 128]
        }
        if color_str in list(color_palette.keys()):
            return color_palette[color_str]
        else:
            return [0, 0, 0]

    color = []
    if color_str:
        try:
            if color_str[0:4] == "rgba":
                color = list(map(int, color_str[5:-1].split(", ")[:-1]))
                color = [col / 255.0 for col in color]
                color.append(float(color_str[5:-1].split(", ")[-1]))

            elif color_str[0:3] == "rgb":
                color = list(map(int, color_str[4:-1].split(", ")))
                color = [col / 255.0 for col in color]
                color.append(0.0)  # Add alpha

            elif color_str[0] == "#" and len(color_str) == 7:  # Handle hex
                color = list(int(color_str[i:i + 2], 16) for i in (1, 3, 5))
                color = [col / 255.0 for col in color]
                color.append(0.0)

            elif color_str[0:3] == "hsl" and color_str[0:4] != "hsla":
                color = _hsl_to_rgb(color_str[4:-1].split(", "))
                color = [col / 255.0 for col in color]
                color.append(0.0)

            elif color_str[0:4] == "hsla":
                color = _hsl_to_rgb(color_str[5:-1].split(", ")[:-1])
                color = [col / 255.0 for col in color]
                color.append(float(color_str[5:-1].split(", ")[-1]))

            elif isinstance(color_str, str):
                color = _named_color_to_rgb(color_str)
                color = [col / 255.0 for col in color]
                color.append(0.0)
            # Next break: handle "current color"

        except (ValueError, TypeError):
            print("color_str: %s" % color_str)
            color = [0.0, 0.0, 0.0, 0.0]
    else:
        print("Missing color string")
        color = [0.0, 0.0, 0.0, 0.0]

    return color


def _exact_match(prop, elem_props):
    if prop in elem_props["attr_names"]:
        return elem_props["attr_names"].index(prop)
    else:
        return None
