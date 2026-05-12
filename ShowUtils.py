class ShowInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "INT": ("INT", {"default": 0, "forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "show_int"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    CATEGORY = "Mine"
    
    def detect_type(self, value):
        return 'integer'

    def show_int(self, INT):
        type_info = [f"{value}" for value in INT]
        return {"ui": {"text": type_info}}

class ShowFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "FLOAT": ("FLOAT", {"default": 0.0, "forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "show_float"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    CATEGORY = "Mine"
    
    def detect_type(self, value):
        return 'float'

    def show_float(self, FLOAT):
        type_info = [f"{value}" for value in FLOAT]
        return {"ui": {"text": type_info}}


class ShowStringText:
    COLOR_CHOICES = [
        "white",
        "black",
        "gray",
        "silver",
        "red",
        "orange",
        "yellow",
        "lime",
        "green",
        "cyan",
        "aqua",
        "blue",
        "navy",
        "purple",
        "magenta",
        "pink",
        "brown",
        "gold",
        "tomato",
        "coral",
        "salmon",
        "violet",
        "plum",
        "skyblue",
        "deepskyblue",
        "dodgerblue",
        "turquoise",
        "springgreen",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "STRING": ("STRING", {"default": "", "forceInput": True}),
                "text_color": (cls.COLOR_CHOICES, {"default": "aqua"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "show_string"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    CATEGORY = "Mine"
    
    def detect_type(self, value):
        if isinstance(value, int):
            return 'integer'
        elif isinstance(value, float):
            # Check if it has a decimal part
            if value % 1 == 0:
                return 'float' if str(value).endswith('.0') else 'integer'
            return 'float'
        elif isinstance(value, str):
            try:
                float_val = float(value)
                if '.' in value:
                    return 'float string'
                if float_val.is_integer():
                    return 'integer string'
                return 'float string'
            except ValueError:
                return 'normal string'
        else:
            return 'other type'

    def show_string(self, STRING, text_color="aqua"):
        if isinstance(text_color, list):
            text_color = text_color[0] if text_color else "aqua"
        type_info = [f"{value}" for value in STRING]
        return {"ui": {"text": type_info, "text_color": [text_color]}}
