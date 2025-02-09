from typing import Dict


__all__ = ["ANSIColor"]


class ANSIColor(object):
    __SGR = "\033[38;2;"
    __default_rgb_code: Dict[str, str] = {
        "red": "255;0;0",
        "green": "0;255;0",
        "yellow": "255;255;0",
        "cyan": "0;255;255"
    }
    RESET = "\033[0m"

    def __init__(self, rgb_code: Dict[str, str] = None) -> None:
        super(ANSIColor, self).__init__()

        if rgb_code is None:
            rgb_code = {}

        rgb_code = {**self.__default_rgb_code, **rgb_code}

        for color in rgb_code:
            val: str = self.__SGR + rgb_code[color] + "m"

            if getattr(self, color.upper(), None) is None:
                setattr(self, color.upper(), val)
