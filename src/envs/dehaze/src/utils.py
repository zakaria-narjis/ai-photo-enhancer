from __future__ import print_function

import os


def format(src, *addition):
    """Format string according to their extension.

    Args:
        src (str): origin string to be formatted
        addition (str or 'type' could use `str(addition)`): target to be added to the `src`
    """  # noqa: E501
    base, ext = os.path.splitext(src)
    fstr = len(addition) * "-{}"
    return (base + fstr + ext).format(*addition)


def assure_dir(dir):
    """Ensure the existence of the dir."""  # noqa: E501
    if not os.path.exists(dir):
        os.mkdir(dir)
