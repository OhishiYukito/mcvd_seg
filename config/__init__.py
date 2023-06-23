# https://github.com/voletiv/mcvd-pytorch/blob/451da2eb635bad50da6a7c03b443a34c6eb08b3a/main.py#L359

import argparse

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace