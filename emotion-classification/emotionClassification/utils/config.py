import json


class Config:
    """Config class which loads the configuration from a json file"""

    def __init__(self, data, train, output):
        self.data = data
        self.train = train
        self.output = output

    @classmethod
    def from_json(cls, cfg):
        """Creates a Config object from a json file"""
        params = json.loads(json.dumps(cfg), object_hook=HelperDict)
        return cls(params.data, params.train, params.output)


class HelperDict(object):
    """Helper class to convert json into a python object"""

    def __init__(self, dict_):
        self.__dict__.update(dict_)
