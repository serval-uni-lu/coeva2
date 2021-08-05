from src.config_parser.config_parser import get_dict_hash


def test_get_dict_hash_sample():
    assert get_dict_hash({"a": 1, "b": 2}) == "8aacdb17187e6acf2b175d4aa08d7213"


def test_get_dict_hash_order():
    assert get_dict_hash({"a": 1, "b": 2}) == get_dict_hash({"b": 2, "a": 1})
