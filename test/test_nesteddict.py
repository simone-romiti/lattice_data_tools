import pickle

import pytest

from lattice_data_tools.dictionaries import NestedDict


def test_default_nested_dict_auto_vivifies():
    d = NestedDict()
    d["a"]["b"] = 1
    assert d["a"]["b"] == 1
    assert d["a", "b"] == 1
    assert d.to_dict() == {"a": {"b": 1}}


def test_rigid_keys_is_chainable_and_returns_same_instance():
    d = NestedDict()
    assert d.rigid_keys() is d


def test_rigid_keys_raises_on_missing_key():
    d = NestedDict()
    d["a"]["b"] = 1
    d.rigid_keys()

    with pytest.raises(KeyError):
        _ = d["missing"]

    with pytest.raises(KeyError):
        _ = d["a"]["missing"]

    with pytest.raises(KeyError):
        _ = d["a", "missing"]


def test_rigid_keys_present_keys_still_work():
    d = NestedDict()
    d["a"]["b"] = 42
    d.rigid_keys()

    assert d["a"] == {"b": 42}
    assert d["a"]["b"] == 42
    assert d["a", "b"] == 42


def test_rigid_keys_propagates_to_existing_nested_dicts():
    d = NestedDict()
    d["a"]["b"] = 1
    d.rigid_keys()
    assert d["a"]._rigid_keys is True


def test_loose_keys_reverts_to_auto_vivifying():
    d = NestedDict()
    d["a"]["b"] = 1
    d.rigid_keys()
    d.loose_keys()

    d["new"]["key"] = 2
    assert d["new"]["key"] == 2


def test_load_from_pickle_then_rigid_keys(tmp_path):
    d = NestedDict()
    d["ensemble"]["obs"]["value"] = 3.14
    d["ensemble"]["obs"]["error"] = 0.02

    pkl_path = tmp_path / "nested.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(d, f)

    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)

    loaded.rigid_keys()

    # present keys work
    assert loaded["ensemble", "obs", "value"] == 3.14
    assert loaded["ensemble"]["obs"]["error"] == 0.02

    # missing keys raise
    with pytest.raises(KeyError):
        _ = loaded["ensemble", "missing"]

    with pytest.raises(KeyError):
        _ = loaded["missing"]
