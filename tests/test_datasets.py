import pytest

from hcrd.datasets import select_cwru_drive_key


def test_cwru_key_selection_uses_record_id_when_file_has_copied_variables():
    keys = ["X098_DE_time", "X098_FE_time", "X099_DE_time", "X099_FE_time"]
    assert select_cwru_drive_key(99, keys) == "X099_DE_time"


def test_cwru_key_selection_fails_loudly_when_expected_record_is_absent():
    with pytest.raises(RuntimeError, match="X100_DE_time"):
        select_cwru_drive_key(100, ["X099_DE_time"])
