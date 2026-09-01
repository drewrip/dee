"""Backend configuration blocks: expansion, labelling and validation."""

from __future__ import annotations

import pytest

from dee_bench.config import (
    ConfigError,
    config_labels,
    expand_backend_config,
    setup_config,
)


class TestExpansion:
    def test_a_block_with_no_lists_is_one_configuration(self):
        assert expand_backend_config({"threads": 8}) == [{"threads": 8}]

    def test_an_empty_block_is_still_one_configuration(self):
        assert expand_backend_config({}) == [{}]

    def test_a_list_setting_expands(self):
        assert expand_backend_config({"max_memory": ["1GB", "8GB"], "threads": 8}) == [
            {"max_memory": "1GB", "threads": 8},
            {"max_memory": "8GB", "threads": 8},
        ]

    def test_nested_settings_expand_too(self):
        # Postgres server settings are a mapping of their own, and are as
        # worth sweeping as anything at the top level.
        got = expand_backend_config({"settings": {"work_mem": ["64MB", "1GB"]}})
        assert got == [
            {"settings": {"work_mem": "64MB"}},
            {"settings": {"work_mem": "1GB"}},
        ]

    def test_several_lists_cross_product(self):
        got = expand_backend_config({"threads": [4, 8], "max_memory": ["1GB", "8GB"]})
        assert len(got) == 4

    def test_an_empty_list_is_an_error(self):
        # Silently expanding to no configurations would drop every cell for
        # this backend without saying so.
        with pytest.raises(ConfigError, match="empty list"):
            expand_backend_config({"max_memory": []})


class TestLabels:
    def test_one_configuration_has_no_label(self):
        assert config_labels([{"threads": 8}]) == [""]

    def test_only_the_varying_setting_is_named(self):
        labels = config_labels([
            {"threads": 8, "max_memory": "1GB"},
            {"threads": 8, "max_memory": "8GB"},
        ])
        assert labels == ["max_memory=1GB", "max_memory=8GB"]

    def test_nested_settings_are_named_by_their_path(self):
        labels = config_labels([
            {"settings": {"work_mem": "64MB"}},
            {"settings": {"work_mem": "1GB"}},
        ])
        assert labels == ["settings.work_mem=64MB", "settings.work_mem=1GB"]


class TestSetupConfig:
    def test_duckdb_has_no_instance_to_configure(self):
        assert setup_config("duckdb", {"threads": 8, "max_memory": "8GB"}) == {}

    def test_postgres_connection_settings_are_not_part_of_setup(self):
        assert setup_config("postgres", {"num_connections": 16, "cpus": 8}) == {"cpus": 8}
