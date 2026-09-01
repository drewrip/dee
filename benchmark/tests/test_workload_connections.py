"""Per-cell backend tuning reaches the server through connections.json."""

from __future__ import annotations

import json

from dee_bench.workload import PreparedProject, apply_backend_config


def _prepared(tmp_path) -> PreparedProject:
    project_dir = tmp_path / "p01_iot"
    project_dir.mkdir()
    return PreparedProject(
        project="p01_iot", backend="duckdb", sf=0.1,
        project_dir=project_dir, src_dir=tmp_path / "src",
        dag_json=project_dir / "dag.json",
        connections_json=project_dir / "connections.json",
        target="dev", warehouse=project_dir / "warehouse.duckdb",
    )


def test_it_rewrites_the_connection_for_the_cells_tuning(tmp_path):
    prepared = _prepared(tmp_path)
    apply_backend_config(prepared, {"threads": 8, "max_memory": "1GB"})
    first = json.loads(prepared.connections_json.read_text())["dev"]
    assert first["max_memory"] == "1GB"

    # The same preparation, reused by a cell with a different ceiling: only
    # the connection changes, and it must not keep the previous cell's value.
    apply_backend_config(prepared, {"threads": 8, "max_memory": "16GB"})
    second = json.loads(prepared.connections_json.read_text())["dev"]
    assert second["max_memory"] == "16GB"
    assert second["database"] == first["database"]


def test_an_unset_ceiling_is_omitted_rather_than_sent_empty(tmp_path):
    prepared = _prepared(tmp_path)
    apply_backend_config(prepared, {"threads": 8})
    assert "max_memory" not in json.loads(prepared.connections_json.read_text())["dev"]
