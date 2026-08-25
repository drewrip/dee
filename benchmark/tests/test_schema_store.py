"""Result schemas and the incremental parquet store."""

from __future__ import annotations

import pytest

from dee_bench import schema
from dee_bench.schema import Verbosity
from dee_bench.store import ResultStore, connect


class TestSchema:
    def test_every_column_is_documented(self):
        for table in schema.ALL_TABLES:
            for col in table.columns:
                assert col.doc.strip(), f"{table.name}.{col.name} has no documentation"

    def test_verbosity_levels_are_nested(self):
        prev: set[str] = set()
        for v in Verbosity:
            names = {t.name for t in schema.tables_for(v)}
            assert prev <= names, f"{v.name} dropped tables present at a lower verbosity"
            prev = names

    def test_partitioned_tables_carry_the_partition_column(self):
        for t in schema.ALL_TABLES:
            for key in t.partition_by:
                assert key in t.column_names

    def test_derived_tables_are_not_recorded_by_the_runner(self):
        for v in Verbosity:
            assert not any(t.derived for t in schema.tables_for(v))

    def test_parse_rejects_unknown_verbosity(self):
        with pytest.raises(ValueError, match="unknown verbosity"):
            Verbosity.parse("chatty")


class TestStore:
    def test_verbosity_gates_what_is_written(self, tmp_path):
        st = ResultStore(tmp_path, Verbosity.SUMMARY)
        assert st.records("runs")
        assert not st.records("system_samples")
        assert st.write("system_samples", [{"cell_id": "a", "elapsed_ms": 1}]) == 0

    def test_missing_columns_are_filled_not_dropped(self, tmp_path):
        st = ResultStore(tmp_path, Verbosity.FULL)
        st.write("runs", [{"cell_id": "a", "run_id": "r", "engine_wall_ms": 10}])
        row = connect(tmp_path).sql("SELECT * FROM runs").fetchall()[0]
        assert len(row) == len(schema.RUNS.columns)

    def test_unknown_columns_are_rejected(self, tmp_path):
        st = ResultStore(tmp_path, Verbosity.FULL)
        with pytest.raises(ValueError, match="unknown column"):
            st.write("runs", [{"cell_id": "a", "not_a_column": 1}])

    def test_fragments_from_many_cells_read_back_together(self, tmp_path):
        st = ResultStore(tmp_path, Verbosity.FULL)
        for cid in ("aaa", "bbb", "ccc"):
            st.write("runs", [{"cell_id": cid, "run_id": f"r-{cid}", "engine_wall_ms": 5}])
        con = connect(tmp_path)
        assert con.sql("SELECT count(*) FROM runs").fetchone()[0] == 3
        assert st.cell_ids_written() == {"aaa", "bbb", "ccc"}

    def test_appending_to_one_cell_does_not_overwrite(self, tmp_path):
        st = ResultStore(tmp_path, Verbosity.FULL)
        st.write("runs", [{"cell_id": "a", "run_id": "r1"}])
        st.write("runs", [{"cell_id": "a", "run_id": "r2"}])
        assert connect(tmp_path).sql("SELECT count(*) FROM runs").fetchone()[0] == 2

    def test_derived_tables_are_replaced_not_appended(self, tmp_path):
        st = ResultStore(tmp_path, Verbosity.FULL)
        st.write_derived("payback", [{"project": "p01_iot", "payback_runs_wall": 10.0}])
        st.write_derived("payback", [{"project": "p01_iot", "payback_runs_wall": 20.0}])
        rows = connect(tmp_path).sql("SELECT payback_runs_wall FROM payback").fetchall()
        assert rows == [(20.0,)]

    def test_no_results_reads_back_empty_rather_than_erroring(self, tmp_path):
        con = connect(tmp_path)
        assert con.sql("SHOW TABLES").fetchall() == []
