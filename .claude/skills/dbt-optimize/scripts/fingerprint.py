#!/usr/bin/env python3
"""Fingerprint every model's relation, so a rewrite can be falsified.

For each model it runs one query returning an order-independent digest of the
whole relation: `count(*)` plus a sum of per-row hashes.  Sums, not xor, so
duplicate rows cannot cancel; no ORDER BY, so a 4M-row relation is not sorted.

Columns are normalized before hashing, using types read from
information_schema in a single query:

  * float / real / double  ->  rounded to --round decimals (default 6)
  * everything else        ->  cast to text
  * NULL                   ->  a literal marker, so NULL and '' differ

The rounding matters.  Materializing an intermediate view as a table changes
the order the engine aggregates in, and floating-point addition is not
associative, so a downstream `avg()` legitimately lands a few ulps away.  An
unrounded digest reports that as "the relation changed", which it is not.
The flip side: rounding can hide a real difference below the last kept
decimal, so treat --round as the tolerance you are choosing.

Executed through `dbt show --inline`, which uses the project's own profile, so
duckdb and postgres both work with no second set of credentials.

    fingerprint.py --project P --out .opt/fp_baseline.json
    # ... change models, rebuild ...
    fingerprint.py --project P --out .opt/fp_candidate.json
    fingerprint.py --diff .opt/fp_baseline.json .opt/fp_candidate.json

A matching fingerprint is NOT proof of equivalence -- it is one sample of one
dataset.  It falsifies; it does not verify.  The argument for equivalence has
to be made about the SQL.  See reference/equivalence.md.

Note: `dbt show` overwrites target/run_results.json.  Take timing measurements
before fingerprints, or copy run_results.json aside first.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

NULL_MARK = "\\N"
FLOAT_TYPES = ("double", "float", "real")

# Type names differ per adapter: postgres has no `double`, only `double
# precision`, and casting to `varchar` vs `text` is likewise adapter-specific.
TYPE_NAMES = {
    "duckdb": {"float": "double", "text": "varchar"},
    "postgres": {"float": "double precision", "text": "text"},
}

# How each adapter turns a normalized row-string `s` into a summable number.
ROW_HASH = {
    "duckdb": "sum(hash({s})::hugeint)",
    "postgres": "sum(('x' || substr(md5({s}), 1, 15))::bit(60)::bigint::numeric)",
}


def dbt_bin(explicit):
    if explicit:
        return explicit
    for cand in (os.path.join(os.path.dirname(sys.executable), "dbt"), shutil.which("dbt")):
        if cand and os.path.exists(cand):
            return cand
    sys.exit("no dbt on PATH -- pass --dbt")


def show(dbt, project, sql, target, limit=1000):
    """Run one statement through `dbt show --inline` and return its rows."""
    argv = [dbt, "show", "--inline", sql, "--output", "json", "--limit", str(limit)]
    if target:
        argv += ["--target", target]
    p = subprocess.run(argv, cwd=project, capture_output=True, text=True)
    m = re.search(r'\{\s*"show":.*?\n\}', p.stdout, re.S)
    if not m:
        # Servers emit WARNING/DETAIL/HINT chatter (a collation mismatch, say)
        # that can fill the whole tail and hide the actual failure.
        noise = re.compile(r"^\s*(WARNING|NOTICE|DETAIL|HINT|CONTEXT):", re.M)
        lines = [l for l in (p.stdout + p.stderr).splitlines()
                 if l.strip() and not noise.match(l)]
        raise RuntimeError("\n".join(lines)[-1200:])
    return json.loads(m.group(0))["show"]


def manifest_models(project, select):
    with open(os.path.join(project, "target", "manifest.json")) as f:
        manifest = json.load(f)
    out = {}
    for uid, node in manifest["nodes"].items():
        if node["resource_type"] != "model":
            continue
        if select and node["name"] not in select:
            continue
        out[node["name"]] = {
            "identifier": node.get("alias") or node["name"],
            "schema": node["schema"],
            "mat": (node.get("config") or {}).get("materialized", "view"),
        }
    return manifest["metadata"]["adapter_type"], out


def fetch_columns(dbt, project, models, target):
    """{identifier: [(column, type), ...]} for every model, in one query."""
    schemas = sorted({m["schema"] for m in models.values()})
    idents = sorted({m["identifier"] for m in models.values()})
    lit = lambda xs: ", ".join("'" + x.replace("'", "''") + "'" for x in xs)
    sql = (
        "select table_name, column_name, data_type, ordinal_position "
        "from information_schema.columns "
        f"where table_schema in ({lit(schemas)}) and table_name in ({lit(idents)}) "
        "order by table_name, ordinal_position"
    )
    rows = show(dbt, project, sql, target, limit=100000)
    cols = {}
    for r in rows:
        cols.setdefault(r["table_name"], []).append((r["column_name"], r["data_type"]))
    return cols


def normalize(col, dtype, adapter, ndigits):
    """SQL expression rendering one column as a comparable string."""
    q = f'"{col}"'
    t = (dtype or "").lower()
    names = TYPE_NAMES[adapter]
    if any(f in t for f in FLOAT_TYPES):
        # numeric() so the rounding is exact and prints without exponent
        inner = f"round(cast({q} as numeric), {ndigits})"
    else:
        inner = q
    return f"coalesce(cast({inner} as {names['text']}), '{NULL_MARK}')"


NUMERIC_TYPES = (
    "int", "numeric", "decimal", "double", "real", "float", "hugeint", "serial",
)


def is_numeric(dtype):
    t = (dtype or "").lower()
    return any(k in t for k in NUMERIC_TYPES) and "interval" not in t


def digest_sql(name, cols, adapter, ndigits, ignore=()):
    """One scan producing the exact row digest AND per-column aggregates.

    The row digest answers "identical?".  The aggregates answer the follow-up
    question when it is not -- "identical up to float noise, or actually
    different?" -- without a second pass over the relation.
    """
    names = TYPE_NAMES[adapter]
    cast, fl = names["text"], names["float"]
    cols = [(c, t) for c, t in cols if c not in ignore]
    if not cols:
        return None
    parts = [normalize(c, t, adapter, ndigits) for c, t in cols]
    row_str = "concat_ws('|', " + ", ".join(parts) + ")"
    sel = ["count(*) as n", ROW_HASH[adapter].format(s=row_str) + " as h"]
    for i, (c, t) in enumerate(cols):
        q = f'"{c}"'
        sel.append(f"count({q}) as c{i}_nn")
        if is_numeric(t):
            sel.append(f"sum(cast({q} as {fl})) as c{i}_sum")
            sel.append(f"min(cast({q} as {fl})) as c{i}_min")
            sel.append(f"max(cast({q} as {fl})) as c{i}_max")
        else:
            sel.append(f"min(cast({q} as {cast})) as c{i}_min")
            sel.append(f"max(cast({q} as {cast})) as c{i}_max")
    return "select " + ", ".join(sel) + " from {{ ref('%s') }}" % name


def collect(args):
    project = os.path.abspath(args.project)
    dbt = dbt_bin(args.dbt)
    adapter, models = manifest_models(project, args.select)
    if adapter not in ROW_HASH:
        sys.exit(f"no digest defined for adapter {adapter!r}")
    columns = fetch_columns(dbt, project, models, args.target)

    ignore = set(args.ignore_cols or ())
    out = {"adapter": adapter, "round": args.round,
           "ignored_cols": sorted(ignore), "models": {}}
    for name in sorted(models):
        ident = models[name]["identifier"]
        cols = columns.get(ident)
        if not cols:
            out["models"][name] = {"error": "no columns in information_schema -- not built?"}
            print(f"  {name:<34} MISSING", file=sys.stderr)
            continue
        try:
            sql = digest_sql(name, cols, adapter, args.round, ignore)
            if sql is None:
                raise RuntimeError("every column ignored")
            row = show(dbt, project, sql, args.target)[0]
        except RuntimeError as e:
            out["models"][name] = {"error": str(e)[-400:]}
            print(f"  {name:<34} ERROR", file=sys.stderr)
            continue
        aggs = {k: (str(v) if isinstance(v, str) else v) for k, v in row.items()
                if k not in ("n", "h")}
        out["models"][name] = {
            "n": row["n"],
            "h": str(row["h"]),
            "cols": [f"{c}:{t}" for c, t in cols],
            "aggs": aggs,
            "mat": models[name]["mat"],
        }
        print(f"  {name:<34} n={row['n']:<12} {len(cols)} cols", file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {args.out}")


def agg_delta(fa, fb, rtol):
    """Largest relative difference between two models' column aggregates.

    Returns (worst_relative_difference, offending_key) -- or (None, key) when
    a non-numeric aggregate differs outright, which no tolerance excuses.
    """
    worst, where = 0.0, None
    for k, va in (fa.get("aggs") or {}).items():
        vb = (fb.get("aggs") or {}).get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            scale = max(abs(va), abs(vb), 1e-12)
            rel = abs(va - vb) / scale
            if rel > worst:
                worst, where = rel, k
        elif va != vb:
            return None, k
    return worst, where


def diff(a_path, b_path, rtol=1e-9):
    a, b = (json.load(open(p)) for p in (a_path, b_path))
    ia, ib = a.get("ignored_cols") or [], b.get("ignored_cols") or []
    if ia != ib:
        print(f"WARNING  the two sides ignored different columns: {ia} vs {ib}")
    elif ia:
        print(f"note: {len(ia)} column(s) excluded from every digest: {', '.join(ia)}\n")
    names = sorted(set(a["models"]) | set(b["models"]))
    PERSISTED = {"table", "incremental", "materialized_view", "snapshot"}
    bad = 0
    close = 0
    added = 0
    for name in names:
        fa, fb = a["models"].get(name), b["models"].get(name)
        if fa is None:
            # Adding models is allowed: the original models are a floor, not a
            # ceiling. Report it so the diff still accounts for everything.
            print(f"ADDED    {name}  -- new model, not in the original (allowed)")
            added += 1
            continue
        if fb is None:
            print(f"DROPPED  {name}  -- no relation with this name in the candidate")
            bad += 1
            continue
        if "error" in fa or "error" in fb:
            print(f"UNKNOWN  {name}  -- {fa.get('error') or fb.get('error')}")
            bad += 1
            continue
        if fa["cols"] != fb["cols"]:
            only_a = [c for c in fa["cols"] if c not in fb["cols"]]
            only_b = [c for c in fb["cols"] if c not in fa["cols"]]
            print(f"SCHEMA   {name}  -{only_a} +{only_b}")
            bad += 1
        elif fa["n"] != fb["n"]:
            print(f"DIFFERS  {name}  row count {fa['n']} -> {fb['n']}")
            bad += 1
        elif fa["h"] != fb["h"]:
            rel, key = agg_delta(fa, fb, rtol)
            if rel is None:
                print(f"DIFFERS  {name}  non-numeric aggregate {key} changed")
                bad += 1
            elif rel <= rtol:
                print(
                    f"CLOSE    {name}  digest differs, every column aggregate "
                    f"within {rel:.1e} (worst: {key})"
                )
                close += 1
            else:
                print(
                    f"DIFFERS  {name}  column {key} off by {rel:.3e} relative "
                    f"(tolerance {rtol:.0e})"
                )
                bad += 1
        elif fa["mat"] in PERSISTED and fb["mat"] not in PERSISTED:
            print(
                f"UNTABLED {name}  -- was {fa['mat']}, now {fb['mat']}: "
                "the declared materialization contract is broken"
            )
            bad += 1
    checked = len(names) - added
    ok = checked - bad
    notes = []
    if close:
        notes.append(f"{close} only float-noise apart")
    if added:
        notes.append(f"{added} added, not in the original")
    note = f" ({'; '.join(notes)})" if notes else ""
    print(f"\n{ok}/{checked} original models pass{note}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--out")
    ap.add_argument("--select", nargs="*", help="model names (default: all)")
    ap.add_argument("--round", type=int, default=6, help="decimals kept for float columns")
    ap.add_argument("--ignore-cols", nargs="*", metavar="COL",
                    help="column names to leave out of the digest entirely. For "
                         "columns that are nondeterministic by design -- a "
                         "`current_timestamp as generated_at` audit column, a "
                         "uuid, a row number over an unstable order. The "
                         "self-diff is what tells you which these are: an "
                         "unchanged project that fails its own diff on a text "
                         "or timestamp aggregate has one. They stay in the "
                         "recorded schema, so a column appearing or vanishing "
                         "is still caught.")
    ap.add_argument("--target")
    ap.add_argument("--dbt")
    ap.add_argument("--diff", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--rtol", type=float, default=1e-9,
                    help="relative tolerance on column aggregates when the exact "
                         "digest differs (default 1e-9)")
    args = ap.parse_args()

    if args.diff:
        sys.exit(diff(*args.diff, rtol=args.rtol))
    if not args.out:
        sys.exit("--out is required")
    collect(args)


if __name__ == "__main__":
    main()
