# Proving you did not change the answer

Constraint 4 is the one that makes this task hard. "Same results" is not a
property you can test your way to - it is a property of the SQL. The
fingerprint is a fast way to be *caught*; it is not a way to be *right*.

## What `fingerprint.py` does

For every model, one query returning:

- `count(*)`
- a sum of per-row hashes over all columns, normalized: floats rounded to
  `--round` decimals, everything else cast to text, NULL given a distinct
  marker
- per-column aggregates (`sum`/`min`/`max` for numerics, `min`/`max` and
  non-null count otherwise)
- the column list with types, from `information_schema`

Sum rather than xor, so duplicate rows cannot cancel. No `ORDER BY`, so a
multi-million-row relation is not sorted to be checked - the digest is
order-independent, which is also correct: row order is not part of a relation.

`--diff` reports, per model:

| verdict | meaning |
|---|---|
| pass | identical digest, identical schema |
| `CLOSE` | digest differs, every column aggregate within `--rtol` - float noise |
| `DIFFERS` | row count, schema, a text aggregate, or a numeric aggregate beyond tolerance changed |
| `SCHEMA` | column names or types changed |
| `DROPPED` | an original model is gone - violates constraint 4 |
| `ADDED` | a model the original did not have - allowed, reported for completeness |
| `UNTABLED` | was a persisted materialization, now is not - constraint 5 |

## What it proves: nothing, on its own

Three separate reasons to distrust a green diff.

**One dataset.** The digest is evaluated over the data that happens to be
loaded. A rewrite that is wrong only when a join key duplicates, when a value
is NULL, when a group is empty, or when a window frame has ties, passes
cleanly against data that never exercises the case. Matching output over one
dataset - or the same dataset several times - is a *failed falsification*.

**Aggregates hide.** Two relations with the same count, the same per-column
sums and extremes, and different rows are easy to construct. The row-hash
catches that in the exact case, but once you have loosened to `--rtol` for a
float-unstable model, you are comparing aggregates only.

**Tolerance hides.** `--round` and `--rtol` are the size of the difference you
have agreed not to see. Set them from the measured self-diff floor and no
looser, and never apply a model's loose tolerance to models that were exactly
reproducible.

## The self-diff, first

Before comparing anything, rebuild the untouched project and fingerprint it
twice. Real projects are usually not bit-reproducible: parallel float
aggregation is not associative, so `sum()` and `avg()` over a large relation
land a few ulps apart per run, and relative error is amplified wherever a sum
nearly cancels. In one 12-model benchmark project the unchanged DAG moved
three models by up to 5e-5 relative between consecutive identical runs.

That measurement gives you three things:

1. the tolerance floor to pass as `--rtol`
2. the list of models that are inherently unstable - for those, a `CLOSE`
   verdict is expected and means nothing suspicious
3. the list of models that are exactly reproducible - for those, *any*
   digest change is a real finding, and must be explained before you keep the
   change

Do not skip this and then explain away failures afterwards. Establishing the
floor first is what makes a later failure informative.

## What actually licenses a rewrite

Separate two kinds of claim, because they have different evidence bars.

**Cost claims** - this relation is small, this join fans out 8x, this filter
keeps 2% of rows, this node takes 0.5s. Measure these from the data freely.
They decide *which* of several equivalent plans is fastest, and being wrong
about one costs you a wasted trial, nothing more. All of step 3's cost model
is claims of this kind.

**Correctness claims** - this key is unique, this join cannot duplicate rows,
this column is never NULL, these branches are disjoint. These decide *whether*
a rewrite is equivalent at all, and being wrong about one silently corrupts
the output. Observing that a property holds in the current data is not enough:
the same rewrite has to be correct on next month's data, which nobody has
looked at.

So when you find a correctness-relevant property by looking at the data, you
have two honest options:

1. **Find the existing guarantee** - a declared key or constraint, a `unique`
   / `not_null` / `relationships` test already in `schema.yml`, or a
   structural argument from the query itself (an upstream `group by` on those
   exact columns does guarantee uniqueness).
2. **Create the guarantee.** Add the test that asserts the property, in the
   same change as the rewrite that depends on it. A `unique` test on the join
   key turns "it happens to hold today" into "the project fails loudly the day
   it stops holding", which is exactly the difference between an assumption
   and a precondition. The test costs one cheap query per `dbt build`, and it
   is the licence for the rewrite - name it in the report.

What is never acceptable is relying on the property silently. If neither
option is available, the rewrite is a guess; leave the model alone and spend
the effort on a plan change instead, which needs no correctness argument at
all.

For every SQL change, be able to state the transformation and its
precondition:

| rewrite | precondition |
|---|---|
| drop `order by` | no consumer depends on order (SQL does not guarantee it through a view anyway) |
| drop `distinct` | the projection is already unique - a declared key, a `unique` test, or an upstream `group by` on those columns |
| `union` -> `union all` | branches provably disjoint by a discriminating predicate |
| push a predicate below a join | the predicate references only one side, and that side is not the null-extended side of an outer join |
| push a predicate below an aggregate | it references only grouping columns |
| prune a column | no downstream model references it, textually, anywhere - including through `select *` |
| aggregate before joining | the aggregate reads only its own side, and the join cannot change multiplicity on that side (a key join, not a fan-out) |
| self-join -> window | frame and tie semantics match: `lag` vs `max(...) over` differ on duplicate ordering keys |
| correlated subquery -> join | no-match rows still produce NULL, i.e. a left join, not an inner one |
| CTE folding / inlining | referenced expressions are deterministic; a CTE with a `limit`, a window, or a volatile function is not freely inlinable |
| materialize a view | none needed for the relation, but on postgres it changes downstream *plans* - `ANALYZE`, and it changes float aggregation order |

If you cannot fill in the precondition column for a change you want to make,
you do not have a rewrite - you have a guess. Find the guarantee, add the test
that creates it, or leave the model alone.

Note what is absent from that table: nothing licenses a *materialization*
change, because there is nothing to license. Moving a node between view and
table, adding a new model and pointing an original at it, changing threads or
adapter settings - none of these can change a relation, so none of them need a
correctness argument. That asymmetry is worth exploiting: plan changes are
where the wins are cheapest to justify.

## Free evidence you already have

- **The project's own tests.** `dbt build` runs `schema.yml` tests -
  uniqueness, not-null, relationships, accepted values. They are already
  written and they check exactly the invariants a rewrite is most likely to
  break. Run `dbt build` at least once before declaring done, and treat a
  newly failing test as a correctness failure, not a flaky test.
- **A second scale factor.** If the sources can be regenerated at a different
  size, rebuild both variants at a second scale and
  fingerprint again. It costs a load and two runs, and it is by far the
  strongest falsification available - most overfitted rewrites die here,
  because the fan-out, tie, and empty-group cases that one dataset never
  produced show up in another.
- **Constructed edge cases.** For a specific risky rewrite, a handful of rows
  with a duplicate key, a NULL, and a tie will falsify it faster than any
  amount of production data. Cheap, targeted, and it costs a few statements.

## Reporting equivalence

The final report should say, for each change: the transformation, its
precondition, and what evidence exists. "Fingerprints match" alone is not a
claim of equivalence - it is one failed attempt at falsification, and it should
be reported as exactly that.
