# The optimization catalogue

Ordered by (expected win) / (cost to find). Each entry says what it does, when
it applies, and what could go wrong. None of these are automatic - the cost
model ranks them, the measurement decides, the fingerprint gates them.

---

## A. Free static fixes

No search, no measurement to *find* them - only one measurement to confirm the
bundle. Do these first and bundle them into a single candidate.

**Drop `order by` from non-final models.** Sorting is O(n log n) on rows that
a consumer immediately regroups or rejoins. A model's row order is not part of
its relation unless something downstream depends on it - and SQL gives no such
guarantee through a view anyway. Keep it only in a final reporting model where
the user's contract implies it. Often the single largest free win on wide
staging models.

**Prune unread columns.** A staging view doing `select *` over a 40-column
source drags every column through every layer above it. On a column store
(duckdb) unread columns are usually never read, so the win is small; on
postgres, and everywhere a node is *materialized*, the pruned columns are rows
you never write. Derive the keep-list from what downstream models actually
reference - textually, over the whole project, including `select *` consumers
which force you to keep everything.

**Hoist filters below joins and aggregations.** `where` on a join result that
only constrains one side can be applied to that side first. Engines do much of
this themselves; they do it much less reliably through aggregations, `distinct`,
window functions, and outer joins - which is exactly where it pays. Careful:
pushing a predicate through the null-extended side of an outer join changes
the result. That is a rewrite that needs an argument.

**Remove redundant `distinct` / `group by`.** If the key is already unique
(a declared primary key, a `unique` test in `schema.yml`, or an upstream
`group by` on the same key), the dedup is a full sort or hash for nothing.
The `schema.yml` tests are your evidence; if there is no test, there is no
guarantee.

**Compute a shared expression once.** The same scalar expression repeated
across a `select` list and a `where`/`group by` is usually free (engines CSE
it), but the same *subquery* written twice often is not. Fold it into a CTE -
but note that in postgres a CTE referenced twice is materialized once, while
in duckdb the optimizer may inline it twice. Verify by measuring.

**`union` -> `union all`** only where the branches are provably disjoint.
`union` deduplicates, which is a full sort/hash of the combined input. Needs
an argument (a discriminating column with disjoint predicates), never an
observation.

---

## B. Materialization placement

The main structural lever, and what the cost model in
`reference/measurement.md` exists to rank.

**Check achieved concurrency first. It decides whether any of this pays.**

An expansion count measures duplicated *work*. It does not measure duplicated
*time*. When a DAG runs several nodes at once, the duplicated work is being
done in parallel on threads that would otherwise be idle - it is already free
in elapsed terms. Materializing it converts that free parallel work into a
serialization barrier: every consumer now waits for the write to finish
before it can start.

Measured across five postgres projects, materializing the top-ranked node:

| concurrency | wall clock | database work |
|---|---|---|
| 1.01 | **-24.5%** | -27.1% |
| 2.08 | +26.9% | +5.5% |
| 2.29 | -1.7% | -37.6% |
| 2.69 | +65.9% | +7.0% |
| 2.93 | +20.3% | -38.2% |

Database work fell in most of them - by as much as 38% - and wall clock still
got worse. The single win was the one DAG running at concurrency ~1.0, where
there was no parallelism to lose and the recomputation was strictly serial.

So: **materialization is a strong candidate when `decompose.py` reports
concurrency near 1, and a poor one when it reports 2 or more.** At high
concurrency, spend your budget on making the hot nodes' SQL cheaper instead,
or accept that the DAG is already near its floor.

Note the two signals diverging in that table. If you are optimizing database
cost rather than latency - a metered warehouse, a shared cluster - the same
change that costs 20% wall clock buys back 38% of the work. Say which signal
you are optimizing before you call one of these a regression.

**View -> table.** Given concurrency ~1, worth it when `expansions(v) > 1` and
`c(v)` is large: the work is done once instead of `expansions(v)` times, at
the price of writing `|v|` rows once. Roughly, materialize when
`c(v) * (expansions(v) - 1) > write_cost(v)`. Wide intermediates with high
row counts have expensive writes; narrow aggregates are cheap to write and
usually the best candidates.

**View -> table for parallelism, even at expansion 1.** A chain of views
ending in one table is *one query on one thread*. Materializing partway down
splits it into two nodes that dbt can schedule - and, more importantly, breaks
a single giant plan into two the engine can optimize and pipeline separately.
Under a high thread count with independent branches this can win even when it
strictly duplicates work.

**Table -> view.** Legal only for tables *you* introduced, never for a
user-declared table (constraint 5). If a probe shows a materialized
intermediate is read exactly once and its write dominates its compute, inline
it.

**`ephemeral`.** dbt inlines an ephemeral model as a CTE and creates no
relation at all - the cheapest possible node. Not available for an original
model, which must keep its relation (constraint 4), but perfectly good for a
model *you* added and nobody outside the project references.

**`incremental`** is the user's decision, not yours. See the hard rules in
SKILL.md.

---

## B2. Adding models

Constraint 4 is one-directional: every original model must survive, and new
ones are free. This is the sharpest structural tool available, because the
original author's model boundaries are wherever the business logic happened to
be cut - not where the *plan* wants to be cut.

**Materialize at the right cut point.** You want to persist a shared subtree,
but the node that computes it also does expensive per-consumer work you do not
want to persist, or the shared part is only half of some model's SQL. Split
it: add a new model holding exactly the shared prefix, materialize *that*,
and rewrite the original model to `ref` it. The original model keeps its name
and its relation; the expensive shared part is now computed once. The cost
model in `reference/measurement.md` prices this the same as any other
materialization - the difference is that you get to choose `v` rather than
take the ones you were given.

**Common subexpression extraction.** Two or more models containing the same
expensive subquery, join, or aggregation spelled out inline. The engine will
not share that work across separate statements - each materialized consumer
plans it independently. Factor it into one new model, materialize it, and
point both at it. `analyze.py`'s expansion counts do not see this case,
because textually-duplicated SQL is not a shared DAG node - you find it by
reading the models. It is often the largest single win in a hand-written
project, precisely because no tool reports it.

**Break a serial chain into parallel branches.** A single model computing
several independent aggregations over the same input is one query on one
thread. Splitting the independent parts into new models lets dbt schedule
them concurrently, and lets each get a plan suited to its own shape. The
original model becomes a join over the new ones - cheap if the pieces are
already grouped to the same key. Costs one extra node's overhead per split,
so it pays only when the pieces are substantial.

**Narrow a wide shared input.** If a large staging model is consumed by five
downstream models that between them read six of its forty columns, a new
narrow model over the same source - materialized - can be much cheaper to
read repeatedly than the wide one, without touching the wide model's own
relation.

Two cautions. Every added model costs dbt's fixed per-node overhead
(transaction, create, catalog lookup, commit), which on a small DAG is not
negligible - do not shard a fast DAG into thirty nodes. And a new model is new
SQL: it needs the same equivalence argument as any rewrite, because the
original model now depends on it.

---

## C. Adapter and engine settings

Frequently the biggest wins per unit of search, because one config change
covers the whole DAG and costs one measurement to test. All of these are
project-level configuration, not machine changes, so they are inside the
constraints.

### postgres

**`jit = off`.** Postgres enables JIT for queries the planner thinks are
expensive, and on analytic queries the compilation frequently costs more than
it saves - sometimes a 2x difference on a large aggregation. Set it per model
with a pre-hook (`set local jit = off`) or globally in the profile. Test it;
it is one measurement and it either lands or it does not.

**`ANALYZE` after materializing.** A `create table as` leaves postgres with no
statistics on the new relation until autovacuum eventually notices. Every
downstream join then plans against a default estimate, which on a large
intermediate is how you get a nested loop over millions of rows. A
`post-hook: "analyze {{ this }}"` on materialized models is cheap and can be
worth more than the materialization itself. **If you materialize on postgres,
analyze.**

**`+unlogged: true`.** dbt-postgres supports unlogged tables. They skip WAL,
which is most of the write cost, and they are still tables - constraint 5 is
satisfied. The tradeoff is they do not survive a crash, which for a rebuildable
intermediate is not a tradeoff at all. Consider it for intermediates first;
for a user-declared table, ask.

**`+indexes`.** dbt-postgres takes an `indexes` config per model. An index on
a join key of a materialized intermediate can turn a hash join over the whole
relation into an index lookup. Costs build time to create, so it only pays when
the consumer is selective. Rarely the first thing to try, occasionally decisive.

**`work_mem` per model.** `pre-hook: "set local work_mem = '256MB'"` keeps a
hash join or sort in memory instead of spilling to disk. Bounded by the fixed
machine's RAM and multiplied by concurrent threads - raising it too far with
16 threads is how you make everything slower. Test at one or two values.

### duckdb

**`threads` and `memory_limit`** in the profile's `settings:` block. duckdb
parallelizes within a single query, so its own thread count usually matters
more than dbt's.

**`preserve_insertion_order = false`.** Lets duckdb write results out of order,
which removes a synchronization point on large materializations. Safe with
respect to relational equivalence - row order was never guaranteed - but note
it may change which run-to-run float aggregation order you get, so re-check the
value noise floor after enabling it.

**`temp_directory`** so large joins spill to disk instead of failing or
thrashing, when `memory_limit` is tight.

**dbt `threads`.** dbt-duckdb serializes work through one connection to a
single database file, so raising dbt's thread count buys less than it does on
postgres. Measure rather than assume.

### both

**dbt `threads`.** Costs nothing, changes no SQL, and the DAG's shape decides
whether it helps: a linear chain gains nothing, a wide fan-out gains a lot.
Free to test, and worth testing early because it changes the ranking of every
materialization candidate that exists to create parallelism.

---

## D. Structural rewrites

Highest risk. Each needs an argument in `reference/equivalence.md`'s terms,
and a re-fingerprint. Worth it when the probe shows one node dominating the
DAG and no config change moves it.

**Merging a one-to-one view chain is not available.** Folding the lower model
into the upper one would delete it, and constraint 4 forbids that. The
equivalent win is usually free anyway: leave both models, make sure neither is
materialized, and the engine fuses them into one plan itself. Go the other
direction instead - see "adding models" above.

**Self-join -> window function.** `a join a on a.id = b.id and b.ts < a.ts`
patterns computing a previous value, running total, or rank are one scan with
a window function instead of a join over the relation with itself. Big win on
large relations. Equivalence hinges on tie handling and frame boundaries -
`lag` vs `max(...) over` differ on duplicates.

**Correlated subquery -> join or window.** A scalar subquery in a select list
executes per outer row in the worst case. Rewriting to a grouped join is a
classic order-of-magnitude change. Watch the outer-join semantics: a
correlated subquery returns NULL for no match, an inner join drops the row.

**Aggregate before joining.** Joining two large relations and then grouping
does the join at full cardinality. Grouping each side to its join key first
(when the aggregate permits) shrinks both inputs. Only valid when the
aggregation does not depend on columns from the other side, and when join
multiplicity would not have duplicated rows into the aggregate - the classic
fan-out trap that changes `sum()` silently.

**Split a wide aggregation with several `filter` clauses.** Sometimes faster
as separate scans joined, sometimes much slower. Purely empirical; only test
it if a single node dominates.

---

## E. Things that look like optimizations and are not

- **Reordering joins by hand.** The planner reorders anyway. You are usually
  just changing which plan it finds, unreproducibly.
- **Adding `limit` anywhere.** Changes the relation. Not an optimization.
- **Replacing exact aggregates with approximate ones.** Different relation.
- **Removing `schema.yml` tests to make `dbt build` faster.** You are deleting
  the evidence that the optimization was correct, and the transformation DAG
  is not faster at all.
- **Turning off `partial_parse`, or other dbt-side tweaks.** These move dbt's
  own overhead, not database time. Legitimate to report, but keep them
  separate from DAG improvements - if `wall_s` drops and `db_s` does not,
  say so explicitly rather than claiming a faster DAG.
