//! The metadata schema, as an ordered list of migrations.
//!
//! Migrations are embedded rather than loaded from disk so a `dee` binary is
//! self-sufficient, and applied inside one transaction each so a failure
//! partway leaves the recorded version behind rather than a half-built schema.
//!
//! Conventions used throughout:
//!
//! * Ids are application-generated UUIDv7 strings. They are time-ordered, so
//!   `ORDER BY id` is chronological and there is no sequence to contend on.
//! * JSON is stored in `VARCHAR` columns, not DuckDB's `JSON` type, so
//!   `serde_json` round-trips byte-exactly and no extension load is required.
//! * `VARCHAR[]` is used where the benchmark's parquet schema already has a
//!   list column, so the two stay directly comparable.

pub struct Migration {
    pub version: i32,
    pub sql: &'static str,
}

pub const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        sql: include_str!("migrations/0001_init.sql"),
    },
    Migration {
        version: 2,
        sql: include_str!("migrations/0002_optimizations.sql"),
    },
    Migration {
        version: 3,
        sql: include_str!("migrations/0003_delivery.sql"),
    },
];

/// The version the code expects. Used by `/v1/info` and by tests asserting the
/// migration list and this constant have not drifted apart.
pub fn latest_version() -> i32 {
    MIGRATIONS.last().map(|m| m.version).unwrap_or(0)
}
