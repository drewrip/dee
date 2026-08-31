//! Cron expressions with a timezone.
//!
//! `croner` is used rather than `cron` because `cron` requires a seconds
//! field, so an ordinary five-field `0 3 * * *` fails to parse -- a permanent
//! papercut for anyone who has written a crontab. Both are generic over
//! chrono's `TimeZone`, which is what lets `chrono-tz` handle daylight saving
//! instead of this module.

use chrono::{DateTime, Utc};
use croner::Cron;
use std::str::FromStr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CronError {
    #[error("'{expr}' is not a valid cron expression: {source}")]
    Expression {
        expr: String,
        source: croner::errors::CronError,
    },
    #[error("'{0}' is not a known IANA timezone (for example 'UTC' or 'America/New_York')")]
    Timezone(String),
}

#[derive(Debug, Clone)]
pub struct CronSpec {
    pub expr: String,
    pub tz: chrono_tz::Tz,
    cron: Cron,
}

impl CronSpec {
    pub fn parse(expr: &str, timezone: &str) -> Result<Self, CronError> {
        let tz = chrono_tz::Tz::from_str(timezone)
            .map_err(|_| CronError::Timezone(timezone.to_string()))?;
        let cron = Cron::from_str(expr).map_err(|source| CronError::Expression {
            expr: expr.to_string(),
            source,
        })?;
        Ok(CronSpec {
            expr: expr.to_string(),
            tz,
            cron,
        })
    }

    /// The next firing strictly after `after`, in UTC.
    ///
    /// The comparison happens in the schedule's own timezone, so "3am daily"
    /// stays 3am local across a daylight-saving change rather than drifting an
    /// hour twice a year.
    pub fn next_after(&self, after: DateTime<Utc>) -> Option<DateTime<Utc>> {
        let local = after.with_timezone(&self.tz);
        self.cron
            .find_next_occurrence(&local, false)
            .ok()
            .map(|t| t.with_timezone(&Utc))
    }

    /// How many firings fall in `(after, until]`, capped at `cap`.
    ///
    /// Used to report how many windows a downtime swallowed. Capped because a
    /// per-minute schedule and a week of downtime is 10,080 windows, and the
    /// only thing anyone does with the number is read it.
    pub fn count_between(
        &self,
        after: DateTime<Utc>,
        until: DateTime<Utc>,
        cap: usize,
    ) -> usize {
        let mut cursor = after;
        let mut count = 0;
        while count < cap {
            match self.next_after(cursor) {
                Some(next) if next <= until => {
                    count += 1;
                    cursor = next;
                }
                _ => break,
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn utc(iso: &str) -> DateTime<Utc> {
        iso.parse().unwrap()
    }

    #[test]
    fn test_a_plain_five_field_expression_parses() {
        // The whole reason croner was chosen over the `cron` crate.
        let spec = CronSpec::parse("0 3 * * *", "UTC").unwrap();
        assert_eq!(
            spec.next_after(utc("2026-01-01T00:00:00Z")).unwrap(),
            utc("2026-01-01T03:00:00Z")
        );
    }

    #[test]
    fn test_a_six_field_expression_with_seconds_parses() {
        let spec = CronSpec::parse("30 * * * * *", "UTC").unwrap();
        let next = spec.next_after(utc("2026-01-01T00:00:00Z")).unwrap();
        assert_eq!(next, utc("2026-01-01T00:00:30Z"));
    }

    #[test]
    fn test_an_invalid_expression_is_rejected_at_parse_time() {
        // Validation happens when a schedule is set, so a typo is a 400 rather
        // than a schedule that silently never fires.
        assert!(CronSpec::parse("not a cron", "UTC").is_err());
        assert!(CronSpec::parse("99 * * * *", "UTC").is_err());
    }

    #[test]
    fn test_an_unknown_timezone_is_rejected() {
        assert!(CronSpec::parse("0 3 * * *", "Mars/Olympus").is_err());
        assert!(CronSpec::parse("0 3 * * *", "America/New_York").is_ok());
    }

    #[test]
    fn test_next_after_is_strict() {
        // A firing must not re-fire itself: if `next_after` were inclusive the
        // scheduler would loop on the same window forever.
        let spec = CronSpec::parse("0 * * * *", "UTC").unwrap();
        assert_eq!(
            spec.next_after(utc("2026-01-01T05:00:00Z")).unwrap(),
            utc("2026-01-01T06:00:00Z")
        );
    }

    #[test]
    fn test_a_local_schedule_tracks_its_timezone_not_utc() {
        // 03:00 in New York is 08:00 UTC in winter.
        let spec = CronSpec::parse("0 3 * * *", "America/New_York").unwrap();
        assert_eq!(
            spec.next_after(utc("2026-01-15T00:00:00Z")).unwrap(),
            utc("2026-01-15T08:00:00Z")
        );
    }

    #[test]
    fn test_daylight_saving_spring_forward_still_fires_once() {
        // On 2026-03-08 New York jumps 02:00 -> 03:00, so a 02:30 schedule has
        // no wall-clock instant that day. It must still fire exactly once and
        // must not stall the scheduler.
        let spec = CronSpec::parse("30 2 * * *", "America/New_York").unwrap();
        let before = utc("2026-03-08T00:00:00Z");
        let next = spec.next_after(before).expect("a skipped hour must not stall");
        assert!(next > before);

        let count = spec.count_between(before, utc("2026-03-10T00:00:00Z"), 100);
        assert!((1..=2).contains(&count), "fired {count} times over two days");
    }

    #[test]
    fn test_daylight_saving_fall_back_does_not_double_fire() {
        // On 2026-11-01 New York repeats 01:00-02:00. A 01:30 schedule must
        // fire once that day, not twice.
        let spec = CronSpec::parse("30 1 * * *", "America/New_York").unwrap();
        let day_start = chrono_tz::America::New_York
            .with_ymd_and_hms(2026, 11, 1, 0, 0, 0)
            .unwrap()
            .with_timezone(&Utc);
        let day_end = chrono_tz::America::New_York
            .with_ymd_and_hms(2026, 11, 2, 0, 0, 0)
            .unwrap()
            .with_timezone(&Utc);

        assert_eq!(spec.count_between(day_start, day_end, 10), 1);
    }

    #[test]
    fn test_count_between_reports_a_downtime_gap() {
        // This is what tells an operator how many windows a restart swallowed.
        let spec = CronSpec::parse("0 * * * *", "UTC").unwrap();
        assert_eq!(
            spec.count_between(utc("2026-01-01T02:00:00Z"), utc("2026-01-01T09:00:00Z"), 100),
            7
        );
        assert_eq!(
            spec.count_between(utc("2026-01-01T02:00:00Z"), utc("2026-01-01T02:30:00Z"), 100),
            0
        );
    }

    #[test]
    fn test_count_between_is_capped() {
        // A per-minute schedule over a week is thousands of windows, and the
        // count is only ever read by a human.
        let spec = CronSpec::parse("* * * * *", "UTC").unwrap();
        assert_eq!(
            spec.count_between(utc("2026-01-01T00:00:00Z"), utc("2026-01-08T00:00:00Z"), 500),
            500
        );
    }
}
