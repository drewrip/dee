//! A clock the scheduler reads time from.
//!
//! The scheduler takes `now` as an argument and gets it from here rather than
//! calling `Utc::now()` internally, so its tests can drive a week of schedule
//! history without a single `sleep`.

use chrono::{DateTime, Utc};

pub trait Clock: Send + Sync {
    fn now(&self) -> DateTime<Utc>;
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

#[cfg(test)]
pub struct MockClock {
    now: std::sync::Mutex<DateTime<Utc>>,
}

#[cfg(test)]
impl MockClock {
    pub fn at(iso: &str) -> Self {
        MockClock {
            now: std::sync::Mutex::new(
                iso.parse::<DateTime<Utc>>().expect("a valid RFC 3339 timestamp"),
            ),
        }
    }

    pub fn advance(&self, delta: chrono::Duration) {
        let mut now = self.now.lock().unwrap();
        *now += delta;
    }

    pub fn set(&self, iso: &str) {
        *self.now.lock().unwrap() = iso.parse().expect("a valid RFC 3339 timestamp");
    }
}

#[cfg(test)]
impl Clock for MockClock {
    fn now(&self) -> DateTime<Utc> {
        *self.now.lock().unwrap()
    }
}
