//! Shadow admission policy: probabilistic sampling of eligible primary requests.
use sha2::{Digest, Sha256};
pub const DEFAULT_SAMPLE_PERCENT: f64 = 15.0;
pub fn sample_percent(value: Option<&str>) -> f64 {
    match value {
        None => DEFAULT_SAMPLE_PERCENT,
        Some(v) => v
            .parse::<f64>()
            .ok()
            .filter(|p| p.is_finite() && (0.0..=100.0).contains(p))
            .unwrap_or(0.0),
    }
}
pub fn sampled(request_id: &str, client_id: &str, utc_day: &str, percent: f64) -> bool {
    if !percent.is_finite() || percent <= 0.0 {
        return false;
    }
    if percent >= 100.0 {
        return true;
    }
    let mut hash = Sha256::new();
    for field in ["routerfuel-shadow-v1", utc_day, client_id, request_id] {
        hash.update((field.len() as u64).to_be_bytes());
        hash.update(field.as_bytes());
    }
    let digest = hash.finalize();
    let bucket =
        u64::from_be_bytes(digest[..8].try_into().unwrap()) as f64 / (u64::MAX as f64 + 1.0);
    bucket < percent / 100.0
}
pub fn cheaper(primary: f64, shadow: f64) -> bool {
    primary.is_finite() && shadow.is_finite() && primary > 0.0 && shadow >= 0.0 && shadow < primary
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shadow_config_rejects_invalid_rates() {
        assert_eq!(sample_percent(None), 15.0);
        for value in ["NaN", "inf", "-1", "101", "garbage"] {
            assert_eq!(sample_percent(Some(value)), 0.0);
        }
        assert_eq!(sample_percent(Some("15")), 15.0);
        assert_eq!(sample_percent(Some("0")), 0.0);
    }
    #[test]
    fn shadow_sampling_is_stable_and_approximately_fifteen_percent() {
        let count = (0..10000)
            .filter(|id| sampled(&id.to_string(), "client", "2026-10-08", 15.0))
            .count();
        assert!((1350..1650).contains(&count), "sampled {count}");
        for id in 0..100 {
            let id = id.to_string();
            assert!(!sampled(&id, "client", "day", 0.0));
            assert!(sampled(&id, "client", "day", 100.0));
            assert_eq!(
                sampled(&id, "client", "day", 15.0),
                sampled(&id, "client", "day", 15.0)
            );
        }
    }
    #[test]
    fn shadow_requires_strictly_lower_valid_estimated_cost() {
        assert!(cheaper(2.0, 1.0));
        for (primary, shadow) in [
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 0.0),
            (f64::NAN, 1.0),
            (1.0, -1.0),
            (1.0, f64::INFINITY),
        ] {
            assert!(!cheaper(primary, shadow));
        }
    }
}
