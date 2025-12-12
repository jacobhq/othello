use serde::de::Error;
use serde::{Deserialize, Deserializer, Serializer};

pub fn serialize<S>(value: &u64, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_str(&format!("0x{:x}", value))
}

pub fn deserialize<'de, D>(d: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(d)?;
    let s = s.trim_start_matches("0x");
    u64::from_str_radix(s, 16).map_err(D::Error::custom)
}
