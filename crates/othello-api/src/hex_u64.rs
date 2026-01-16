use serde::Serializer;

pub fn serialize<S>(value: &u64, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_str(&format!("0x{:x}", value))
}
