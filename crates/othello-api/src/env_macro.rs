#[macro_export]
macro_rules! env_or_dotenv {
    ($key:expr) => {{
        #[cfg(debug_assertions)]
        {
            dotenvy_macro::dotenv!($key)
        }

        #[cfg(not(debug_assertions))]
        {
            env!($key)
        }
    }};
}
