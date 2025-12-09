use std::time::Duration;
use async_posthog::ClientOptions;
use async_posthog::Client;
use crate::env_or_dotenv;

const POSTHOG_API_KEY: &str = env_or_dotenv!("POSTHOG_API_KEY");

struct Analytics {
    client: Client
}

struct InternalClientOptions {
    api_endpoint: String,
    api_key: String,
    timeout: Duration,
}

impl Into<ClientOptions> for &str {
    fn into(self) -> ClientOptions {
        todo!()
    }
}

impl Analytics {
    fn new() -> Self {
        Analytics {
            client: async_posthog::client(POSTHOG_API_KEY)
        }
    }
}