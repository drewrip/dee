use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use crate::connectors::{Connector, ConnectorError};

/// A registry that maps string identifiers to connector factories.
///
/// This enables dynamic connector loading — new connector types can be
/// registered without modifying the CLI's exhaustive `match` blocks.
///
/// Usage:
/// ```ignore
/// let mut registry = ConnectionRegistry::new();
/// registry.register::<DuckDBConnection>("duckdb");
/// registry.register::<PostgresConnection>("postgres");
///
/// let conn = registry.create::<DuckDBConnection>("duckdb", config).await?;
/// ```
#[derive(Default)]
pub struct ConnectionRegistry {
    factories: HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
}

impl ConnectionRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a connector factory under the given identifier.
    pub fn register<C: Connector<Connection = C> + Send + Sync + 'static>(&mut self, name: impl Into<String>)
    where
        C::Config: Send,
    {
        let name = name.into();
        self.factories.insert(name, Box::new(Self::factory_fn::<C>));
    }

    /// Create a connector by identifier and config.
    pub async fn create<C: Connector<Connection = C> + Send + Sync + 'static>(
        &self,
        name: &str,
        config: C::Config,
    ) -> Result<Arc<C>, ConnectorError> {
        let factory = self.factories.get(name).ok_or_else(|| {
            ConnectorError::Create(format!("unknown connector type '{}'", name))
        })?;

        let fn_ptr = factory
            .downcast_ref::<fn(C::Config) -> Pin<Box<dyn Future<Output = Result<Arc<C>, ConnectorError>> + Send + 'static>>>()
            .ok_or_else(|| {
                ConnectorError::Create(format!(
                    "connector '{}' is not a {} connector",
                    name,
                    std::any::type_name::<C>()
                ))
            })?;

        fn_ptr(config).await
    }

    /// Return the list of registered connector names.
    pub fn names(&self) -> Vec<&str> {
        self.factories.keys().map(|s| s.as_str()).collect()
    }

    fn factory_fn<C: Connector<Connection = C> + Send + Sync + 'static>(
        config: C::Config,
    ) -> Pin<Box<dyn Future<Output = Result<Arc<C>, ConnectorError>> + Send + 'static>>
    where
        C::Config: Send,
    {
        Box::pin(async move { C::new(config).await })
    }
}
