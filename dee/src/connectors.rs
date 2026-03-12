/// All pre-implemented connectors
pub mod duckdb;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConnectorError {
    #[error("couldn't create a connection to the DB - {0}")]
    Create(String),
    #[error("couldn't execute query against connector - {0}")]
    Execute(String),
}

pub trait Connector {
    type Profile;
    type Connection;

    fn new(profile: Self::Profile) -> Result<Self::Connection, ConnectorError>;
    fn execute(&mut self, query_text: String) -> Result<usize, ConnectorError>;
}
