"""Managing a `dee serve` process for the duration of a sweep.

The harness used to shell out to `dee-cli` once per phase, so every cell paid
for process startup and connection-pool construction. dee is a server now, so
the sweep starts one and talks to it over HTTP: pools stay warm across cells,
and the run history the server records is itself an artifact of the sweep.

Two consequences worth knowing about:

* The server binds to port 0 and announces the port it got on stdout, so
  concurrent sweeps on one machine never collide.
* Resource sampling attaches to the server process rather than a short-lived
  child. CPU and IO are counter deltas from an attach-time baseline, so they
  stay correct; RSS does not, because it is absolute and carries over between
  cells. See `schema.py` for how each column is qualified.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

STARTUP_LINE = "dee-server listening on "


class ServerError(RuntimeError):
    """The server could not be started, reached, or did what was asked."""


class ApiError(ServerError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.status = status
        self.code = code
        self.message = message


class DeeClient:
    """A thin JSON client for one dee server."""

    def __init__(self, url: str, timeout: int = 3600):
        self.url = url.rstrip("/")
        self.timeout = timeout

    # -- transport ---------------------------------------------------------

    def _request(self, method: str, path: str, body: Any = None,
                 timeout: int | None = None) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            f"{self.url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            # The server's error envelope is {"error": {"code", "message"}}.
            try:
                payload = json.loads(raw)["error"]
                raise ApiError(e.code, payload["code"], payload["message"]) from None
            except (ValueError, KeyError, TypeError):
                raise ServerError(f"HTTP {e.code}: {raw.decode(errors='replace')[:400]}") from None
        except urllib.error.URLError as e:
            raise ServerError(f"cannot reach the dee server at {self.url}: {e.reason}") from None

        if not raw.strip():
            return None
        return json.loads(raw)

    def get(self, path: str, **params: Any) -> Any:
        query = {k: v for k, v in params.items() if v is not None}
        if query:
            path = f"{path}?{urllib.parse.urlencode(query)}"
        return self._request("GET", path)

    def get_text(self, path: str) -> str:
        request = urllib.request.Request(f"{self.url}{path}", method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read().decode()

    def post(self, path: str, body: Any = None, timeout: int | None = None) -> Any:
        return self._request("POST", path, body if body is not None else {}, timeout)

    def put(self, path: str, body: Any) -> Any:
        return self._request("PUT", path, body)

    # -- operations --------------------------------------------------------

    def info(self) -> dict[str, Any]:
        return self.get("/v1/info")

    def optimizer_options(self) -> list[dict[str, Any]]:
        return self.get("/v1/optimizer/options")

    def register_connection(self, name: str, config: dict[str, Any]) -> dict[str, Any]:
        # Always an upsert: each prepared project points at its own freshly
        # built warehouse, and replacing the config is what evicts the pool
        # still holding the previous cell's database file open.
        return self.post("/v1/connections?upsert=true", {"name": name, "config": config})

    def submit_dag(self, name: str, definition: dict[str, Any],
                   target: str) -> dict[str, Any]:
        return self.post(
            "/v1/dags",
            {"name": name, "definition": definition, "target": target},
        )

    def dag_version(self, name: str, version: int | None = None) -> dict[str, Any]:
        if version is None:
            version = self.get(f"/v1/dags/{name}")["current_version"]
        return self.get(f"/v1/dags/{name}/versions/{version}")

    def trigger(self, name: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
        return self.post(
            f"/v1/dags/{name}/runs?wait=true&timeout_s={timeout}", body, timeout + 60
        )

    def run_group(self, group_id: str) -> dict[str, Any]:
        return self.get(f"/v1/run-groups/{group_id}")

    def group_report(self, group_id: str) -> dict[str, Any]:
        return self.get(f"/v1/run-groups/{group_id}/report")

    def run_nodes(self, run_id: str) -> list[dict[str, Any]]:
        return self.get(f"/v1/runs/{run_id}/nodes")

    def run_plans(self, run_id: str) -> list[dict[str, Any]]:
        return self.get(f"/v1/runs/{run_id}/plans")

    def optimize(self, name: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
        return self.post(
            f"/v1/dags/{name}/optimize?wait=true&timeout_s={timeout}", body, timeout + 60
        )

    def optimization(self, optimization_id: str) -> dict[str, Any]:
        return self.get(f"/v1/optimizations/{optimization_id}")

    def optimization_report(self, optimization_id: str) -> dict[str, Any]:
        return self.get(f"/v1/optimizations/{optimization_id}/report")

    def optimization_explain(self, optimization_id: str) -> str:
        return self.get_text(f"/v1/optimizations/{optimization_id}/explain.html")


class DeeServer:
    """Start a `dee serve` for a sweep, or attach to one already running."""

    def __init__(self, dee_bin: Path, run_dir: Path, bind: str = "127.0.0.1:0",
                 url: str | None = None, startup_timeout_s: int = 60,
                 timeout_s: int = 3600):
        self.dee_bin = Path(dee_bin)
        self.run_dir = Path(run_dir)
        self.bind = bind
        self.external_url = url
        self.startup_timeout_s = startup_timeout_s
        self.timeout_s = timeout_s
        self.process: subprocess.Popen | None = None
        self.url: str | None = url
        self.log_path = self.run_dir / "server.log"

    @property
    def pid(self) -> int | None:
        """The process resource sampling should attach to."""
        return self.process.pid if self.process else None

    def __enter__(self) -> DeeClient:
        if self.external_url:
            client = DeeClient(self.external_url, self.timeout_s)
            client.info()  # fail now if it is not actually there
            return client

        self.run_dir.mkdir(parents=True, exist_ok=True)
        metadata_db = self.run_dir / "metadata.duckdb"
        log = self.log_path.open("w")
        self.process = subprocess.Popen(
            [
                str(self.dee_bin), "serve",
                "--bind", self.bind,
                "--metadata-db", str(metadata_db),
            ],
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            # Its own process group, so a Ctrl-C in the harness does not race
            # the orderly shutdown in __exit__.
            start_new_session=True,
        )

        self.url = self._await_startup()
        return DeeClient(self.url, self.timeout_s)

    def _await_startup(self) -> str:
        assert self.process is not None and self.process.stdout is not None
        deadline = time.monotonic() + self.startup_timeout_s

        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise ServerError(
                    f"dee serve exited with code {self.process.returncode} before it was "
                    f"ready; see {self.log_path}"
                )
            line = self.process.stdout.readline()
            if not line:
                continue
            if STARTUP_LINE in line:
                url = line.split(STARTUP_LINE, 1)[1].strip()
                # Drain the rest of stdout in the background so a full pipe
                # buffer can never block the server mid-sweep.
                self._drain_stdout()
                return url

        self._terminate()
        raise ServerError(
            f"dee serve did not report a listening address within "
            f"{self.startup_timeout_s}s; see {self.log_path}"
        )

    def _drain_stdout(self) -> None:
        import threading

        def drain(stream, path: Path):
            with path.open("a") as sink:
                for line in stream:
                    sink.write(line)

        assert self.process is not None
        threading.Thread(
            target=drain, args=(self.process.stdout, self.log_path), daemon=True
        ).start()

    def __exit__(self, *exc) -> None:
        self._terminate()

    def _terminate(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        # SIGTERM lets the server finalize in-flight runs and record a clean
        # exit; without it every run is orphaned on the next start.
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self.process.kill()
            self.process.wait(timeout=10)
