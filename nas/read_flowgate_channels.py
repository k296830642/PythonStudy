#!/usr/bin/env python3
"""Read and decrypt FlowGate upstream-account credentials over SSH.

The script performs a read-only query through the remote Docker PostgreSQL
container. It never writes query results to disk.

Dependencies:
    python -m pip install paramiko cryptography

Examples:
    python read_flowgate_channels.py
    python read_flowgate_channels.py --account-id 1
    python read_flowgate_channels.py --config another-config.json
"""

#使用私钥时，填写 key_filename，并把 password 保持为空。
# show_secrets: true 才会显示完整渠道密钥。read.json
# 含明文 SSH 密码和 Data Key，不要提交、同步或发送；此前暴露的 root 密码也应立即轮换。
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import paramiko
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError as exc:
    print(
        "Missing dependency. Install it with:\n"
        "  python -m pip install paramiko cryptography",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


V1_PREFIX = "enc:v1:"
V2_PREFIX = "enc:v2:"
PBKDF2_ITERATIONS = 100_000
SENSITIVE_CREDENTIAL_NAMES = {
    "api_key",
    "bearer_token",
    "token",
    "password",
    "secret",
    "access_key",
    "secret_key",
}


@dataclass(frozen=True)
class SSHConnection:
    host: str
    port: int
    username: str
    password: str | None
    key_filename: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read and decrypt FlowGate upstream accounts over SSH."
    )
    parser.add_argument("--account-id", type=int, help="Only read this account ID")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("read.json"),
        help="JSON configuration path (default: read.json beside this script)",
    )
    parser.add_argument(
        "--show-secrets",
        action="store_true",
        help="Print complete API keys and other sensitive credential values",
    )
    parser.add_argument(
        "--accept-new-host-key",
        action="store_true",
        help="Trust and save an unknown SSH host key for this connection",
    )
    return parser.parse_args()


def load_config(path: Path) -> tuple[SSHConnection, dict[str, str], str, int | None, bool, bool]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc

    ssh = raw.get("ssh") or {}
    postgres = raw.get("postgres") or {}
    security = raw.get("security") or {}
    host = str(ssh.get("host") or "").strip()
    if not host:
        raise ValueError("ssh.host is required")
    try:
        port = int(ssh.get("port") or 22)
    except (TypeError, ValueError) as exc:
        raise ValueError("ssh.port must be a number") from exc

    password = str(ssh.get("password") or "") or None
    key_filename = str(ssh.get("key_filename") or "") or None
    if bool(password) == bool(key_filename):
        raise ValueError("configure exactly one of ssh.password or ssh.key_filename")

    data_key = str(security.get("data_key") or "")
    if not data_key:
        raise ValueError("security.data_key is required")
    account_id = raw.get("account_id")
    if account_id is not None:
        try:
            account_id = int(account_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("account_id must be a number or null") from exc

    return (
        SSHConnection(host, port, str(ssh.get("username") or "root"), password, key_filename),
        {
            "container": str(postgres.get("container") or "flowgate-postgres"),
            "user": str(postgres.get("user") or "flowgate"),
            "database": str(postgres.get("database") or "flowgate"),
        },
        data_key,
        account_id,
        bool(raw.get("show_secrets", False)),
        bool(raw.get("accept_new_host_key", False)),
    )


def decrypt_value(value: str, data_key: str) -> str:
    """Decrypt FlowGate enc:v1 and enc:v2 values; pass plain values through."""
    value = value.strip()
    if not value or not (value.startswith(V1_PREFIX) or value.startswith(V2_PREFIX)):
        return value

    if value.startswith(V2_PREFIX):
        payload = base64.b64decode(value.removeprefix(V2_PREFIX), validate=True)
        if len(payload) < 16 + 12 + 16:
            raise ValueError("enc:v2 payload is too short")
        salt, nonce, ciphertext = payload[:16], payload[16:28], payload[28:]
        key = hashlib.pbkdf2_hmac(
            "sha256", data_key.encode("utf-8"), salt, PBKDF2_ITERATIONS, dklen=32
        )
        return AESGCM(key).decrypt(nonce, ciphertext, None).decode("utf-8")

    payload = base64.b64decode(value.removeprefix(V1_PREFIX), validate=True)
    if len(payload) < 12 + 16:
        raise ValueError("enc:v1 payload is too short")
    key = hashlib.sha256(data_key.encode("utf-8")).digest()
    return AESGCM(key).decrypt(payload[:12], payload[12:], None).decode("utf-8")


def fetch_accounts(
    connection: SSHConnection,
    container: str,
    postgres_user: str,
    database: str,
    account_id: int | None,
    accept_new_host_key: bool,
) -> list[dict[str, Any]]:
    where_clause = ""
    if account_id is not None:
        where_clause = f"WHERE id = {account_id}"

    sql = f"""
SELECT COALESCE(json_agg(row_to_json(accounts)), '[]'::json)::text
FROM (
    SELECT
        id, name, provider, type, credentials, models, priority, weight,
        concurrency, current_concurrency, rate_multiplier, rate_limit, status,
        error_count, probe_failure_count, last_error_at, last_probe_status,
        last_probe_message, last_probe_latency_ms, last_probe_model_count,
        last_probe_at, created_at, updated_at
    FROM upstream_accounts
    {where_clause}
    ORDER BY priority ASC, weight DESC, id ASC
) AS accounts;
"""
    remote_command = " ".join(
        [
            "docker exec",
            shlex.quote(container),
            "psql",
            "-U",
            shlex.quote(postgres_user),
            "-d",
            shlex.quote(database),
            "-t",
            "-A",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            shlex.quote(sql),
        ]
    )

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    if accept_new_host_key:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    else:
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(
            hostname=connection.host,
            port=connection.port,
            username=connection.username,
            password=connection.password,
            key_filename=connection.key_filename,
            look_for_keys=connection.key_filename is None,
            allow_agent=True,
            timeout=20,
            banner_timeout=20,
            auth_timeout=20,
        )
        _, stdout, stderr = client.exec_command(remote_command, timeout=30)
        output = stdout.read().decode("utf-8").strip()
        error_output = stderr.read().decode("utf-8").strip()
        status = stdout.channel.recv_exit_status()
    finally:
        client.close()

    if status != 0:
        raise RuntimeError(error_output or f"remote query failed with exit status {status}")
    if not output:
        return []
    return json.loads(output)


def redact(name: str, value: str, show_secrets: bool) -> str:
    if show_secrets or name.lower() not in SENSITIVE_CREDENTIAL_NAMES:
        return value
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


def decrypt_credentials(
    encrypted_credentials: dict[str, Any], data_key: str, show_secrets: bool
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, raw_value in encrypted_credentials.items():
        value = "" if raw_value is None else str(raw_value)
        try:
            result[name] = redact(name, decrypt_value(value, data_key), show_secrets)
        except Exception as exc:
            result[name] = f"<decryption failed: {exc}>"
    return result


def main() -> None:
    args = parse_args()
    connection, postgres, data_key, config_account_id, config_show_secrets, config_accept_new_host_key = load_config(args.config)
    account_id = args.account_id if args.account_id is not None else config_account_id
    show_secrets = args.show_secrets or config_show_secrets
    accept_new_host_key = args.accept_new_host_key or config_accept_new_host_key
    print(f"Reading configuration from: {args.config}")
    print("The remote database query is read-only; query results are not written to disk.")

    accounts = fetch_accounts(
        connection,
        postgres["container"],
        postgres["user"],
        postgres["database"],
        account_id,
        accept_new_host_key,
    )
    if not accounts:
        print("No upstream accounts found.")
        return

    for account in accounts:
        encrypted_credentials = account.pop("credentials", {}) or {}
        account["credentials"] = decrypt_credentials(
            encrypted_credentials, data_key, show_secrets
        )

    print(json.dumps(accounts, ensure_ascii=False, indent=2, default=str))
    if not show_secrets:
        print("\nSensitive credential values are masked. Re-run with --show-secrets to reveal them.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, paramiko.SSHException, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
