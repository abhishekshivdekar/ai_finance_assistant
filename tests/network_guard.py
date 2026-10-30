"""Blocks non-local network access in tests (used by the autouse fixture in conftest.py).

A test that really needs the network must say so with @pytest.mark.network; every other test that tries to open a
connection to a non-local host gets a NetworkBlockedError instead of a slow, billed or flaky call.
"""
from __future__ import annotations

import ipaddress
import socket
from typing import Any

LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo


class NetworkBlockedError(RuntimeError):
    """Raised when a test tries to reach a non-local host without the `network` marker."""


def is_local(host: Any) -> bool:
    """True for loopback names/addresses (and empty hosts). Example: '127.0.0.1' and 'localhost' yes, 'api.openai.com' no."""
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    if host in (None, ""):
        return True
    text = str(host).strip("[]").lower()
    if text in LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(text.split("%")[0]).is_loopback
    except ValueError:
        return False


def _host_of(address: Any) -> Any:
    """The host part of a socket address; AF_UNIX paths (str/bytes) are local by nature."""
    return address[0] if isinstance(address, tuple) else None


def _check(host: Any) -> None:
    if not is_local(host):
        raise NetworkBlockedError(f"network access to {host!r} is blocked in tests (use @pytest.mark.network to allow)")


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    _check(_host_of(address))
    return _real_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> Any:
    _check(_host_of(address))
    return _real_connect_ex(self, address)


def _guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    _check(host)
    return _real_getaddrinfo(host, *args, **kwargs)


def block(monkeypatch: Any) -> None:
    """Install the guards for the duration of one test (monkeypatch undoes them)."""
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", _guarded_getaddrinfo)
