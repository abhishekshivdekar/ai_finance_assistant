import socket

import pytest

import network_guard
from network_guard import NetworkBlockedError, is_local


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST", "::1", "[::1]", "", None, b"127.0.0.1"])
def test_loopback_hosts_are_local(host):
    assert is_local(host)


@pytest.mark.parametrize("host", ["api.openai.com", "huggingface.co", "8.8.8.8", "192.0.2.1", "localhost.evil.com"])
def test_other_hosts_are_not_local(host):
    assert not is_local(host)


def test_connecting_to_an_outside_host_is_blocked():
    with pytest.raises(NetworkBlockedError, match="api.openai.com"):
        socket.create_connection(("api.openai.com", 443), timeout=1)


def test_connecting_to_an_outside_ip_is_blocked_before_any_packet_is_sent():
    sock = socket.socket()
    try:
        with pytest.raises(NetworkBlockedError):
            sock.connect(("192.0.2.1", 80))
        with pytest.raises(NetworkBlockedError):
            sock.connect_ex(("192.0.2.1", 80))
    finally:
        sock.close()


def test_name_lookup_of_an_outside_host_is_blocked():
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("huggingface.co", 443)


def test_loopback_connections_still_work():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        client = socket.create_connection(server.getsockname(), timeout=2)
        client.close()
    finally:
        server.close()


@pytest.mark.network
def test_the_network_marker_leaves_the_real_functions_in_place():
    assert socket.socket.connect is network_guard._real_connect
    assert socket.socket.connect_ex is network_guard._real_connect_ex
    assert socket.getaddrinfo is network_guard._real_getaddrinfo


def test_an_unmarked_test_has_the_guards_installed():
    assert socket.socket.connect is network_guard._guarded_connect
