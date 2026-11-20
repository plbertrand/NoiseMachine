"""Announces the server on the local network as noisemachine.local (mDNS).

Phones and computers that resolve .local names can then reach the server
without knowing its IP address. Speakers are still given an IP address.
"""

import contextlib
import ipaddress
import socket
import sys
from collections.abc import Iterator

import ifaddr
import zeroconf

NAME = "noisemachine"
HOST = f"{NAME}.local"
SERVICE = "_http._tcp.local."


def addresses(bind: str) -> list[str]:
    """The IPv4 addresses the name stands for, for a server listening on bind.

    None for a server that listens on this host only: nobody else could use it.
    """
    listening = ipaddress.ip_address(bind)
    if not listening.is_unspecified:
        return [] if listening.is_loopback else [bind]
    found = [
        str(ip.ip)
        for adapter in ifaddr.get_adapters()
        for ip in adapter.ips
        if isinstance(ip.ip, str)  # IPv6 addresses are tuples
    ]
    return [address for address in found if not ipaddress.ip_address(address).is_loopback]


def service(found: list[str], port: int) -> zeroconf.ServiceInfo:
    """What is announced: a web server named noisemachine, at these addresses."""
    return zeroconf.ServiceInfo(
        SERVICE,
        f"{NAME}.{SERVICE}",
        addresses=[socket.inet_aton(address) for address in found],
        port=port,
        server=f"{HOST}.",
    )


@contextlib.contextmanager
def announced(bind: str, port: int) -> Iterator[str | None]:
    """Announce a server listening on bind and port, for as long as the block runs.

    Gives the name it answers to, or None when it is not announced: the server
    works without it, so a failure is a warning.
    """
    found = addresses(bind)
    if not found:
        yield None
        return
    try:
        responder = zeroconf.Zeroconf(interfaces=found, ip_version=zeroconf.IPVersion.V4Only)
    except OSError as error:
        print(f"Not announced as {HOST}: {error}", file=sys.stderr)
        yield None
        return
    with contextlib.closing(responder):
        try:
            responder.register_service(service(found, port))
        except (OSError, zeroconf.Error) as error:
            print(f"Not announced as {HOST}: {error!r}", file=sys.stderr)
            yield None
            return
        yield HOST
