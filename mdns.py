"""Announces the server on the local network as noisemachine.local (mDNS).

Phones and computers that resolve .local names can then reach the server
without knowing its IP address. Speakers are still given an IP address.

The announcement is checked every PERIOD seconds, so it follows a host whose
address changes and is tried again after a failure.
"""

import contextlib
import ipaddress
import socket
import sys
import threading
from collections.abc import Iterator

import ifaddr
import zeroconf

NAME = "noisemachine"
HOST = f"{NAME}.local"
SERVICE = "_http._tcp.local."
PERIOD = 60.0  # seconds between two checks of the announcement


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


class Announcer:
    """Keeps a server listening on bind and port announced: call refresh() now and then.

    It says on stderr what it announces and what went wrong, once per change.
    The server works without the name, so a failure is only a warning.
    """

    def __init__(self, bind: str, port: int) -> None:
        self.bind = bind
        self.port = port
        self.found: list[str] = []
        self.responder: zeroconf.Zeroconf | None = None
        self.said = ""

    def refresh(self) -> None:
        """Announce if not announced yet, or again if this host's addresses changed."""
        found = addresses(self.bind)
        if self.responder is not None and found == self.found:
            return
        self.close()

        if not found:
            return

        try:
            responder = zeroconf.Zeroconf(interfaces=found, ip_version=zeroconf.IPVersion.V4Only)
        except OSError as error:
            self.say(f"Not announced as {HOST}: {error}")
            return

        try:
            responder.register_service(service(found, self.port))
        except (OSError, zeroconf.Error) as error:
            responder.close()
            self.say(f"Not announced as {HOST}: {error!r}")
            return

        self.responder, self.found = responder, found
        at = ", ".join(found)
        self.say(f"Also at http://{HOST}:{self.port} ({at}), where .local names work")

    def close(self) -> None:
        """Withdraw the name."""
        if self.responder is not None:
            self.responder.close()
            self.responder = None

    def say(self, message: str) -> None:
        if message != self.said:
            print(message, file=sys.stderr, flush=True)
            self.said = message


@contextlib.contextmanager
def announced(bind: str, port: int, period: float = PERIOD) -> Iterator[None]:
    """Keep a server listening on bind and port announced while the block runs.

    Announces once before the block starts, then checks every period seconds
    from a thread of its own.
    """
    announcer = Announcer(bind, port)
    stop = threading.Event()

    def keep() -> None:
        while not stop.wait(period):
            announcer.refresh()

    announcer.refresh()
    thread = threading.Thread(target=keep, name="mdns", daemon=True)
    thread.start()

    try:
        yield
    finally:
        stop.set()
        thread.join()
        announcer.close()
