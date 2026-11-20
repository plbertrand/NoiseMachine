"""Tests for mdns.py. Run with: python3 -m unittest

zeroconf.Zeroconf is replaced in every test that would announce, so nothing
is sent to the network while the tests run.
"""

import contextlib
import io
import socket
import unittest
from unittest import mock

import ifaddr
import zeroconf

import mdns


def adapters() -> list[ifaddr.Adapter]:
    """A host with a loopback, a LAN address, a second network, and IPv6."""
    return [
        ifaddr.Adapter("lo", "lo", [ifaddr.IP("127.0.0.1", 8, "lo")]),
        ifaddr.Adapter(
            "eth0",
            "eth0",
            [ifaddr.IP("192.168.1.10", 24, "eth0"), ifaddr.IP(("fe80::1", 0, 2), 64, "eth0")],
        ),
        ifaddr.Adapter("eth1", "eth1", [ifaddr.IP("10.0.0.5", 8, "eth1")]),
    ]


class AddressesTest(unittest.TestCase):
    def test_a_server_on_one_address_is_announced_there(self) -> None:
        self.assertEqual(mdns.addresses("192.168.1.10"), ["192.168.1.10"])

    def test_a_server_on_this_host_only_is_not_announced(self) -> None:
        self.assertEqual(mdns.addresses("127.0.0.1"), [])

    def test_a_server_on_every_address_is_announced_on_each_network(self) -> None:
        with mock.patch("ifaddr.get_adapters", return_value=adapters()):
            self.assertEqual(mdns.addresses("0.0.0.0"), ["192.168.1.10", "10.0.0.5"])


class ServiceTest(unittest.TestCase):
    def test_is_a_web_server_named_noisemachine(self) -> None:
        info = mdns.service(["192.168.1.10"], 8000)
        self.assertEqual(info.type, "_http._tcp.local.")
        self.assertEqual(info.name, "noisemachine._http._tcp.local.")
        self.assertEqual(info.server, "noisemachine.local.")
        self.assertEqual(info.port, 8000)
        self.assertEqual(info.addresses, [socket.inet_aton("192.168.1.10")])


class AnnouncedTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch("zeroconf.Zeroconf", autospec=True)
        self.zeroconf = patcher.start()
        self.addCleanup(patcher.stop)
        self.responder = self.zeroconf.return_value
        self.stderr = io.StringIO()
        redirect = contextlib.redirect_stderr(self.stderr)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def test_gives_the_name_and_registers_on_the_bound_address(self) -> None:
        with mdns.announced("192.168.1.10", 8000) as name:
            self.assertEqual(name, "noisemachine.local")
            self.zeroconf.assert_called_once_with(
                interfaces=["192.168.1.10"], ip_version=zeroconf.IPVersion.V4Only
            )
            (info,), _ = self.responder.register_service.call_args
            self.assertEqual(info.port, 8000)
            self.responder.close.assert_not_called()

    def test_stops_announcing_when_the_block_ends(self) -> None:
        with mdns.announced("192.168.1.10", 8000):
            pass
        self.responder.close.assert_called_once_with()

    def test_a_server_on_this_host_only_announces_nothing(self) -> None:
        with mdns.announced("127.0.0.1", 8000) as name:
            self.assertIsNone(name)
        self.zeroconf.assert_not_called()

    def test_a_network_that_refuses_is_a_warning(self) -> None:
        self.zeroconf.side_effect = OSError("no multicast here")
        with mdns.announced("192.168.1.10", 8000) as name:
            self.assertIsNone(name)
        self.assertEqual(
            self.stderr.getvalue(), "Not announced as noisemachine.local: no multicast here\n"
        )

    def test_a_name_already_taken_is_a_warning(self) -> None:
        self.responder.register_service.side_effect = zeroconf.NonUniqueNameException()
        with mdns.announced("192.168.1.10", 8000) as name:
            self.assertIsNone(name)
        self.assertIn("Not announced as noisemachine.local", self.stderr.getvalue())
        self.responder.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
