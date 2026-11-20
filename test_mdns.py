"""Tests for mdns.py. Run with: python3 -m unittest

zeroconf.Zeroconf is replaced in every test that would announce, so nothing
is sent to the network while the tests run.
"""

import contextlib
import io
import socket
import threading
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


class ZeroconfCase(unittest.TestCase):
    """zeroconf.Zeroconf replaced, and what is said on stderr kept."""

    def setUp(self) -> None:
        patcher = mock.patch("zeroconf.Zeroconf", autospec=True)
        self.zeroconf = patcher.start()
        self.addCleanup(patcher.stop)
        self.responder = self.zeroconf.return_value
        self.stderr = io.StringIO()
        redirect = contextlib.redirect_stderr(self.stderr)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)


class AnnouncerTest(ZeroconfCase):
    def test_registers_on_the_bound_address_and_says_so(self) -> None:
        mdns.Announcer("192.168.1.10", 8000).refresh()
        self.zeroconf.assert_called_once_with(
            interfaces=["192.168.1.10"], ip_version=zeroconf.IPVersion.V4Only
        )
        (info,), _ = self.responder.register_service.call_args
        self.assertEqual(info.port, 8000)
        self.assertEqual(
            self.stderr.getvalue(),
            "Also at http://noisemachine.local:8000 (192.168.1.10), where .local names work\n",
        )

    def test_does_nothing_more_while_the_addresses_stay_the_same(self) -> None:
        announcer = mdns.Announcer("192.168.1.10", 8000)
        announcer.refresh()
        announcer.refresh()
        self.zeroconf.assert_called_once()
        self.responder.close.assert_not_called()
        self.assertEqual(self.stderr.getvalue().count("\n"), 1)

    def test_announces_again_when_the_address_changes(self) -> None:
        announcer = mdns.Announcer("0.0.0.0", 8000)
        with mock.patch("mdns.addresses", side_effect=[["192.168.1.10"], ["192.168.1.20"]]):
            announcer.refresh()
            announcer.refresh()
        self.responder.close.assert_called_once_with()
        self.assertEqual(self.zeroconf.call_args.kwargs["interfaces"], ["192.168.1.20"])
        self.assertIn("(192.168.1.20)", self.stderr.getvalue().splitlines()[-1])

    def test_withdraws_the_name_when_the_host_has_no_address_left(self) -> None:
        announcer = mdns.Announcer("0.0.0.0", 8000)
        with mock.patch("mdns.addresses", side_effect=[["192.168.1.10"], []]):
            announcer.refresh()
            announcer.refresh()
        self.responder.close.assert_called_once_with()
        self.zeroconf.assert_called_once()

    def test_a_server_on_this_host_only_announces_nothing(self) -> None:
        mdns.Announcer("127.0.0.1", 8000).refresh()
        self.zeroconf.assert_not_called()
        self.assertEqual(self.stderr.getvalue(), "")

    def test_a_network_that_refuses_is_a_warning_and_is_tried_again(self) -> None:
        self.zeroconf.side_effect = [OSError("no multicast here"), self.responder]
        announcer = mdns.Announcer("192.168.1.10", 8000)
        announcer.refresh()
        self.assertEqual(
            self.stderr.getvalue(), "Not announced as noisemachine.local: no multicast here\n"
        )
        announcer.refresh()
        self.responder.register_service.assert_called_once()
        self.assertIn("Also at", self.stderr.getvalue().splitlines()[-1])

    def test_a_name_already_taken_is_a_warning_and_is_tried_again(self) -> None:
        self.responder.register_service.side_effect = [zeroconf.NonUniqueNameException(), None]
        announcer = mdns.Announcer("192.168.1.10", 8000)
        announcer.refresh()
        self.assertIn("Not announced as noisemachine.local", self.stderr.getvalue())
        self.responder.close.assert_called_once_with()
        announcer.refresh()
        self.assertIn("Also at", self.stderr.getvalue().splitlines()[-1])

    def test_the_same_warning_is_said_once(self) -> None:
        self.zeroconf.side_effect = OSError("no multicast here")
        announcer = mdns.Announcer("192.168.1.10", 8000)
        announcer.refresh()
        announcer.refresh()
        self.assertEqual(self.stderr.getvalue().count("\n"), 1)

    def test_close_withdraws_the_name(self) -> None:
        announcer = mdns.Announcer("192.168.1.10", 8000)
        announcer.refresh()
        announcer.close()
        self.responder.close.assert_called_once_with()


class AnnouncedTest(ZeroconfCase):
    def test_announces_before_the_block_and_withdraws_after_it(self) -> None:
        with mdns.announced("192.168.1.10", 8000):
            self.responder.register_service.assert_called_once()
            self.responder.close.assert_not_called()
        self.responder.close.assert_called_once_with()

    def test_checks_again_every_period(self) -> None:
        checked = threading.Semaphore(0)
        with mock.patch.object(
            mdns.Announcer, "refresh", autospec=True, side_effect=lambda _: checked.release()
        ):
            with mdns.announced("192.168.1.10", 8000, period=0.01):
                for _ in range(3):  # the first one, then two from the thread
                    self.assertTrue(checked.acquire(timeout=10))

    def test_stops_checking_when_the_block_ends(self) -> None:
        before = threading.active_count()
        with mdns.announced("192.168.1.10", 8000, period=0.01):
            self.assertEqual(threading.active_count(), before + 1)
        self.assertEqual(threading.active_count(), before)


if __name__ == "__main__":
    unittest.main()
