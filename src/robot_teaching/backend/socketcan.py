"""Bring the SocketCAN interface up before the bus is opened.

``ip link set <interface> up type can bitrate <bitrate>`` needs CAP_NET_ADMIN, so unless
the server already runs as root the command goes through ``sudo -n`` (no password
prompt: a sudoers rule makes it silent, see the README). An interface that is already up
is left alone, since setting the bitrate of an up interface fails with "Device or
resource busy". The interface state comes from sysfs, which needs no privileges.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .base import BackendError

SYSFS_NET = Path("/sys/class/net")
IFF_UP = 0x1
SUDOERS_FILE = "/etc/sudoers.d/robot-teaching-can"


def is_socketcan_channel(channel: str) -> bool:
    """The vendor YAML's ``channel`` names either a SocketCAN interface or a serial device."""
    return not str(channel).startswith("/dev/")


def interface_is_up(interface: str, sysfs: Path = SYSFS_NET) -> bool | None:
    """``True``/``False`` from the interface's IFF_UP flag; ``None`` if there is no such interface."""
    flags = sysfs / interface / "flags"
    if not flags.exists():
        return None
    return bool(int(flags.read_text().strip(), 16) & IFF_UP)


def bring_up_command(interface: str, bitrate: int, as_root: bool) -> list[str]:
    cmd = ["ip", "link", "set", interface, "up", "type", "can", "bitrate", str(int(bitrate))]
    return cmd if as_root else ["sudo", "-n", *cmd]


def sudoers_hint(interface: str) -> str:
    ip = shutil.which("ip") or "/usr/sbin/ip"
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - no passwd entry
        user = "<user>"
    return (f"Bring it up by hand (sudo ip link set {interface} up type can bitrate <bitrate>) or let the "
            f"server do it without a password: echo '{user} ALL=(root) NOPASSWD: {ip} link set {interface} *' "
            f"| sudo tee {SUDOERS_FILE}")


def ensure_can_up(
    interface: str,
    bitrate: int,
    *,
    sysfs: Path = SYSFS_NET,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    as_root: bool | None = None,
) -> str:
    """Bring ``interface`` up at ``bitrate`` bit/s if it is down; return what was done.

    Raises :class:`BackendError` with the command and the system's reason when it cannot,
    plus the sudoers hint when the problem is privileges.
    """
    state = interface_is_up(interface, sysfs)
    if state is None:
        raise BackendError(f"{interface}: no such network interface; is the CAN adapter plugged in and its driver loaded?")
    if state:
        return f"{interface} is up"
    if as_root is None:
        as_root = hasattr(os, "geteuid") and os.geteuid() == 0
    cmd = bring_up_command(interface, bitrate, as_root)
    shown = " ".join(cmd)
    try:
        res = run(cmd, capture_output=True, text=True, timeout=10.0)
    except FileNotFoundError as e:
        raise BackendError(f"could not bring {interface} up: {e} (is iproute2 installed?)") from e
    except subprocess.TimeoutExpired as e:
        raise BackendError(f"could not bring {interface} up: '{shown}' did not return in 10 s") from e
    if res.returncode != 0:
        reason = (res.stderr or res.stdout or "").strip() or f"exit status {res.returncode}"
        hint = "" if as_root else " " + sudoers_hint(interface)
        raise BackendError(f"could not bring {interface} up ('{shown}': {reason}).{hint}")
    if interface_is_up(interface, sysfs) is False:
        raise BackendError(f"{interface} is still down after '{shown}'")
    return f"{interface} brought up at {int(bitrate)} bit/s"
