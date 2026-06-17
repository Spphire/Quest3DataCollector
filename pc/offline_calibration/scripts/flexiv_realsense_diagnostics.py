from __future__ import annotations

import argparse
import concurrent.futures
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_ELEMENTS_ROOT = Path("/ssd1/mzc/FlexivElementsStudio")
DEFAULT_PROJECT_ROOT = Path("/ssd1/shenyibo/Quest3DataCollector")
DEFAULT_ROBOT_SUBNET = "192.168.2"
DEFAULT_ROBOT_HOSTS = ["192.168.2.100", "192.168.2.101", "192.168.2.102", "192.168.2.103", "192.168.2.104"]
DEFAULT_PORTS = [
    22,
    80,
    443,
    8000,
    8080,
    9000,
    9100,
    50051,
    50052,
    6000,
    6001,
    6379,
    10000,
    10001,
    15001,
    17001,
    17005,
    17006,
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only diagnostics for Quest/Flexiv/RealSense setup.")
    parser.add_argument("--elements-root", type=Path, default=DEFAULT_ELEMENTS_ROOT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--host", action="append", dest="hosts", help="Host/IP to ping and port-probe. May repeat.")
    parser.add_argument("--ports", default=",".join(str(port) for port in DEFAULT_PORTS))
    parser.add_argument("--robot-sn", help="Optionally try a read-only Flexiv RDK Robot connection with this serial.")
    parser.add_argument(
        "--network-interface",
        action="append",
        dest="network_interfaces",
        help="Local IPv4 address for the Flexiv RDK network interface whitelist. May repeat.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    hosts = args.hosts or DEFAULT_ROBOT_HOSTS
    ports = [int(part) for part in str(args.ports).split(",") if part.strip()]
    result = {
        "python": sys.version,
        "flexivrdk": flexivrdk_info(),
        "realsense": realsense_info(),
        "network": network_info(),
        "elements": elements_info(args.elements_root),
        "probe": probe_hosts(hosts, ports),
        "robotConnection": robot_connection_info(args.robot_sn, args.network_interfaces),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_human(result)
    return 0


def flexivrdk_info() -> dict[str, Any]:
    try:
        import flexivrdk  # type: ignore[import-not-found]

        return {
            "ok": True,
            "version": getattr(flexivrdk, "__version__", ""),
            "file": getattr(flexivrdk, "__file__", ""),
            "hasRobot": hasattr(flexivrdk, "Robot"),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def robot_connection_info(robot_sn: str | None, network_interfaces: list[str] | None) -> dict[str, Any] | None:
    if not robot_sn:
        return None
    started = time.monotonic()
    interfaces = normalize_network_interfaces(network_interfaces)
    try:
        import flexivrdk  # type: ignore[import-not-found]

        robot = flexivrdk.Robot(robot_sn.strip(), interfaces) if interfaces else flexivrdk.Robot(robot_sn.strip())
        states = robot.states()
        return {
            "ok": True,
            "robotSn": robot_sn.strip(),
            "networkInterfaces": interfaces,
            "elapsedSeconds": time.monotonic() - started,
            "connected": bool(robot.connected()),
            "mode": str(robot.mode()),
            "q": [float(v) for v in getattr(states, "q", [])],
            "tcpPose": [float(v) for v in getattr(states, "tcp_pose", [])],
            "flangePose": [float(v) for v in getattr(states, "flange_pose", [])],
        }
    except Exception as exc:
        return {
            "ok": False,
            "robotSn": robot_sn.strip(),
            "networkInterfaces": interfaces,
            "elapsedSeconds": time.monotonic() - started,
            "error": str(exc),
        }


def realsense_info() -> dict[str, Any]:
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]

        context = rs.context()
        cameras = []
        for device in context.query_devices():
            cameras.append(
                {
                    "name": device_info(device, rs.camera_info.name),
                    "serial": device_info(device, rs.camera_info.serial_number),
                    "firmware": device_info(device, rs.camera_info.firmware_version),
                    "usb": device_info(device, rs.camera_info.usb_type_descriptor),
                    "productLine": device_info(device, rs.camera_info.product_line),
                    "productId": device_info(device, rs.camera_info.product_id),
                }
            )
        return {"ok": True, "cameras": cameras}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cameras": []}


def device_info(device: Any, key: Any) -> str:
    try:
        if device.supports(key):
            return str(device.get_info(key))
    except Exception:
        pass
    return "unknown"


def network_info() -> dict[str, Any]:
    return {
        "ipAddr": run_text(["ip", "-br", "addr"]),
        "routes": run_text(["ip", "route"]),
        "neighbors": run_text(["ip", "neigh"]),
    }


def elements_info(root: Path) -> dict[str, Any]:
    version_path = root / "system_version.info"
    payload: dict[str, Any] = {"root": str(root), "systemVersionPath": str(version_path), "systemVersion": None}
    if version_path.exists():
        try:
            payload["systemVersion"] = json.loads(version_path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload["systemVersionError"] = str(exc)
    return payload


def probe_hosts(hosts: list[str], ports: list[int]) -> list[dict[str, Any]]:
    ping_status = {host: ping_host(host) for host in hosts}
    items = [(host, port) for host in hosts for port in ports]
    open_ports: dict[str, list[int]] = {host: [] for host in hosts}
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        for host, port, ok in executor.map(probe_port, items):
            if ok:
                open_ports.setdefault(host, []).append(port)
    return [{"host": host, "ping": ping_status[host], "openPorts": open_ports.get(host, [])} for host in hosts]


def ping_host(host: str) -> bool:
    return subprocess.call(["ping", "-c", "1", "-W", "1", host], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def probe_port(item: tuple[str, int]) -> tuple[str, int, bool]:
    host, port = item
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.25)
    try:
        sock.connect((host, port))
        return host, port, True
    except Exception:
        return host, port, False
    finally:
        sock.close()


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return str(exc)


def print_human(result: dict[str, Any]) -> None:
    print("Flexiv/RealSense diagnostics")
    flexiv = result["flexivrdk"]
    print(f"  flexivrdk: ok={flexiv.get('ok')} version={flexiv.get('version')} file={flexiv.get('file')}")
    rs = result["realsense"]
    print(f"  realsense: ok={rs.get('ok')} cameras={len(rs.get('cameras') or [])}")
    for camera in rs.get("cameras") or []:
        print(f"    {camera.get('serial')} {camera.get('name')} fw={camera.get('firmware')} usb={camera.get('usb')}")
    system = (result.get("elements") or {}).get("systemVersion") or {}
    serials = system.get("serial_number") if isinstance(system, dict) else {}
    versions = system.get("software_version") if isinstance(system, dict) else {}
    if serials:
        print(f"  Elements serials: {json.dumps(serials, ensure_ascii=False)}")
    if versions:
        print(f"  Elements software: {json.dumps(versions, ensure_ascii=False)}")
    print("  network interfaces:")
    print(indent(str((result.get("network") or {}).get("ipAddr") or "").strip(), "    "))
    print("  routes:")
    print(indent(str((result.get("network") or {}).get("routes") or "").strip(), "    "))
    print("  probes:")
    for row in result.get("probe") or []:
        print(f"    {row['host']}: ping={row['ping']} open={row['openPorts']}")
    connection = result.get("robotConnection")
    if connection is not None:
        print("  RDK connection:")
        print(f"    ok={connection.get('ok')} sn={connection.get('robotSn')} iface={connection.get('networkInterfaces')}")
        if connection.get("ok"):
            print(f"    mode={connection.get('mode')} connected={connection.get('connected')}")
            print(f"    q={connection.get('q')}")
            print(f"    tcpPose={connection.get('tcpPose')}")
        else:
            print(f"    error={connection.get('error')}")


def indent(text: str, prefix: str) -> str:
    if not text:
        return prefix + "n/a"
    return "\n".join(prefix + line for line in text.splitlines())


def normalize_network_interfaces(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return []
    return [str(item).strip() for item in items if str(item).strip()]


if __name__ == "__main__":
    raise SystemExit(main())
