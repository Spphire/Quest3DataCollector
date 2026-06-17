from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
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
RDK_ROBOT_SOFTWARE_COMPATIBILITY = {
    (1, 7): (3, 9),
}
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
    parser.add_argument("--_robot-connection-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    if args._robot_connection_worker:
        print(json.dumps(robot_connection_info_direct(args.robot_sn, args.network_interfaces), ensure_ascii=False))
        return 0

    hosts = args.hosts or DEFAULT_ROBOT_HOSTS
    ports = [int(part) for part in str(args.ports).split(",") if part.strip()]
    flexivrdk = flexivrdk_info()
    elements = elements_info(args.elements_root)
    result = {
        "python": sys.version,
        "flexivrdk": flexivrdk,
        "realsense": realsense_info(),
        "network": network_info(),
        "elements": elements,
        "compatibility": compatibility_info(flexivrdk, elements),
        "probe": probe_hosts(hosts, ports),
        "robotConnection": robot_connection_info(args.robot_sn, args.network_interfaces),
    }
    result["interpretation"] = interpret_result(result)
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
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_robot-connection-worker",
        "--robot-sn",
        str(robot_sn).strip(),
    ]
    for interface in normalize_network_interfaces(network_interfaces):
        command.extend(["--network-interface", interface])
    started = time.monotonic()
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=8.0, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "robotSn": str(robot_sn).strip(),
            "networkInterfaces": normalize_network_interfaces(network_interfaces),
            "elapsedSeconds": time.monotonic() - started,
            "error": f"RDK connection worker timed out after {exc.timeout} seconds",
        }
    stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    for line in reversed(stdout_lines):
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                parsed.setdefault("workerReturnCode", proc.returncode)
                if proc.stderr.strip():
                    parsed["workerStderr"] = proc.stderr.strip()[-2000:]
                return parsed
        except json.JSONDecodeError:
            continue
    return {
        "ok": False,
        "robotSn": str(robot_sn).strip(),
        "networkInterfaces": normalize_network_interfaces(network_interfaces),
        "elapsedSeconds": time.monotonic() - started,
        "workerReturnCode": proc.returncode,
        "error": "RDK connection worker did not return JSON",
        "workerStdout": proc.stdout.strip()[-2000:],
        "workerStderr": proc.stderr.strip()[-2000:],
    }


def robot_connection_info_direct(robot_sn: str | None, network_interfaces: list[str] | None) -> dict[str, Any] | None:
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


def compatibility_info(flexiv: dict[str, Any], elements: dict[str, Any]) -> dict[str, Any]:
    rdk_version = str(flexiv.get("version") or "")
    system = elements.get("systemVersion") or {}
    versions = system.get("software_version") if isinstance(system, dict) else {}
    robot_version = str(versions.get("RobotControlApp") or "") if isinstance(versions, dict) else ""
    rdk_major_minor = parse_major_minor(rdk_version)
    robot_major_minor = parse_major_minor(robot_version)
    payload: dict[str, Any] = {
        "ok": None,
        "rdkVersion": rdk_version,
        "robotSoftwareVersion": robot_version,
        "rdkMajorMinor": list(rdk_major_minor) if rdk_major_minor else None,
        "robotSoftwareMajorMinor": list(robot_major_minor) if robot_major_minor else None,
        "expectedRobotSoftwareMajorMinor": None,
        "expectedRdkMajorMinor": None,
        "message": "RDK/Elements compatibility could not be determined.",
        "source": "https://www.flexiv.com/software/rdk/manual/robot_software_compatibility.html",
    }
    if not rdk_major_minor or not robot_major_minor:
        return payload
    expected_robot = RDK_ROBOT_SOFTWARE_COMPATIBILITY.get(rdk_major_minor)
    expected_rdk = next(
        (rdk for rdk, robot in RDK_ROBOT_SOFTWARE_COMPATIBILITY.items() if robot == robot_major_minor),
        None,
    )
    if expected_robot:
        payload["expectedRobotSoftwareMajorMinor"] = list(expected_robot)
    if expected_rdk:
        payload["expectedRdkMajorMinor"] = list(expected_rdk)
    if expected_rdk is None and expected_robot is None:
        payload["message"] = (
            f"No local compatibility-table entry for RDK v{rdk_major_minor[0]}.{rdk_major_minor[1]} "
            f"or RobotControlApp v{robot_major_minor[0]}.{robot_major_minor[1]}."
        )
        return payload
    if expected_rdk == rdk_major_minor and expected_robot == robot_major_minor:
        payload["ok"] = True
        payload["message"] = (
            f"RDK/Elements version match: RDK v{rdk_major_minor[0]}.{rdk_major_minor[1]} "
            f"with RobotControlApp v{robot_major_minor[0]}.{robot_major_minor[1]}."
        )
    else:
        payload["ok"] = False
        expected = (
            f"RobotControlApp v{robot_major_minor[0]}.{robot_major_minor[1]} expects "
            f"RDK v{expected_rdk[0]}.{expected_rdk[1]}."
            if expected_rdk
            else f"RDK v{rdk_major_minor[0]}.{rdk_major_minor[1]} expects "
            f"RobotControlApp v{expected_robot[0]}.{expected_robot[1]}."
        )
        payload["message"] = (
            f"RDK/Elements version mismatch: active RDK is v{rdk_major_minor[0]}.{rdk_major_minor[1]}, "
            f"RobotControlApp is v{robot_major_minor[0]}.{robot_major_minor[1]}. {expected}"
        )
    return payload


def parse_major_minor(value: str) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\.(\d+)", str(value))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


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
    compatibility = result.get("compatibility") or {}
    if compatibility:
        print(f"  compatibility: ok={compatibility.get('ok')} {compatibility.get('message')}")
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
    interpretation = result.get("interpretation") or {}
    if interpretation:
        print("  interpretation:")
        for line in interpretation.get("summary") or []:
            print(f"    {line}")
        next_steps = interpretation.get("nextSteps") or []
        if next_steps:
            print("  next steps:")
            for line in next_steps:
                print(f"    {line}")


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


def interpret_result(result: dict[str, Any]) -> dict[str, Any]:
    summary: list[str] = []
    next_steps: list[str] = []
    severity = "ok"

    flexiv = result.get("flexivrdk") or {}
    if flexiv.get("ok"):
        version = flexiv.get("version") or "unknown"
        summary.append(f"Flexiv RDK import OK: {version}.")
    else:
        severity = "error"
        summary.append(f"Flexiv RDK import failed: {flexiv.get('error')}")
        next_steps.append("Install the RDK package that matches Flexiv Elements robot software.")

    system = (result.get("elements") or {}).get("systemVersion") or {}
    versions = system.get("software_version") if isinstance(system, dict) else {}
    robot_version = versions.get("RobotControlApp") if isinstance(versions, dict) else None
    if robot_version:
        summary.append(f"Elements RobotControlApp: {robot_version}.")

    compatibility = result.get("compatibility") or {}
    if compatibility:
        compat_ok = compatibility.get("ok")
        if compat_ok is True:
            summary.append(str(compatibility.get("message") or "RDK/Elements version match."))
        elif compat_ok is False:
            severity = "error"
            summary.append(str(compatibility.get("message") or "RDK/Elements version mismatch."))
            next_steps.append("Install the Flexiv RDK package version that matches RobotControlApp.")
        else:
            summary.append(str(compatibility.get("message") or "RDK/Elements compatibility unknown."))

    probes = result.get("probe") or []
    reachable = [row for row in probes if row.get("ping") or row.get("openPorts")]
    if reachable:
        first = reachable[0]
        summary.append(
            f"Robot network peer reachable: {first.get('host')} open={first.get('openPorts') or []}."
        )
    else:
        severity = "error"
        summary.append("No robot network peer responded to ping or port probes.")
        next_steps.append("Check the Ethernet cable, robot network IP, and the PC interface on the 192.168.2.0/24 network.")

    connection = result.get("robotConnection")
    if connection is not None:
        if connection.get("ok"):
            summary.append("RDK Robot connection OK.")
        else:
            severity = "error"
            error = str(connection.get("error") or "")
            if "All whitelist interfaces were filtered out" in error:
                summary.append("RDK rejected the interface whitelist.")
                next_steps.append("Use the local IPv4 address, for example 192.168.2.108, not an interface name or CIDR.")
            else:
                summary.append("RDK Robot connection failed at discovery.")
                next_steps.append("In Flexiv Elements, enable Remote mode for RDK and select Ethernet.")
                next_steps.append("Put the robot into Auto(Remote), not Manual.")
                next_steps.append("Disable the host firewall or whitelist the RDK Python process on the robot network.")

    realsense = result.get("realsense") or {}
    if not realsense.get("ok"):
        severity = "warning" if severity == "ok" else severity
        summary.append(f"RealSense enumeration failed: {realsense.get('error')}")
    elif not realsense.get("cameras"):
        severity = "warning" if severity == "ok" else severity
        summary.append("No RealSense cameras detected.")

    return {"severity": severity, "summary": summary, "nextSteps": dedupe(next_steps)}


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
