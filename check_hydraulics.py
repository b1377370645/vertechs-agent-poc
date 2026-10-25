#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_hydraulics.py
ZIWIGPT POC — lightweight standpipe-pressure (SPP) and ROP anomaly checker.

Engineering rules
-----------------
1) SPP alarm  : increase greater than 25.0% versus the baseline reading.
2) ROP warning: reduction greater than 50.0% versus the baseline reading.
3) If RPM, WOB and flow are unchanged, the combined SPP/ROP response is
   flagged as a possible downhole restriction / stuck-pipe precursor.

The script uses only the Python standard library. It reads a structured DDR
text file, prints a field-friendly check report, and returns exit code 0 when
the check completes (alarm findings are not process failures). Invalid input
returns exit code 1.

Usage
-----
    python check_hydraulics.py [DDR_FILE] [--json]

Examples
--------
    python check_hydraulics.py
    python check_hydraulics.py sample_ddr.txt
    python check_hydraulics.py sample_ddr.txt --json > check_result.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SPP_ALARM_PCT = 25.0
ROP_WARN_PCT = 50.0
DEFAULT_DDR = Path(__file__).resolve().with_name("sample_ddr.txt")
SECTION_RE = re.compile(r"^\[([A-Z][A-Z0-9_ ]*)\]\s*$")
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


class InputError(ValueError):
    """Raised when the DDR file is missing required data."""


@dataclass(frozen=True)
class Reading:
    time: str
    rpm: float
    wob_kn: float
    spp_mpa: float
    rop_m_h: float
    flow_l_s: float
    torque_kn_m: Optional[float]


@dataclass(frozen=True)
class DDRAnalysis:
    well: str
    measured_depth_m: float
    formation: str
    baseline: Reading
    current: Reading
    spp_limit_mpa: Optional[float]


def configure_stdout() -> None:
    """Keep Chinese output stable when stdout is captured by the harness."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def parse_sections(text: str) -> dict[str, dict[str, str]]:
    """Parse [SECTION] blocks containing 'Key: value' lines."""
    sections: dict[str, dict[str, str]] = {}
    current: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        section_match = SECTION_RE.match(line)
        if section_match:
            current = section_match.group(1)
            sections.setdefault(current, {})
            continue

        if current is None or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip().lower()
        if key:
            sections[current][key] = value.strip()

    return sections


def number_from(fields: dict[str, str], key: str, label: str,
                required: bool = True) -> Optional[float]:
    """Return the first numeric value in a field, with a clear error message."""
    raw = fields.get(key)
    if raw is None or raw == "":
        if required:
            raise InputError(f"缺少字段 / missing field: {label} ({key})")
        return None

    match = NUMBER_RE.search(raw.replace(",", ""))
    if not match:
        if required:
            raise InputError(f"字段无数值 / no numeric value: {label} = {raw!r}")
        return None
    return float(match.group(0))


def reading_from(fields: dict[str, str], section: str) -> Reading:
    return Reading(
        time=fields.get("time", "n/a"),
        rpm=number_from(fields, "rpm", f"{section}.RPM") or 0.0,
        wob_kn=number_from(fields, "wob", f"{section}.WOB") or 0.0,
        spp_mpa=number_from(fields, "spp", f"{section}.SPP") or 0.0,
        rop_m_h=number_from(fields, "rop", f"{section}.ROP") or 0.0,
        flow_l_s=number_from(fields, "flow", f"{section}.FLOW") or 0.0,
        torque_kn_m=number_from(
            fields, "torque", f"{section}.TORQUE", required=False
        ),
    )


def load_ddr(path: Path) -> DDRAnalysis:
    if not path.is_file():
        raise InputError(f"找不到数据文件 / file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"无法读取数据文件 / cannot read file: {exc}") from exc

    sections = parse_sections(text)
    missing = [name for name in ("WELL", "BASELINE", "CURRENT")
               if name not in sections]
    if missing:
        raise InputError("缺少数据段 / missing section(s): " + ", ".join(missing))

    well_fields = sections["WELL"]
    baseline = reading_from(sections["BASELINE"], "BASELINE")
    current = reading_from(sections["CURRENT"], "CURRENT")

    for name, reading in (("BASELINE", baseline), ("CURRENT", current)):
        for label, value in (
            ("SPP", reading.spp_mpa),
            ("ROP", reading.rop_m_h),
            ("FLOW", reading.flow_l_s),
        ):
            if value <= 0:
                raise InputError(
                    f"{name}.{label} 必须为正数 / must be positive: {value}"
                )

    well = well_fields.get("well", "").strip()
    formation = well_fields.get("formation", "n/a").strip()
    depth = number_from(
        well_fields, "measured depth", "WELL.Measured Depth", required=True
    )
    if not well:
        raise InputError("缺少井号 / missing WELL.Well")

    return DDRAnalysis(
        well=well,
        measured_depth_m=float(depth),
        formation=formation,
        baseline=baseline,
        current=current,
        spp_limit_mpa=number_from(
            well_fields,
            "spp continuous limit",
            "WELL.SPP Continuous Limit",
            required=False,
        ),
    )


def pct_change(current: float, baseline: float) -> float:
    return (current - baseline) / baseline * 100.0


def analyze(data: DDRAnalysis) -> dict:
    """Apply the rules and build a JSON-serialisable result."""
    base = data.baseline
    cur = data.current

    spp_delta = cur.spp_mpa - base.spp_mpa
    spp_change_pct = pct_change(cur.spp_mpa, base.spp_mpa)
    spp_threshold_mpa = base.spp_mpa * (1.0 + SPP_ALARM_PCT / 100.0)
    spp_triggered = spp_change_pct > SPP_ALARM_PCT

    rop_delta = cur.rop_m_h - base.rop_m_h
    rop_drop_pct = (base.rop_m_h - cur.rop_m_h) / base.rop_m_h * 100.0
    rop_threshold_m_h = base.rop_m_h * (1.0 - ROP_WARN_PCT / 100.0)
    rop_triggered = rop_drop_pct > ROP_WARN_PCT

    base_kw = base.spp_mpa * base.flow_l_s
    current_kw = cur.spp_mpa * cur.flow_l_s
    power_delta_kw = current_kw - base_kw

    utilization_pct: Optional[float] = None
    margin_mpa: Optional[float] = None
    if data.spp_limit_mpa is not None and data.spp_limit_mpa > 0:
        margin_mpa = data.spp_limit_mpa - cur.spp_mpa
        utilization_pct = cur.spp_mpa / data.spp_limit_mpa * 100.0

    torque_base = base.torque_kn_m
    torque_current = cur.torque_kn_m
    torque_change_pct: Optional[float] = None
    if torque_base not in (None, 0) and torque_current is not None:
        torque_change_pct = pct_change(torque_current, torque_base)

    controls_unchanged = all(
        abs(value) < 0.01
        for value in (
            pct_change(cur.rpm, base.rpm),
            pct_change(cur.wob_kn, base.wob_kn),
            pct_change(cur.flow_l_s, base.flow_l_s),
        )
    )

    triggers = int(spp_triggered) + int(rop_triggered)
    if triggers == 2:
        risk_label = "HIGH / 高风险"
        risk_code = "RED"
    elif spp_triggered:
        risk_label = "HIGH / 高风险"
        risk_code = "ORANGE"
    elif rop_triggered:
        risk_label = "MEDIUM / 中风险"
        risk_code = "YELLOW"
    else:
        risk_label = "LOW / 低风险"
        risk_code = "GREEN"

    if spp_triggered and rop_triggered and controls_unchanged:
        likely_mechanism = (
            "环空局部堵塞/岩屑床 或 钻头泥包/喷嘴堵塞；"
            "卡钻前兆风险升高"
        )
    elif spp_triggered:
        likely_mechanism = "循环系统压耗异常升高，排查环空憋堵、喷嘴堵塞或钻井液性能变化"
    elif rop_triggered:
        likely_mechanism = "机械钻速突降，排查钻头磨损/泥包、地层变化或参数传递失效"
    else:
        likely_mechanism = "未触发 SPP/ROP 异常规则，继续趋势监测"

    return {
        "well": data.well,
        "measured_depth_m": data.measured_depth_m,
        "formation": data.formation,
        "baseline_time": base.time,
        "current_time": cur.time,
        "spp": {
            "baseline_mpa": round(base.spp_mpa, 2),
            "current_mpa": round(cur.spp_mpa, 2),
            "delta_mpa": round(spp_delta, 2),
            "change_pct": round(spp_change_pct, 2),
            "threshold_pct": SPP_ALARM_PCT,
            "threshold_mpa": round(spp_threshold_mpa, 2),
            "excess_mpa": round(cur.spp_mpa - spp_threshold_mpa, 2),
            "triggered": spp_triggered,
            "status": "ALARM" if spp_triggered else "NORMAL",
        },
        "rop": {
            "baseline_m_h": round(base.rop_m_h, 2),
            "current_m_h": round(cur.rop_m_h, 2),
            "delta_m_h": round(rop_delta, 2),
            "drop_pct": round(rop_drop_pct, 2),
            "threshold_pct": ROP_WARN_PCT,
            "threshold_m_h": round(rop_threshold_m_h, 2),
            "shortfall_m_h": round(cur.rop_m_h - rop_threshold_m_h, 2),
            "triggered": rop_triggered,
            "status": "WARNING" if rop_triggered else "NORMAL",
        },
        "hydraulics": {
            "flow_l_s": round(cur.flow_l_s, 1),
            "baseline_pump_power_kw": round(base_kw, 1),
            "current_pump_power_kw": round(current_kw, 1),
            "power_delta_kw": round(power_delta_kw, 1),
            "spp_limit_mpa": (
                round(data.spp_limit_mpa, 2)
                if data.spp_limit_mpa is not None else None
            ),
            "remaining_margin_mpa": (
                round(margin_mpa, 2) if margin_mpa is not None else None
            ),
            "utilization_pct": (
                round(utilization_pct, 2)
                if utilization_pct is not None else None
            ),
        },
        "controls": {
            "rpm_baseline": base.rpm,
            "rpm_current": cur.rpm,
            "wob_baseline_kn": base.wob_kn,
            "wob_current_kn": cur.wob_kn,
            "flow_baseline_l_s": base.flow_l_s,
            "flow_current_l_s": cur.flow_l_s,
            "unchanged": controls_unchanged,
            "torque_baseline_kn_m": torque_base,
            "torque_current_kn_m": torque_current,
            "torque_change_pct": (
                round(torque_change_pct, 2)
                if torque_change_pct is not None else None
            ),
        },
        "overall": {
            "risk_label": risk_label,
            "risk_code": risk_code,
            "triggers": triggers,
            "likely_mechanism": likely_mechanism,
        },
    }


# Public integration aliases used by app.py and other callers.
loadddr = load_ddr
analyzedata = analyze


def print_report(result: dict, source: Path) -> None:
    spp = result["spp"]
    rop = result["rop"]
    hyd = result["hydraulics"]
    ctrl = result["controls"]
    overall = result["overall"]
    width = 80

    print("=" * width)
    print(" ZIWIGPT POC | 钻井水力学与 ROP 异常校验")
    print(" Hydraulics & ROP Anomaly Check")
    print("=" * width)
    print(f"Well              : {result['well']}")
    print(f"Measured Depth    : {result['measured_depth_m']:,.0f} m")
    print(f"Formation         : {result['formation']}")
    print(f"Baseline -> Now   : {result['baseline_time']} -> {result['current_time']}")
    print(f"Source            : {source}")
    print("-" * width)

    print(f"[1] 立管压力 SPP / Standpipe Pressure     阈值: +{spp['threshold_pct']:.1f}%")
    print(f"    Baseline        : {spp['baseline_mpa']:.2f} MPa")
    print(f"    Current         : {spp['current_mpa']:.2f} MPa")
    print(f"    Change          : {spp['delta_mpa']:+.2f} MPa ({spp['change_pct']:+.2f}%)")
    print(f"    Alarm level     : {spp['threshold_mpa']:.2f} MPa (baseline x 1.25)")
    print(f"    Threshold excess: {spp['excess_mpa']:+.2f} MPa")
    if spp["triggered"]:
        print("    Result          : ALARM / 告警  [TRIGGERED]")
    else:
        print("    Result          : NORMAL / 正常  [WITHIN LIMIT]")
    print()

    print(f"[2] 机械钻速 ROP / Rate of Penetration    阈值: -{rop['threshold_pct']:.1f}%")
    print(f"    Baseline        : {rop['baseline_m_h']:.2f} m/h")
    print(f"    Current         : {rop['current_m_h']:.2f} m/h")
    print(f"    Change          : {rop['delta_m_h']:+.2f} m/h (-{rop['drop_pct']:.2f}%)")
    print(f"    Warning level   : {rop['threshold_m_h']:.2f} m/h (baseline x 0.50)")
    print(f"    Threshold short : {rop['shortfall_m_h']:+.2f} m/h")
    if rop["triggered"]:
        print("    Result          : WARNING / 预警  [TRIGGERED]")
    else:
        print("    Result          : NORMAL / 正常  [WITHIN LIMIT]")
    print()

    print("[3] 水力载荷与设备余量 / Hydraulic Load & Margin")
    print(f"    Flow rate       : {hyd['flow_l_s']:.1f} L/s")
    print(
        "    Pump power      : "
        f"{hyd['baseline_pump_power_kw']:.1f} -> "
        f"{hyd['current_pump_power_kw']:.1f} kW "
        f"({hyd['power_delta_kw']:+.1f} kW)"
    )
    if hyd["spp_limit_mpa"] is not None and hyd["remaining_margin_mpa"] is not None:
        print(f"    SPP limit       : {hyd['spp_limit_mpa']:.2f} MPa")
        print(
            "    Margin          : "
            f"{hyd['remaining_margin_mpa']:+.2f} MPa "
            f"({hyd['utilization_pct']:.2f}% utilization)"
        )
        if hyd["remaining_margin_mpa"] < 0:
            print("    Result          : LIMIT EXCEEDED / 超限")
        elif hyd["utilization_pct"] >= 95.0:
            print("    Result          : CRITICAL MARGIN / 余量不足")
        else:
            print("    Result          : WITHIN LIMIT / 余量正常")
    else:
        print("    SPP limit       : NOT PROVIDED / 未提供")
    print()

    print("[4] 控制变量与控制指标 / Control Variables")
    print(f"    RPM             : {ctrl['rpm_baseline']:.0f} -> {ctrl['rpm_current']:.0f}")
    print(f"    WOB             : {ctrl['wob_baseline_kn']:.0f} -> {ctrl['wob_current_kn']:.0f} kN")
    print(f"    Flow            : {ctrl['flow_baseline_l_s']:.0f} -> {ctrl['flow_current_l_s']:.0f} L/s")
    if ctrl["torque_change_pct"] is not None:
        print(
            "    Torque          : "
            f"{ctrl['torque_baseline_kn_m']:.1f} -> "
            f"{ctrl['torque_current_kn_m']:.1f} kN·m "
            f"({ctrl['torque_change_pct']:+.2f}%) [supporting]"
        )
    unchanged_text = "YES / 是" if ctrl["unchanged"] else "NO / 否"
    print(f"    RPM/WOB/Q stable: {unchanged_text}")
    print("-" * width)

    print(f"OVERALL RISK    : {overall['risk_label']} ({overall['risk_code']})")
    print(f"TRIGGERS        : {overall['triggers']}/2 (SPP alarm + ROP warning)")
    print(f"PRELIM. CAUSE   : {overall['likely_mechanism']}")
    if spp["triggered"] and rop["triggered"]:
        print("STUCK-PIPE RISK : ELEVATED / 前兆风险升高")
    print("RECOMMENDATION  : 停钻并保持循环；核对 SPP、扭矩、出口流量与返出量；")
    print("                  循环稳定后短起下，必要时提钻检查钻头、喷嘴与 BHA。")
    print("NOTE            : POC 规则校验，不替代现场工程师判断。")
    print("=" * width)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ZIWIGPT POC: check SPP and ROP anomalies in a DDR text file."
    )
    parser.add_argument(
        "ddr_file",
        nargs="?",
        type=Path,
        default=DEFAULT_DDR,
        help=f"path to the DDR file (default: {DEFAULT_DDR.name})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable JSON result instead of the text report",
    )
    return parser


def main() -> int:
    configure_stdout()
    args = build_parser().parse_args()
    source = args.ddr_file.expanduser().resolve()

    try:
        data = load_ddr(source)
        result = analyze(data)
    except (InputError, OSError, ValueError) as exc:
        print(f"[INPUT ERROR] {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_report(result, source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
