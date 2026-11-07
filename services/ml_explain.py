"""
Turns what the trained model found into the same kind of information a rule-based alert gives (v3.1):
which sensors are odd and their real values, why the model calls the AC abnormal, how serious it is,
what could go wrong, and what to do.

Pure Python (no web or database code). The input is the result of ml_per_ac.score_reading():
margin, distance_ratio, checks, operating_mode and `deviations` (each sensor's real value, this AC's usual range
while running/idle, and how many "spreads" away it is). Nothing here is a fixed safety limit: everything is compared
with what THIS AC learned as normal.
"""

UNITS = {
    "pzem_current": "A", "pzem_voltage": "V", "pzem_power": "W", "pzem_frequency": "Hz", "pzem_power_factor": "",
    "dht_temp": "°C", "dht_humidity": "%", "ds18b20_temp1": "°C", "ds18b20_temp2": "°C", "vibration": "Hz",
}
NAMES = {
    "pzem_current": "Operating current", "pzem_voltage": "Supply voltage", "pzem_power": "Power usage",
    "pzem_frequency": "Line frequency", "pzem_power_factor": "Power factor", "dht_temp": "Output air temperature",
    "dht_humidity": "Output humidity", "ds18b20_temp1": "Outlet compressor temperature",
    "ds18b20_temp2": "Inlet compressor temperature", "vibration": "Compressor vibration",
}
SHORT = {  # lower-case names for sentences / SMS
    "pzem_current": "current", "pzem_voltage": "voltage", "pzem_power": "power", "pzem_frequency": "frequency",
    "pzem_power_factor": "power factor", "dht_temp": "air temp", "dht_humidity": "humidity",
    "ds18b20_temp1": "outlet temp", "ds18b20_temp2": "inlet temp", "vibration": "vibration",
}
DECIMALS = {"pzem_power_factor": 2, "pzem_frequency": 2, "pzem_current": 2, "pzem_voltage": 1, "pzem_power": 0,
            "dht_temp": 1, "dht_humidity": 1, "ds18b20_temp1": 1, "ds18b20_temp2": 1, "vibration": 1}
NON_NEGATIVE = {"pzem_current", "pzem_voltage", "pzem_power", "pzem_frequency", "pzem_power_factor", "dht_humidity", "vibration"}

# What each odd reading can mean, which direction it went. (means, risk, do_now, check, call_technician_if)
KNOW = {
    ("pzem_current", "above"): ("The AC is drawing more current than it normally does.",
        "Overloaded compressor or fan motor, a tripping breaker, and heated wiring.",
        ["If the current keeps climbing or you smell burning, switch the AC off at the breaker.", "Check for a dirty filter, a dirty outdoor coil or something blocking the airflow."],
        ["Compressor and fan motor.", "Run capacitor."], "the current stays high after cleaning the filter and coils."),
    ("pzem_current", "below"): ("The AC is drawing less current than it normally does.",
        "The compressor is not pulling its full load: low refrigerant, a weak capacitor or a failing compressor. Cooling will be poor.",
        ["Feel the air: is it cooling as well as usual?", "Check that the compressor is actually running (listen for it)."],
        ["Run capacitor and compressor.", "Refrigerant level."], "the air is not cold or the compressor does not run."),
    ("pzem_power", "above"): ("The AC is using more power than it normally does.",
        "Higher electricity bill and extra stress on the compressor.",
        ["Clean the air filter.", "Check that nothing blocks the outdoor side of the unit."],
        ["Compressor load.", "Condenser coil."], "the power stays high after cleaning."),
    ("pzem_power", "below"): ("The AC is using less power than it normally does.",
        "The compressor is probably not working fully, so cooling drops.",
        ["Feel the air: is it cooling as well as usual?", "Check the thermostat setting."],
        ["Compressor.", "Refrigerant level."], "cooling is weak."),
    ("pzem_voltage", "above"): ("The supply voltage is higher than this AC normally sees.",
        "Over-voltage can damage the compressor and the control board.",
        ["Switch the AC off if the voltage stays high.", "Ask the building's electrician to check the supply."],
        ["Main supply and voltage regulator."], "the high voltage lasts more than a few minutes."),
    ("pzem_voltage", "below"): ("The supply voltage is lower than this AC normally sees.",
        "Low voltage makes the compressor draw more current and overheat, and it may fail to start.",
        ["Switch the AC off if the voltage stays low.", "Check if other big appliances are running on the same line."],
        ["Main supply, wiring and breaker."], "the voltage stays low."),
    ("pzem_power_factor", "above"): ("The power factor is higher than usual.", "Usually harmless, but it means the motor load has changed.",
        ["Compare with the graphs of a normal day."], ["Run capacitor."], "it keeps changing."),
    ("pzem_power_factor", "below"): ("The power factor is lower than usual.", "A weak run capacitor or a struggling motor wastes power and heats the motor.",
        ["Listen for humming or clicking from the compressor."], ["Run capacitor."], "the AC hums or struggles to start."),
    ("pzem_frequency", "above"): ("The line frequency is off its usual value.", "An unstable supply (generator or grid) can stress the motors.",
        ["Check whether the building is on a generator."], ["Power source."], "it keeps drifting."),
    ("pzem_frequency", "below"): ("The line frequency is off its usual value.", "An unstable supply (generator or grid) can stress the motors.",
        ["Check whether the building is on a generator."], ["Power source."], "it keeps drifting."),
    ("dht_temp", "above"): ("The air coming out is warmer than usual.",
        "Cooling is weaker: low refrigerant, a dirty filter or coil, or a blocked water drain.",
        ["Clean the air filter.", "Check that the water drain is not blocked."],
        ["Filter, coils and drain.", "Refrigerant level."], "the air stays warm after cleaning the filter."),
    ("dht_temp", "below"): ("The air coming out is colder than usual.",
        "The coil may be freezing because of low airflow (dirty filter) or low refrigerant, which can damage the compressor.",
        ["Check that the filter is clean and the fan is spinning.", "If ice is visible, switch the AC off and let it thaw."],
        ["Filter, evaporator coil and fan."], "ice keeps coming back."),
    ("dht_humidity", "above"): ("The air coming out is more humid than usual.",
        "Often a blocked drain or weak airflow. Humidity is shown for information; it is not a limit alert.",
        ["Check that the water drain is not blocked.", "Check that the filter is clean."],
        ["Drain line.", "Airflow."], "water drips inside or the air feels clammy for days."),
    ("dht_humidity", "below"): ("The air coming out is drier than usual.", "Check the humidity sensor and the airflow.",
        ["Check the sensor and the airflow through the unit."], ["DHT22 sensor."], "it stays low for days."),
    ("ds18b20_temp1", "above"): ("The compressor outlet line is hotter than usual.",
        "Compressor overheating: low refrigerant, a dirty condenser or a failing compressor. It can seize if it continues.",
        ["Switch the AC off for 15 minutes to cool down.", "Clean the outdoor coil and make sure it has airflow."],
        ["Condenser coil.", "Refrigerant level."], "it overheats again after cleaning."),
    ("ds18b20_temp1", "below"): ("The compressor outlet line is cooler than usual.",
        "The compressor may not be pumping properly.", ["Listen: is the compressor running?"], ["Compressor."], "cooling is weak."),
    ("ds18b20_temp2", "above"): ("The compressor inlet line is warmer than usual.",
        "Often low refrigerant: the system cannot absorb enough heat, so cooling drops.",
        ["Check that the filter is clean.", "Feel the air: is it cooling as well as usual?"], ["Refrigerant level."], "cooling is weak."),
    ("ds18b20_temp2", "below"): ("The compressor inlet line is colder than usual.",
        "Risk of icing, usually from low airflow (dirty filter) or low refrigerant.",
        ["Check that the filter is clean and the fan is spinning.", "If ice is visible, switch the AC off and let it thaw."],
        ["Filter and evaporator coil."], "ice keeps coming back."),
    ("vibration", "above"): ("The compressor vibrates more than usual.",
        "Loose mounts, worn bearings or an unbalanced fan. It can get worse quickly and damage the compressor.",
        ["Listen for rattling or grinding.", "Check the compressor mounts and fan blades."],
        ["Compressor mounts.", "Fan and bearings."], "the noise or vibration continues."),
    ("vibration", "below"): ("The compressor vibrates less than usual.",
        "The compressor may not be running even though other readings say the AC is on, or the sensor came loose.",
        ["Check that the compressor is actually running.", "Check that the vibration sensor is still fixed in place."],
        ["Compressor.", "Vibration sensor mount."], "the compressor does not run."),
}
_GENERIC = ("This reading is far from what this AC normally shows.", "Something changed in how the AC runs.",
            ["Look at the AC: listen for new noises, feel the air, check the filter."], ["Compare with a normal day on the graphs."],
            "it keeps happening for a day.")

SEV_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}


def severity_of(ml):
    """How serious the model's finding is, from how far past normal it is (not from any safety limit).

    Calibrated on generated fault data (tests/fault_data.py): marginal drift -> Low/Medium, one sensor clearly off -> Medium,
    wrong supply voltage or a frozen coil -> High, an overload or a stuck compressor -> Critical.
    """
    ratio = ml.get("distance_ratio")
    zmax = max((d.get("z", 0) for d in ml.get("deviations") or []), default=0)
    r = ratio if ratio is not None else 0
    if r >= 14 or zmax >= 30:
        return "Critical"
    if r >= 6.5 or zmax >= 15:
        return "High"
    if r >= 3.5 or zmax >= 7:
        return "Medium"
    return "Low"


def sensor_severity(z):
    return "High" if z >= 20 else "Medium" if z >= 8 else "Low"


def condition_of(severity, odd):
    """The AC condition label the Alerts pages show (same words as for rule-based alerts)."""
    keys = {d["key"]: d["direction"] for d in odd}
    if severity == "Critical":
        return "Need Repair"
    if severity == "High":
        overload = keys.get("pzem_current") == "above" or keys.get("pzem_power") == "above" or keys.get("ds18b20_temp1") == "above"
        if overload:
            return "Need Repair"
        if keys and set(keys) <= {"pzem_voltage", "pzem_frequency", "pzem_power_factor"}:
            return "Check Power Supply"
    return "Need Maintenance"


def _fmt(key, v):
    return f"{v:.{DECIMALS.get(key, 1)}f}"


def _range(key, low, high):
    if key in NON_NEGATIVE:
        low = max(low, 0.0)
    return f"{_fmt(key, low)}–{_fmt(key, high)}"


def _sensor_entry(d):
    key, unit = d["key"], UNITS.get(d["key"], "")
    means, risk, do_now, check, call = KNOW.get((key, d["direction"]), _GENERIC)
    diff = d["value"] - d["mid"]
    sev = sensor_severity(d["z"])
    usual = _range(key, d["low"], d["high"])
    return {
        "key": key, "name": NAMES.get(key, key), "short": SHORT.get(key, key), "unit": unit,
        "value": d["value"], "value_text": f"{_fmt(key, d['value'])} {unit}".strip(),
        "usual_low": d["low"], "usual_high": d["high"], "usual_text": f"{usual} {unit}".strip(),
        "usual_mid": d["mid"], "difference": diff, "direction": d["direction"], "z": d["z"], "severity": sev,
        "means": means, "risk": risk, "do_now": do_now, "check": check, "call": call,
        "compare": f"{'+' if diff >= 0 else '-'}{_fmt(key, abs(diff))} {unit}".strip() + (" above usual" if diff >= 0 else " below usual"),
    }


def assess(ml, reading=None, label=None):
    """Full explanation of an ML finding, or None when the model found nothing abnormal."""
    if not ml or not ml.get("is_anomaly"):
        return None
    label = label or "This AC"
    sev = severity_of(ml)
    odd = [_sensor_entry(d) for d in (ml.get("deviations") or [])]
    mode = ml.get("operating_mode", "running")
    margin = float(ml.get("margin", 0.0))
    ratio = ml.get("distance_ratio")
    checks = ml.get("checks") or []

    # ---- why: plain language, with the real numbers
    why = [f"The model learned how {label} normally runs from {int(ml.get('n_rows') or 0):,} of its own readings."] if ml.get("n_rows") else \
          [f"The model learned how {label} normally runs from its own readings."]
    if odd:
        bits = [f"{o['name'].lower()} is {o['value_text']} (usually {o['usual_text']})" for o in odd[:3]]
        why.append(f"This reading, taken while the compressor was {mode}, does not match: " + "; ".join(bits) + ".")
    else:
        why.append("No single sensor is far out, but the readings together (power, current, temperatures) do not fit the way this AC normally runs.")
    how = []
    if "forest" in checks:
        how.append("an unusual combination of readings (Isolation Forest)")
    if "distance" in checks:
        how.append(f"too far from every normal reading it has seen ({ratio:.1f}× the allowed distance)" if ratio else "too far from every normal reading it has seen")
    why.append(("It was flagged because of " + " and ".join(how) + ". ") if how else "")
    why.append(f"Distance from normal: {margin:+.2f} (below 0 is abnormal, 0 is the edge of normal).")
    why.append("This compares the AC with its own history, not with a fixed safety limit, so it can alert even when every limit is fine.")
    why_text = " ".join(w for w in why if w)

    # ---- findings: same shape as the rule engine's, so the Alerts page can show them the same way
    findings = []
    for o in odd:
        findings.append({
            "sensor": o["name"], "sensor_key": o["key"], "measured_value": o["value_text"], "unit": o["unit"],
            "normal_range": o["usual_text"], "computed_expression": f"ML: {o['z']:.1f} spreads {o['direction']} the usual value",
            "issue": f"{o['name']} is {'higher' if o['direction'] == 'above' else 'lower'} than this AC normally runs",
            "severity": o["severity"], "status": "Current", "origin": "ml",
            "root_cause": f"{o['means']} {o['risk']}", "recommended_action": o["do_now"][0],
        })
    if not odd:
        findings.append({
            "sensor": "Pattern of readings", "sensor_key": "ml_pattern", "measured_value": "", "unit": "", "normal_range": "",
            "computed_expression": f"ML: distance from normal {margin:+.2f}", "issue": "The readings do not fit together like this AC normally runs",
            "severity": sev if sev in ("Low", "Medium") else "Medium", "status": "Current", "origin": "ml",
            "root_cause": "Each sensor may look fine alone, but together they match no normal operating pattern the model learned. This is often the first sign of a developing problem.",
            "recommended_action": "Look at the AC: listen for new noises, feel the air, check the filter.",
        })

    # ---- action plan steps (same shape as action_guide's steps)
    urgency = {"Critical": "stop_now", "High": "today", "Medium": "today", "Low": "soon"}[sev]
    steps = []
    for o in odd:
        steps.append({
            "id": f"ml_{o['key']}", "title": f"{o['name']} is {'higher' if o['direction'] == 'above' else 'lower'} than normal",
            "severity": o["severity"], "urgency": urgency if SEV_RANK[o["severity"]] >= SEV_RANK.get(sev, 1) - 1 else "soon",
            "measured": o["value_text"], "normal": o["usual_text"], "short": o["do_now"][0],
            "why": f"{o['means']} {o['risk']}", "do_now": list(o["do_now"]), "check": list(o["check"]),
            "call_technician_if": o["call"], "origin": "ml",
        })
    if not odd:
        steps.append({
            "id": "ml_pattern", "title": "The AC is running differently from its normal", "severity": "Medium", "urgency": urgency,
            "measured": "", "normal": "", "short": "Look at the AC: listen for new noises, feel the air, check the filter.",
            "why": "The combination of readings no longer looks like this AC's usual pattern, even though no single value is far out.",
            "do_now": ["Look at the AC: listen for new noises, feel the air, check the filter."],
            "check": ["Compare today's readings with a normal day on the graphs."],
            "call_technician_if": "it keeps happening for a day or the cooling gets worse.", "origin": "ml",
        })
    risks, actions = [], []
    for o in odd[:3]:
        if o["risk"] not in risks:
            risks.append(o["risk"])
        for a in o["do_now"]:
            if a not in actions:
                actions.append(a)
    if not odd:
        risks = ["An unnoticed fault can grow and damage the compressor."]
        actions = ["Look at the AC: listen for new noises, feel the air, check the filter."]

    # ---- short lines for SMS / ntfy
    n = len(odd)
    head = (f"ML model [{sev}]: {label} is off its learned normal" + (f" ({n} sensor{'s' if n != 1 else ''})" if n else " (unusual pattern)"))
    lines = [head] + [f"{o['name']} {o['value_text']} (usual {o['usual_text']})" for o in odd[:4]]
    if n > 4:
        lines.append(f"+{n - 4} more sensors")
    if risks:
        lines.append("Risk: " + risks[0])
    sms_bits = [f"{o['short']} {_fmt(o['key'], o['value'])}{o['unit']} (usual {_range(o['key'], o['usual_low'], o['usual_high'])})" for o in odd[:2]]
    sms_line = f"ML [{sev}]: " + (", ".join(sms_bits) + (f" +{n - 2} more" if n > 2 else "") if odd else "readings do not fit this AC's normal pattern")

    return {
        "severity": sev, "urgency": urgency, "condition": condition_of(sev, odd), "headline": head, "why": why_text,
        "odd": odd, "findings": findings, "steps": steps, "risks": risks, "actions": actions[:4], "lines": lines, "sms_line": sms_line,
        "model": {"margin": margin, "distance_ratio": ratio, "checks": checks, "operating_mode": mode,
                  "trained_at": ml.get("trained_at"), "n_rows": ml.get("n_rows")},
    }
