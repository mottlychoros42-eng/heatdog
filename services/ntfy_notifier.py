"""
ntfy push notifications for AirBnBrrr.

One alert event = ONE short notification: the worst problems in plain words, then what to do.
Since v3.0 an alert comes from ONE source (ML model, Safety limits or Sensor offline) and the title says which.
The full details (all readings, every step) are on the Alerts page.

Configure with environment variables on Render:
    NTFY_TOPIC    required. Use a long random name, public topics are readable by anyone who
                  knows the name, e.g. AirBnrr_alien_test_code_k8x2m9qv4t
    NTFY_SERVER   optional, default https://ntfy.sh (set it if you self-host)
    NTFY_TOKEN    optional access token, for protected topics
    APP_URL       optional, your site address; tapping the notification opens the Alerts page
Uses only the standard library, so there is nothing new to install.
"""
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from services import action_guide

SHOW_PROBLEMS = 3     # problems listed in the notification
MAX_BYTES = 1500


def finding_text(f):
    """One plain line for a rule-engine finding (dict) or a plain string."""
    if isinstance(f, str):
        return f.strip()
    if isinstance(f, dict):
        return action_guide.plain_line(f)
    return str(f)


_SEV = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}


def build_problems(critical_reasons, findings, ml):
    """Plain problem lines, worst first, no duplicates (the AI note goes last).

    The rule engine uses the same limits as the instant safety alerts, so when it has findings the raw
    limit lines would only repeat them; they are used only when the rule engine found nothing.
    """
    ordered = sorted(findings or [], key=lambda f: -_SEV.get(f.get("severity") if isinstance(f, dict) else None, 0))
    items = [finding_text(f) for f in ordered] if ordered else list(critical_reasons or [])
    line = action_guide.ml_line(ml)
    if line:
        items.append(line)
    seen, out = set(), []
    for it in items:
        key = it.lower()
        if it and key not in seen:
            seen.add(key)
            out.append(it)
    return out


SOURCE_TITLE = {"ml": "ML", "safety": "Safety", "sensor": "Sensor"}
SOURCE_TAGS = {"ml": "robot", "safety": "rotating_light", "sensor": "electric_plug"}


def build_message(unit_title, is_critical, problems, readings=None, actions=None, source=None, assessment=None):
    """Return (title, body). Title is plain ASCII because it travels in an HTTP header.

    The title says where the alert comes from: 'Check (ML): AC1', 'STOP (Safety): AC1', 'Check (Sensor): AC1'.
    An ML alert (`assessment` given) lists the odd sensors with their real values and the AC's usual range, the
    severity, the main risk and what to do, like a rule-based alert does.
    """
    tag = SOURCE_TITLE.get(source)
    head = ("STOP" if is_critical else "Check") + (f" ({tag})" if tag else "")
    title = f"{head}: {unit_title}".encode("ascii", "replace").decode("ascii")
    if assessment:
        lines = [assessment["lines"][0]] + [f"- {l}" for l in assessment["lines"][1:-1]] + ([assessment["lines"][-1]] if len(assessment["lines"]) > 1 else [])
        lines.append("Why: " + (assessment["why"].split(". ")[1] if ". " in assessment["why"] else assessment["why"]).rstrip(".") + ".")
    else:
        ai = [p for p in problems if p.startswith("ML model")]
        real = [p for p in problems if not p.startswith("ML model")]
        lines = [f"- {p}" for p in real[:SHOW_PROBLEMS]]
        if len(real) > SHOW_PROBLEMS:
            lines.append(f"+ {len(real) - SHOW_PROBLEMS} more on the Alerts page")
        if ai:
            lines.append(ai[0])
        if not lines:
            lines.append("The AC is acting differently from its normal.")
    if actions:
        lines += ["", "Do this now:"] + [f"{i}. {a}" for i, a in enumerate(actions, start=1)]
    body = "\n".join(lines)
    return title, body.encode("utf-8")[:MAX_BYTES].decode("utf-8", "ignore")


RETRY_WAITS = (0, 1.5, 4.0)       # seconds before each attempt: network errors, HTTP 5xx and 429 are retried
_status_lock = threading.Lock()
_status = {"configured": None, "last_ok": None, "last_error": None, "last_attempt_at": None, "last_http_status": None,
           "last_title": None, "sent": 0, "failed": 0}


def _record(ok, title, http_status=None, error=None):
    with _status_lock:
        _status["last_attempt_at"] = datetime.now(timezone.utc).isoformat()
        _status["last_title"] = title
        _status["last_http_status"] = http_status
        _status["last_ok"] = ok
        _status["last_error"] = None if ok else error
        _status["sent" if ok else "failed"] += 1


def status():
    """What the dashboard shows: is ntfy set up, and did the last notification arrive (and if not, why)."""
    topic = _topic()
    with _status_lock:
        out = dict(_status)
    out["configured"] = bool(topic)
    out["topic_hint"] = (topic[:4] + "…") if topic else None
    out["server"] = _server()
    return out


def _topic():
    """The topic from NTFY_TOPIC, tolerating spaces and quotes pasted by mistake."""
    return os.environ.get("NTFY_TOPIC", "").strip().strip("\"'").strip().strip("/")


def _server():
    """NTFY_SERVER with https:// added when it was left out (a bare 'ntfy.sh' makes urllib fail with 'unknown url type')."""
    server = os.environ.get("NTFY_SERVER", "").strip().strip("\"'").rstrip("/") or "https://ntfy.sh"
    if "://" not in server:
        server = "https://" + server
    return server


def _post(title, body, priority, tags):
    topic = _topic()
    if not topic:
        msg = "NTFY_TOPIC is not set on the server, so no notification can be sent"
        print(f"[NTFY] {msg}", flush=True)
        _record(False, title, None, msg)
        return False
    url = f"{_server()}/{urllib.parse.quote(topic, safe='')}"
    headers = {"Title": title.replace("\n", " ")[:200], "Priority": str(priority), "Tags": tags,
               "User-Agent": "AirBnBrrr/3.1 (+ntfy)", "Content-Type": "text/plain; charset=utf-8"}
    if os.environ.get("APP_URL"):
        headers["Click"] = os.environ["APP_URL"].strip().rstrip("/") + "/pages/alerts"
    if os.environ.get("NTFY_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["NTFY_TOKEN"].strip()
    data = body.encode("utf-8")

    last_error, http_status = "unknown error", None
    for attempt, wait in enumerate(RETRY_WAITS, start=1):
        if wait:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, data=data, method="POST", headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                http_status = resp.status
                if 200 <= resp.status < 300:
                    print(f"[NTFY] sent '{title}' (HTTP {resp.status}, attempt {attempt})", flush=True)
                    _record(True, title, resp.status)
                    return True
                last_error = f"unexpected HTTP status {resp.status}"
        except urllib.error.HTTPError as e:      # the server answered with an error: say exactly which and why
            http_status = e.code
            try:
                detail = e.read().decode("utf-8", "replace").strip()[:200]
            except Exception:
                detail = ""
            hint = {401: "the topic needs a login: set NTFY_TOKEN", 403: "access denied: wrong NTFY_TOKEN or a protected topic",
                    404: "wrong NTFY_SERVER address", 429: "rate limited by the ntfy server (too many messages from this IP: use NTFY_TOKEN or self-host)"
                    }.get(e.code, "")
            last_error = f"HTTP {e.code}" + (f" ({hint})" if hint else "") + (f": {detail}" if detail else "")
            if e.code < 500 and e.code != 429:   # retrying a 4xx would not help
                break
        except Exception as e:                   # no answer at all: DNS, timeout, blocked connection
            last_error = f"{type(e).__name__}: {e}"
        print(f"[NTFY] attempt {attempt}/{len(RETRY_WAITS)} failed for '{title}': {last_error}", flush=True)
    print(f"[NTFY] NOT delivered '{title}': {last_error}", flush=True)
    _record(False, title, http_status, last_error)
    return False


def send_alert(unit_title, is_critical, problems, readings=None, actions=None, source=None, assessment=None):
    title, body = build_message(unit_title, is_critical, problems, readings, actions, source, assessment)
    base = SOURCE_TAGS.get(source, "rotating_light" if is_critical else "warning")
    tags = base if source is None else f"{base},{'rotating_light' if is_critical else 'warning'}"
    return _post(title, body, 5 if is_critical else 4, tags)


def send_test():
    return _post("AirBnBrrr test", "ntfy is connected. Real alerts look like this: the worst problems in plain words, then what to do.", 3, "white_check_mark")
