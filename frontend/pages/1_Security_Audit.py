"""
FintechCID -- Security Audit Dashboard (Page 2)
====================================================
Reads Bandit SAST, pip-audit dependency, and simulated Trivy container
scan reports from frontend/data/ and renders them as an interactive
DevSecOps visibility dashboard.

Fail-safe: gracefully handles missing report files.
"""

import json
import pathlib

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

st.set_page_config(
    page_title="Security Audit -- FintechCID",
    page_icon="shield",
    layout="wide",
)

st.title("DevSecOps Security Audit")
st.caption("SAST | Dependency Scan | Container Scan | EY Compliance Visibility")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def load_json(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def severity_colour(sev: str) -> str:
    return {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}.get(
        sev.upper(), "#6b7280"
    )


# ---------------------------------------------------------------------------
# 1. Bandit SAST
# ---------------------------------------------------------------------------
st.header("1. SAST Analysis (Bandit)")

bandit = load_json(DATA_DIR / "bandit_report.json")

if bandit is None:
    st.warning(
        "No Bandit report found. Run the security audit script to generate metrics:  \n"
        "```bash\nbash run_security_audit.sh\n```"
    )
else:
    totals = bandit.get("metrics", {}).get("_totals", {})
    high   = int(totals.get("SEVERITY.HIGH",   0))
    med    = int(totals.get("SEVERITY.MEDIUM",  0))
    low    = int(totals.get("SEVERITY.LOW",     0))
    loc    = int(totals.get("loc",              0))
    nosec  = int(totals.get("nosec",            0))
    results = bandit.get("results", [])

    # -- Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("High Severity",   high,  delta=None)
    c2.metric("Medium Severity", med,   delta=None)
    c3.metric("Low Severity",    low,   delta=None)
    c4.metric("Lines Scanned",   f"{loc:,}")
    c5.metric("Total Findings",  len(results))

    st.divider()

    # -- Severity bar chart
    if high + med + low > 0:
        chart_data = pd.DataFrame({
            "Severity": ["High", "Medium", "Low"],
            "Count":    [high, med, low],
        })
        st.subheader("Findings by Severity")
        st.bar_chart(chart_data.set_index("Severity"), color=["#dc2626"])
    else:
        st.success("No security issues found by Bandit SAST.")

    # -- Findings by file
    if results:
        st.subheader("Findings by File")
        file_counts: dict[str, int] = {}
        for r in results:
            f = r.get("filename", "unknown").split("/")[-1]
            file_counts[f] = file_counts.get(f, 0) + 1
        file_df = pd.DataFrame(
            list(file_counts.items()), columns=["File", "Findings"]
        ).sort_values("Findings", ascending=False)
        st.bar_chart(file_df.set_index("File"))

    # -- Raw findings expander
    with st.expander(f"Raw Findings ({len(results)} issues)", expanded=False):
        if not results:
            st.info("No findings.")
        else:
            for r in results:
                sev   = r.get("issue_severity", "LOW")
                col   = severity_colour(sev)
                fname = r.get("filename", "")
                line  = r.get("line_number", "?")
                text  = r.get("issue_text", "")
                test  = r.get("test_id", "")
                st.markdown(
                    f'<span style="background:{col};color:white;padding:2px 8px;'
                    f'border-radius:4px;font-size:0.8rem;">{sev}</span> '
                    f'**{test}** — `{fname}:{line}`  \n{text}',
                    unsafe_allow_html=True,
                )
                code_snippet = r.get("code", "").strip()
                if code_snippet:
                    st.code(code_snippet, language="python")
                st.divider()

    # -- Generated at
    st.caption(f"Report generated: {bandit.get('generated_at', 'unknown')}")

# ---------------------------------------------------------------------------
# 2. Dependency Vulnerability Scan (pip-audit)
# ---------------------------------------------------------------------------
st.header("2. Dependency Vulnerabilities (pip-audit)")

dep = load_json(DATA_DIR / "dependency_report.json")

if dep is None:
    st.warning("No dependency report found. Run `bash run_security_audit.sh`.")
else:
    all_deps     = dep.get("dependencies", [])
    vuln_pkgs    = [p for p in all_deps if p.get("vulns")]
    total_vulns  = sum(len(p["vulns"]) for p in vuln_pkgs)
    total_scanned = len(all_deps)

    cv1, cv2, cv3 = st.columns(3)
    cv1.metric("Packages Scanned",      total_scanned)
    cv2.metric("Vulnerable Packages",   len(vuln_pkgs),  delta=None)
    cv3.metric("Total CVEs / Advisories", total_vulns,   delta=None)

    if not vuln_pkgs:
        st.success("No known vulnerabilities found in dependencies.")
    else:
        st.warning(f"{total_vulns} known vulnerabilities found across {len(vuln_pkgs)} packages.")
        st.divider()
        for pkg in vuln_pkgs:
            with st.expander(
                f"{pkg['name']} {pkg['version']} — {len(pkg['vulns'])} vulnerability(ies)",
                expanded=True,
            ):
                for v in pkg["vulns"]:
                    vid  = v.get("id", "N/A")
                    fix  = ", ".join(v.get("fix_versions", [])) or "No fix available"
                    desc = v.get("description", "No description.")
                    st.markdown(
                        f'<span style="background:#dc2626;color:white;padding:2px 8px;'
                        f'border-radius:4px;font-size:0.8rem;">CVE</span> '
                        f'**{vid}** — Fix: `{fix}`',
                        unsafe_allow_html=True,
                    )
                    st.caption(desc[:300] + ("..." if len(desc) > 300 else ""))
                    st.divider()

# ---------------------------------------------------------------------------
# 3. Container Scan (Trivy -- simulated locally)
# ---------------------------------------------------------------------------
st.header("3. Container Scan (Trivy)")

trivy = load_json(DATA_DIR / "trivy_report.json")

if trivy is None:
    st.warning("No Trivy report found. Run `bash run_security_audit.sh`.")
else:
    note = trivy.get("_note", "")
    artifact = trivy.get("ArtifactName", "unknown")
    results  = trivy.get("Results", [])

    t1, t2 = st.columns(2)
    t1.metric("Image Scanned", artifact)

    total_trivy_vulns = sum(
        len(r.get("Vulnerabilities") or []) for r in results
    )
    t2.metric("Container Vulnerabilities", total_trivy_vulns)

    if note:
        st.info(f"Note: {note}")

    if total_trivy_vulns == 0:
        st.success("No container vulnerabilities found.")
    else:
        for r in results:
            vulns = r.get("Vulnerabilities") or []
            if vulns:
                with st.expander(f"{r.get('Target')} — {len(vulns)} findings"):
                    st.json(vulns)

# ---------------------------------------------------------------------------
# 4. CI/CD Pipeline Status
# ---------------------------------------------------------------------------
st.header("4. CI/CD Pipeline")
st.markdown(
    "The GitHub Actions pipeline (`.github/workflows/devsecops.yml`) runs on every push:  \n\n"
    "| Step | Tool | Fail Condition |\n"
    "|---|---|---|\n"
    "| Code Lint | Flake8 | Any syntax / style error |\n"
    "| SAST | Bandit | HIGH severity findings |\n"
    "| Container Scan | Trivy | HIGH or CRITICAL CVEs |\n\n"
    "Configure the badge in your README:  \n"
    "```\n![DevSecOps](../../actions/workflows/devsecops.yml/badge.svg)\n```"
)
