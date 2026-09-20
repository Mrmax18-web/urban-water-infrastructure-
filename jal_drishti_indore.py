"""
================================================================================
 JAL-DRISHTI (जल-दृष्टि)  |  Agentic Intelligence for Urban Water Infrastructure
 Indore Municipal Corporation -- prototype operations console
================================================================================
 A single-file bilingual (English / हिंदी) Streamlit control room for a
 hackathon prototype, modelled on Indore's actual water network.

 Run it:
     pip install streamlit plotly pandas
     streamlit run jal_drishti_indore.py

 Tested against streamlit>=1.32, plotly>=5.18, pandas>=2.0

 WHY THE CITY DETAIL MATTERS
 --------------------------------------------------------------------------
 Every zone, source and figure below is real. Indore lifts Narmada water
 534 m from Jalud and carries it ~70 km to the city, which makes its supply
 among the costliest in India -- so a burst main is not just lost water, it
 is lost pumping energy. IMC already runs SCADA over ~102 tanks and takes
 citizen complaints through the Indore 311 app (Open311). This prototype
 sits on top of both: SCADA gives it telemetry, 311 gives it corroboration.
 Swap the mock feeds in Sections 6 and 7 for those two real sources and the
 agent is running on live city data.

 ARCHITECTURE
 --------------------------------------------------------------------------
   SECTION 1  City model + tunables       <- zones, sources, thresholds
   SECTION 2  Bilingual strings           <- every word the UI can say
   SECTION 3  Theme (CSS)                 <- includes Devanagari handling
   SECTION 4  Session state               <- the entire "database"
   SECTION 5  Pictures                    <- live network map + 311 photos
   SECTION 6  LLM / agent backend         <- ** YOUR INTEGRATION POINT **
   SECTION 7  Telemetry simulation        <- replace with IMC SCADA
   SECTION 8  Agent cycle (state machine)
   SECTION 9  UI render functions
   SECTION 10 Main + the 2-second loop

 LANGUAGE NOTE
 --------------------------------------------------------------------------
 The log stores message KEYS and parameters, never rendered sentences. So
 switching to हिंदी mid-demo re-renders the entire reasoning history in
 Hindi, including lines written minutes ago. That is the point: the operator
 in a zonal office and the commissioner reviewing the trace can read the
 same audit trail in the language each prefers.
================================================================================
"""

from __future__ import annotations

import base64
import html as html_lib
import json  # noqa: F401  (kept: you'll want it for LLM payloads)
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# SECTION 1 -- CITY MODEL AND TUNABLE CONSTANTS
# =============================================================================

APP_NAME = "JAL-DRISHTI"
APP_NAME_HI = "जल-दृष्टि"
BUILD_TAG = "v1.2.0-demo"

TICK_SECONDS = 2.0          # one simulated interval
HISTORY_TICKS = 60          # how much trend the chart keeps

# --- Indore district metered areas -------------------------------------------
# Four representative zones out of IMC's 22. Each has a different failure
# personality, which is what makes the agent's prioritisation interesting.
ZONES = {
    "IDZ-02": {"en": "Rajwada", "hi": "राजवाड़ा", "color": "#E8A33D",
               "esr": "Jinsi Hat ESR", "esr_hi": "जिंसी हाट ESR",
               "note_en": "Old city core, 1960s cast-iron mains",
               "note_hi": "पुराना शहर, 1960 के दशक की कास्ट-आयरन लाइनें",
               "xy": (952, 104), "side": "top"},
    "IDZ-07": {"en": "Vijay Nagar", "hi": "विजय नगर", "color": "#2BC6D8",
               "esr": "Vijay Nagar OHT", "esr_hi": "विजय नगर OHT",
               "note_en": "High-rise demand, newest network, farthest from the inlet",
               "note_hi": "बहुमंज़िला मांग, सबसे नया नेटवर्क, इनलेट से सबसे दूर",
               "xy": (1094, 104), "side": "top"},
    "IDZ-13": {"en": "Sudama Nagar", "hi": "सुदामा नगर", "color": "#7C8CF8",
               "esr": "Sudama Nagar OHT", "esr_hi": "सुदामा नगर OHT",
               "note_en": "Dense residential, alternate-day supply",
               "note_hi": "सघन आवासीय, एक दिन छोड़कर आपूर्ति",
               "xy": (812, 236), "side": "bottom"},
    "IDZ-19": {"en": "Rau", "hi": "राऊ", "color": "#3FB68B",
               "esr": "Rau OHT", "esr_hi": "राऊ OHT",
               "note_en": "Western edge, lowest head in the zone set",
               "note_hi": "पश्चिमी छोर, इन ज़ोनों में सबसे कम हेड",
               "xy": (668, 236), "side": "bottom"},
}

# Nominal operating envelope at the zone district meter (PSI and L/s).
# The old city sits lower because its mains are older and the head is worse.
BASE_PRESSURE = {"IDZ-02": 62.0, "IDZ-07": 74.0, "IDZ-13": 68.0, "IDZ-19": 58.0}
BASE_FLOW = {"IDZ-02": 46.0, "IDZ-07": 58.0, "IDZ-13": 51.0, "IDZ-19": 38.0}

PRESSURE_ALARM_PSI = 45.0   # below this, upper floors stop getting water

# The scripted demo failure: a burst main in the old city.
ANOMALY_ZONE = "IDZ-02"
ANOMALY_PRESSURE_DROP = 0.40
ANOMALY_FLOW_SURGE = 1.38   # a burst main gains flow as water escapes
RECOVERY_RATE = 0.07

# --- City facts shown in the header strip (all verified, see docstring) ------
CITY_FACTS = {
    "supply_mld": 450,          # present Narmada supply to the city
    "network_km": 3000,         # distribution network length
    "tanks_on_scada": 102,      # tanks already digitally mapped by IMC
    "wards": 85,
    "zones": 22,
    "lift_m": 534,              # Jalud -> Indore static lift
    "trunk_km": 70,             # Jalud -> Indore distance
    "phase4_mld": 1650,         # Narmada Phase IV intake under construction
}

LLM_BACKENDS = {
    "Llama 3 8B (local)": {"id": "llama3:8b", "runtime": "ollama @ localhost:11434",
                           "latency_ms": (180, 420), "cost": "0.00 / 1k tok"},
    "Gemini 1.5 Flash": {"id": "gemini-1.5-flash", "runtime": "google-generativeai",
                         "latency_ms": (320, 780), "cost": "0.075 / 1M tok"},
    "Claude Sonnet (API)": {"id": "claude-sonnet-4-5", "runtime": "anthropic",
                            "latency_ms": (400, 900), "cost": "3.00 / 1M tok"},
    "Sarvam-M (Indic)": {"id": "sarvam-m", "runtime": "sarvam.ai",
                         "latency_ms": (260, 640), "cost": "Indic-tuned"},
}

# Routine background signals. Key -> (min severity, max severity, photo?).
# `photo` marks the types a citizen would actually photograph and upload to
# Indore 311 -- dirty water gets pictures, a pressure drift does not.
MINOR_ISSUE_TYPES = {
    "pressure_drift": (28, 46, False),
    "turbidity": (40, 58, True),
    "chlorine_drop": (44, 62, True),
    "flow_imbalance": (30, 55, False),
    "acoustic_leak": (41, 62, False),
    "valve_telemetry": (22, 38, False),
}

STATE_VERSION = 11

# Optional real photographs. Drop JPG/PNG files here and they appear in the
# UI automatically; if the folder is missing you get drawn placeholders and
# nothing breaks. See the README note at the bottom of this file.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _st_version() -> tuple:
    """(major, minor) of the installed Streamlit, parsed defensively."""
    out = []
    for chunk in st.__version__.split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


# Streamlit 1.49 renamed use_container_width=True to width="stretch".
WIDE = {"width": "stretch"} if _st_version() >= (1, 49) else {"use_container_width": True}


# =============================================================================
# SECTION 2 -- BILINGUAL STRINGS
# Every user-visible sentence lives here, in both languages, under the same
# key. Nothing else in the file contains display text. To add Marathi or
# Malayalam later, add a third dict -- no other code changes.
# =============================================================================

STRINGS = {
"en": {
    # -- shell -------------------------------------------------------------
    "subtitle": "Agentic intelligence for urban water infrastructure",
    "operator": "Indore Municipal Corporation · water operations",
    "tick": "interval", "uptime": "uptime",
    # -- sidebar -----------------------------------------------------------
    "mission_control": "Mission control",
    "language": "Language",
    "st_gate": "Operator decision needed", "st_gate_sub": "autonomy suspended",
    "st_incident": "Critical incident open", "st_incident_sub": "{zone} degraded",
    "st_nominal": "All zones nominal", "st_nominal_sub": "{n} district meters streaming",
    "st_idle": "Stream idle", "st_idle_sub": "no live telemetry",
    "simulation": "Simulation",
    "start_stream": "Start data stream",
    "start_stream_help": "Pulls a SCADA frame every 2 seconds and runs one agent "
                         "reasoning pass per frame.",
    "trigger": "Trigger critical anomaly",
    "reset": "Reset simulation",
    "agent_config": "Agent configuration",
    "model": "Reasoning model",
    "autonomy": "Autonomy level",
    "autonomy_help": "Semi-autonomous lets the agent act alone on low-impact work "
                     "and escalates anything that interrupts customer supply.",
    "aut_supervised": "Supervised", "aut_semi": "Semi-autonomous", "aut_auto": "Autonomous",
    "temperature": "Temperature",
    "threshold": "Auto-approve below severity",
    "threshold_help": "Issues scoring below this are remediated without asking.",
    "lbl_runtime": "runtime", "lbl_model": "model", "lbl_cost": "cost",
    "lbl_tokens": "tokens", "lbl_resolved": "resolved",
    "decisions": "Operator decisions",
    "approved": "approved", "overridden": "overridden",
    # -- sections ----------------------------------------------------------
    "sec_network": "NETWORK  /  Narmada trunk main to zone meters",
    "sec_perception": "PERCEPTION  /  live zone telemetry",
    "sec_priority": "PRIORITISATION  /  ranked by severity",
    "sec_reasoning": "REASONING  /  agent chain of thought",
    "sec_execution": "EXECUTION  /  work orders and crew dispatch",
    # -- map ---------------------------------------------------------------
    "map_narmada": "Narmada river", "map_jalud": "Jalud intake + pumping",
    "map_lift": "{lift} m lift  ·  {km} km trunk main",
    "map_city": "INDORE", "map_wtp": "Treatment",
    "map_phase": "Phase I–III live  ·  Phase IV {p4} MLD under construction",
    "map_legend_ok": "nominal", "map_legend_watch": "watch", "map_legend_crit": "critical",
    "facts": "{mld} MLD supplied  ·  {km} km network  ·  {tanks} tanks on SCADA  "
             "·  {wards} wards / {zones} zones",
    # -- KPIs --------------------------------------------------------------
    "kpi_pressure": "AVG ZONE PRESSURE", "kpi_flow": "TOTAL FLOW",
    "kpi_anomalies": "ACTIVE ANOMALIES", "kpi_lowest": "LOWEST METER — {zone}",
    "kpi_floor": "{v} PSI to service floor",
    "chart_pressure": "Pressure (PSI)", "chart_flow": "Flow (L/s)",
    "service_floor": " service floor {v} PSI ",
    # -- priority queue ----------------------------------------------------
    "queue_empty": "Queue clear. No open anomalies across the monitored zones.",
    "node": "node", "open_for": "{s}s open", "confidence": "agent confidence",
    "crew": "crew", "photo_311": "Indore 311 photo", "no_photo": "no photo attached",
    # -- log ---------------------------------------------------------------
    "log_empty": "No reasoning steps yet. Start the data stream to wake the agent.",
    # -- kanban ------------------------------------------------------------
    "kb_eval": "Evaluating", "kb_action": "Actioning", "kb_wait": "Awaiting approval",
    "kb_empty": "nothing here", "unassigned": "unassigned",
    # -- escalation gate ---------------------------------------------------
    "gate_banner": "Autonomous execution is paused. The agent needs an operator "
                   "decision before it can continue.",
    "gate_kicker": "HIGH-IMPACT ACTION PROPOSED",
    "gate_action": "Close feeder valve {valve} — {zone}",
    "gate_body": "Closing this valve stops the loss immediately but interrupts "
                 "supply to roughly {households} households for up to four hours, "
                 "in a zone already on alternate-day supply. The agent ranked it "
                 "above the alternative because it minimises total service-hours lost.",
    "gate_issue": "issue", "gate_zone": "zone", "gate_severity": "severity",
    "gate_confidence": "confidence", "gate_raised": "raised",
    "gate_alt": "if overridden",
    "gate_alt_val": "Throttle node {node} only (41% of the loss continues)",
    "approve": "Approve action", "override": "Override",
    "gate_audit": "Either decision is written to the audit trail with a timestamp "
                  "and the full reasoning chain that produced it.",
    "gate_site": "Site photo, {zone}",
    # -- issue types -------------------------------------------------------
    "ty_pressure_drift": "Pressure drift", "ty_turbidity": "Turbidity spike",
    "ty_chlorine_drop": "Chlorine residual drop", "ty_flow_imbalance": "Flow imbalance",
    "ty_acoustic_leak": "Acoustic leak signature", "ty_valve_telemetry": "Valve telemetry loss",
    "ty_suspected_break": "Suspected main break", "ty_main_break": "Main break",
    # -- work order titles -------------------------------------------------
    "wo_verify": "Verify {type} at {node}", "wo_assess": "Assess pressure collapse at {node}",
    # -- log tags ----------------------------------------------------------
    "tag_boot": "boot", "tag_detect": "detect", "tag_correlate": "correlate",
    "tag_diagnose": "diagnose", "tag_plan": "plan", "tag_escalate": "escalate",
    "tag_triage": "triage", "tag_act": "act", "tag_resolve": "resolve",
    "tag_human": "operator", "tag_execute": "execute", "tag_monitor": "monitor",
    "tag_scan": "scan", "tag_model": "model",
    # -- log messages ------------------------------------------------------
    "lg_boot1": "{app} online. {n} zone meters bound to the IMC SCADA feed, "
                "{hist}-interval window.",
    "lg_boot2": "Hydraulic model of the Narmada feeder loaded. Indore 311 "
                "complaint stream connected. Awaiting live telemetry.",
    "lg_detect": "Pressure at the {zone} zone meter fell {drop:.1f}% in one "
                 "interval ({before:.1f} → {after:.1f} PSI). Rate of change exceeds "
                 "the 8%/interval burst threshold.",
    "lg_correlate": "Cross-referenced Indore 311: {complaints} complaints within "
                    "1.4 km of node {node} in the last 20 minutes, {tag} the "
                    "dominant tag. Neighbouring zone meters hold nominal, so this "
                    "is local, not a Jalud supply problem.",
    "lg_diagnose": "Hydraulic model reproduces the observed profile with a {dia} mm "
                   "rupture at node {node}. Flow is up {surge:.0f}% while pressure "
                   "is down: water is leaving the network. Classifying as MAIN "
                   "BREAK. Severity upgraded to critical.",
    "lg_plan": "Two options. (a) Throttle node {node} only: 41% of the loss "
               "continues, est. 380 m³ lost before the crew arrives — roughly "
               "₹{cost} of pumped Narmada water. (b) Close feeder valve {valve}: "
               "stops the loss now, drops supply to ~{households} households for up "
               "to 4 hours. Option (b) minimises total service-hours lost.",
    "lg_escalate": "Option (b) interrupts customer supply → HIGH IMPACT. Autonomy "
                   "policy requires operator authorisation. Holding the action and "
                   "pausing autonomous execution.",
    "lg_dup": "Critical incident already open on {zone}. Ignoring duplicate trigger.",
    "lg_triage": "New signal on {zone} node {node}: {type}, severity {sev}, "
                 "confidence {conf:.2f}. Queued.",
    "lg_autoapprove": "Auto-approved low-impact remediation for {id} on {zone} "
                      "(severity {sev} is below the {threshold} approval threshold).",
    "lg_resolve_minor": "{type} at {node} cleared. {zone} back inside nominal bounds.",
    "lg_human_ok": "Operator approved: {action}. Authorisation logged against {id}.",
    "lg_human_no": "Operator overrode: {action} was not executed.",
    "lg_exec_ok": "Closing {valve} at 15% per minute to avoid a surge transient. "
                  "Crew {crew} dispatched, ETA 14 min. Notifying {households} "
                  "households by SMS and pushing a notice to Indore 311.",
    "lg_exec_no": "Falling back to partial throttle at node {node}. Loss continues "
                  "at ~41%. Crew {crew} dispatched, ETA 14 min. Re-evaluating every "
                  "interval.",
    "lg_monitor": "{zone} recovering: {psi:.1f} PSI ({pct:.0f}% of nominal). Flow "
                  "differential narrowing, consistent with a successful isolation.",
    "lg_hold": "{zone} holding at {psi:.1f} PSI, still {gap:.1f} PSI below the "
               "service floor. Re-proposing valve isolation on the next confidence "
               "update.",
    "lg_resolved": "{zone} restored to {psi:.1f} PSI. Main break at node {node} "
                   "isolated and repaired. Incident closed, full trace written to "
                   "the audit log.",
    "lg_scan1": "Swept {n} zone meters against the SCADA baseline. All within the "
                "nominal envelope.",
    "lg_scan2": "Night-flow baseline for {zone} recalculated: {v:.1f} L/s. No "
                "unexplained minimum-hour demand.",
    "lg_scan3": "Acoustic loggers on {zone} quiet. No leak signature above the "
                "noise floor.",
    "lg_model1": "Demand forecast refreshed. Peak expected {hh}:00, {v:.0f} L/s "
                 "across the four zones.",
    "lg_model2": "Jalud pumping schedule re-optimised against the solar window. "
                 "{v:.1f} kWh shifted off grid tariff this cycle.",
    "cmp_dirty": "'dirty water'", "cmp_nowater": "'no water'",
    "cmp_flood": "'street flooding'",
},
"hi": {
    # -- shell -------------------------------------------------------------
    "subtitle": "शहरी जल अवसंरचना के लिए एजेंटिक इंटेलिजेंस",
    "operator": "नगर पालिक निगम इंदौर · जल संचालन",
    "tick": "अंतराल", "uptime": "अपटाइम",
    # -- sidebar -----------------------------------------------------------
    "mission_control": "नियंत्रण कक्ष",
    "language": "भाषा",
    "st_gate": "संचालक का निर्णय आवश्यक", "st_gate_sub": "स्वायत्त कार्रवाई रोकी गई",
    "st_incident": "गंभीर घटना सक्रिय", "st_incident_sub": "{zone} की स्थिति बिगड़ी",
    "st_nominal": "सभी ज़ोन सामान्य", "st_nominal_sub": "{n} ज़ोन मीटर चालू",
    "st_idle": "डेटा स्ट्रीम बंद", "st_idle_sub": "कोई लाइव टेलीमेट्री नहीं",
    "simulation": "सिमुलेशन",
    "start_stream": "डेटा स्ट्रीम चालू करें",
    "start_stream_help": "हर 2 सेकंड में एक SCADA फ़्रेम लेता है और हर फ़्रेम पर एजेंट "
                         "एक बार तर्क करता है।",
    "trigger": "गंभीर गड़बड़ी शुरू करें",
    "reset": "सिमुलेशन रीसेट करें",
    "agent_config": "एजेंट सेटिंग",
    "model": "तर्क मॉडल",
    "autonomy": "स्वायत्तता स्तर",
    "autonomy_help": "अर्ध-स्वायत्त में एजेंट कम असर वाले काम खुद करता है और उपभोक्ता "
                     "आपूर्ति रोकने वाली हर कार्रवाई संचालक को भेजता है।",
    "aut_supervised": "पर्यवेक्षित", "aut_semi": "अर्ध-स्वायत्त", "aut_auto": "पूर्ण स्वायत्त",
    "temperature": "टेम्परेचर",
    "threshold": "इस गंभीरता से नीचे स्वतः स्वीकृति",
    "threshold_help": "इससे कम अंक वाली समस्याएँ बिना पूछे ठीक की जाती हैं।",
    "lbl_runtime": "रनटाइम", "lbl_model": "मॉडल", "lbl_cost": "लागत",
    "lbl_tokens": "टोकन", "lbl_resolved": "निपटाए",
    "decisions": "संचालक के निर्णय",
    "approved": "स्वीकृत", "overridden": "अस्वीकृत",
    # -- sections ----------------------------------------------------------
    "sec_network": "नेटवर्क  /  नर्मदा मुख्य लाइन से ज़ोन मीटर तक",
    "sec_perception": "अवलोकन  /  लाइव ज़ोन टेलीमेट्री",
    "sec_priority": "प्राथमिकता  /  गंभीरता के क्रम में",
    "sec_reasoning": "तर्क  /  एजेंट की विचार-शृंखला",
    "sec_execution": "निष्पादन  /  कार्य आदेश और टीम भेजना",
    # -- map ---------------------------------------------------------------
    "map_narmada": "नर्मदा नदी", "map_jalud": "जलूद इनटेक + पंपिंग",
    "map_lift": "{lift} मी ऊँचाई  ·  {km} किमी मुख्य लाइन",
    "map_city": "इंदौर", "map_wtp": "शोधन संयंत्र",
    "map_phase": "फेज़ I–III चालू  ·  फेज़ IV {p4} MLD निर्माणाधीन",
    "map_legend_ok": "सामान्य", "map_legend_watch": "निगरानी", "map_legend_crit": "गंभीर",
    "facts": "{mld} MLD आपूर्ति  ·  {km} किमी नेटवर्क  ·  {tanks} टंकियाँ SCADA पर  "
             "·  {wards} वार्ड / {zones} ज़ोन",
    # -- KPIs --------------------------------------------------------------
    "kpi_pressure": "औसत ज़ोन दाब", "kpi_flow": "कुल प्रवाह",
    "kpi_anomalies": "सक्रिय गड़बड़ियाँ", "kpi_lowest": "सबसे कम मीटर — {zone}",
    "kpi_floor": "{v} PSI (सेवा सीमा)",
    "chart_pressure": "दाब (PSI)", "chart_flow": "प्रवाह (L/s)",
    "service_floor": " सेवा सीमा {v} PSI ",
    # -- priority queue ----------------------------------------------------
    "queue_empty": "सूची खाली। निगरानी वाले किसी ज़ोन में कोई खुली गड़बड़ी नहीं।",
    "node": "नोड", "open_for": "{s} सेकंड से", "confidence": "एजेंट विश्वास",
    "crew": "टीम", "photo_311": "इंदौर 311 फ़ोटो", "no_photo": "कोई फ़ोटो नहीं",
    # -- log ---------------------------------------------------------------
    "log_empty": "अभी कोई तर्क नहीं। एजेंट को जगाने के लिए डेटा स्ट्रीम चालू करें।",
    # -- kanban ------------------------------------------------------------
    "kb_eval": "जाँच में", "kb_action": "कार्रवाई में", "kb_wait": "स्वीकृति हेतु",
    "kb_empty": "कुछ नहीं", "unassigned": "अनिर्धारित",
    # -- escalation gate ---------------------------------------------------
    "gate_banner": "स्वायत्त निष्पादन रोका गया है। आगे बढ़ने से पहले एजेंट को संचालक "
                   "के निर्णय की आवश्यकता है।",
    "gate_kicker": "उच्च प्रभाव वाली कार्रवाई प्रस्तावित",
    "gate_action": "फीडर वाल्व {valve} बंद करें — {zone}",
    "gate_body": "यह वाल्व बंद करने से पानी का बहाव तुरंत रुकेगा, पर लगभग {households} "
                 "घरों की आपूर्ति चार घंटे तक बंद रहेगी — वह भी ऐसे ज़ोन में जहाँ पहले से "
                 "एक दिन छोड़कर पानी आता है। एजेंट ने इसे दूसरे विकल्प से ऊपर रखा क्योंकि "
                 "इससे कुल सेवा-घंटों का नुकसान सबसे कम होता है।",
    "gate_issue": "समस्या", "gate_zone": "ज़ोन", "gate_severity": "गंभीरता",
    "gate_confidence": "विश्वास", "gate_raised": "समय",
    "gate_alt": "अस्वीकार करने पर",
    "gate_alt_val": "सिर्फ़ नोड {node} थ्रॉटल (41% बहाव जारी रहेगा)",
    "approve": "कार्रवाई स्वीकृत करें", "override": "अस्वीकार करें",
    "gate_audit": "दोनों में से जो भी निर्णय हो, वह समय और पूरी तर्क-शृंखला के साथ "
                  "ऑडिट रिकॉर्ड में दर्ज होता है।",
    "gate_site": "मौके की फ़ोटो, {zone}",
    # -- issue types -------------------------------------------------------
    "ty_pressure_drift": "दाब में बहाव", "ty_turbidity": "गंदलापन बढ़ा",
    "ty_chlorine_drop": "क्लोरीन स्तर गिरा", "ty_flow_imbalance": "प्रवाह असंतुलन",
    "ty_acoustic_leak": "रिसाव की ध्वनि", "ty_valve_telemetry": "वाल्व सिग्नल बंद",
    "ty_suspected_break": "संभावित मेन लाइन फटना", "ty_main_break": "मेन लाइन फटी",
    # -- work order titles -------------------------------------------------
    "wo_verify": "{node} पर {type} की जाँच करें",
    "wo_assess": "{node} पर दाब गिरने का आकलन करें",
    # -- log tags ----------------------------------------------------------
    "tag_boot": "बूट", "tag_detect": "पहचान", "tag_correlate": "मिलान",
    "tag_diagnose": "निदान", "tag_plan": "योजना", "tag_escalate": "एस्केलेट",
    "tag_triage": "छँटाई", "tag_act": "कार्रवाई", "tag_resolve": "निपटान",
    "tag_human": "संचालक", "tag_execute": "निष्पादन", "tag_monitor": "निगरानी",
    "tag_scan": "स्कैन", "tag_model": "मॉडल",
    # -- log messages ------------------------------------------------------
    "lg_boot1": "{app} चालू। IMC SCADA फ़ीड से {n} ज़ोन मीटर जुड़े, {hist} अंतराल की "
                "विंडो।",
    "lg_boot2": "नर्मदा फीडर का हाइड्रॉलिक मॉडल लोड हुआ। इंदौर 311 शिकायत स्ट्रीम "
                "जुड़ी। लाइव टेलीमेट्री की प्रतीक्षा।",
    "lg_detect": "{zone} ज़ोन मीटर पर दाब एक ही अंतराल में {drop:.1f}% गिरा "
                 "({before:.1f} → {after:.1f} PSI)। बदलाव की दर 8%/अंतराल की बर्स्ट "
                 "सीमा से ऊपर है।",
    "lg_correlate": "इंदौर 311 से मिलान: पिछले 20 मिनट में नोड {node} के 1.4 किमी "
                    "दायरे से {complaints} शिकायतें, सबसे ज़्यादा {tag}। आसपास के ज़ोन "
                    "मीटर सामान्य हैं, यानी गड़बड़ी स्थानीय है, जलूद आपूर्ति की नहीं।",
    "lg_diagnose": "हाइड्रॉलिक मॉडल इस प्रोफ़ाइल को नोड {node} पर {dia} मिमी फटाव से "
                   "मिलाता है। दाब गिर रहा है पर प्रवाह {surge:.0f}% बढ़ा है — पानी "
                   "नेटवर्क से बाहर जा रहा है। वर्गीकरण: मेन लाइन फटी। गंभीरता बढ़ाकर "
                   "‘गंभीर’ की गई।",
    "lg_plan": "दो विकल्प। (क) सिर्फ़ नोड {node} थ्रॉटल करें: 41% बहाव जारी रहेगा, टीम "
               "पहुँचने तक अनुमानित 380 घन मीटर पानी बर्बाद — यानी लगभग ₹{cost} की "
               "पंप की हुई नर्मदा। (ख) फीडर वाल्व {valve} बंद करें: बहाव तुरंत रुकेगा, "
               "पर ~{households} घरों की आपूर्ति 4 घंटे तक बंद। कुल सेवा-घंटों के "
               "हिसाब से (ख) बेहतर है।",
    "lg_escalate": "विकल्प (ख) उपभोक्ता आपूर्ति रोकता है → उच्च प्रभाव। स्वायत्तता नीति "
                   "के अनुसार संचालक की स्वीकृति आवश्यक। कार्रवाई रोकी गई और स्वायत्त "
                   "निष्पादन स्थगित।",
    "lg_dup": "{zone} पर पहले से गंभीर घटना खुली है। दोबारा ट्रिगर अनदेखा किया गया।",
    "lg_triage": "{zone} के नोड {node} पर नया संकेत: {type}, गंभीरता {sev}, विश्वास "
                 "{conf:.2f}। सूची में जोड़ा गया।",
    "lg_autoapprove": "{zone} पर {id} के लिए कम असर वाली कार्रवाई स्वतः स्वीकृत "
                      "(गंभीरता {sev}, स्वीकृति सीमा {threshold} से कम)।",
    "lg_resolve_minor": "{node} पर {type} ठीक हुआ। {zone} फिर सामान्य सीमा में।",
    "lg_human_ok": "संचालक ने स्वीकृति दी: {action}। स्वीकृति {id} के विरुद्ध दर्ज।",
    "lg_human_no": "संचालक ने अस्वीकार किया: {action} नहीं किया गया।",
    "lg_exec_ok": "सर्ज से बचने के लिए {valve} को 15% प्रति मिनट की दर से बंद किया जा "
                  "रहा है। टीम {crew} रवाना, अनुमानित समय 14 मिनट। {households} घरों को "
                  "SMS और इंदौर 311 पर सूचना भेजी जा रही है।",
    "lg_exec_no": "नोड {node} पर आंशिक थ्रॉटल पर लौटा जा रहा है। लगभग 41% बहाव जारी। "
                  "टीम {crew} रवाना, अनुमानित समय 14 मिनट। हर अंतराल पर पुनर्मूल्यांकन।",
    "lg_monitor": "{zone} सुधर रहा है: {psi:.1f} PSI (सामान्य का {pct:.0f}%)। प्रवाह "
                  "का अंतर घट रहा है, यानी वाल्व बंद करना सफल रहा।",
    "lg_hold": "{zone} {psi:.1f} PSI पर टिका है, सेवा सीमा से अब भी {gap:.1f} PSI नीचे। "
               "अगले विश्वास अपडेट पर वाल्व बंद करने का प्रस्ताव दोबारा रखा जाएगा।",
    "lg_resolved": "{zone} {psi:.1f} PSI पर बहाल। नोड {node} की मेन लाइन अलग कर मरम्मत "
                   "पूरी। घटना बंद, पूरा रिकॉर्ड ऑडिट लॉग में दर्ज।",
    "lg_scan1": "SCADA बेसलाइन के विरुद्ध {n} ज़ोन मीटर जाँचे। सभी सामान्य सीमा में।",
    "lg_scan2": "{zone} के लिए रात्रि-प्रवाह बेसलाइन दोबारा निकाली: {v:.1f} L/s। "
                "न्यूनतम-घंटा मांग में कोई अस्पष्ट वृद्धि नहीं।",
    "lg_scan3": "{zone} पर ध्वनि लॉगर शांत। शोर स्तर से ऊपर रिसाव का कोई संकेत नहीं।",
    "lg_model1": "मांग पूर्वानुमान अपडेट। चरम {hh}:00 बजे अनुमानित, चारों ज़ोन मिलाकर "
                 "{v:.0f} L/s।",
    "lg_model2": "जलूद पंपिंग शेड्यूल सौर विंडो के अनुसार फिर से तय किया। इस चक्र में "
                 "{v:.1f} kWh ग्रिड टैरिफ से हटाया गया।",
    "cmp_dirty": "‘गंदा पानी’", "cmp_nowater": "‘पानी नहीं आया’",
    "cmp_flood": "‘सड़क पर पानी’",
},
}


def lang() -> str:
    return st.session_state.get("lang", "en")


def t(key: str, **kw) -> str:
    """Look up a string in the active language, falling back to English.

    Both dictionaries use identical placeholder names, so the same params
    render either language. A missing Hindi key degrades to English rather
    than showing a raw key to a judge.
    """
    table = STRINGS.get(lang(), STRINGS["en"])
    s = table.get(key) or STRINGS["en"].get(key, key)
    try:
        return s.format(**kw) if kw else s
    except (KeyError, IndexError, ValueError):
        return s


def zone_name(tag: str) -> str:
    """'IDZ-02 Rajwada' / 'IDZ-02 राजवाड़ा' -- the ID never translates."""
    return f"{tag} {ZONES[tag][lang()] if lang() in ZONES[tag] else ZONES[tag]['en']}"


def indian_number(n: int) -> str:
    """1234567 -> '12,34,567'. Indian grouping, used in both languages."""
    s = str(int(n))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


# =============================================================================
# SECTION 3 -- THEME
# =============================================================================

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&family=Noto+Sans+Devanagari:wght@400;500;600&display=swap');

:root {
    --void:   #060B12;
    --panel:  #0C1522;
    --panel2: #101E2E;
    --line:   #17293C;
    --text:   #C9D8E6;
    --muted:  #617B92;
    --flow:   #2BC6D8;
    --ok:     #3FB68B;
    --warn:   #E8A33D;
    --crit:   #E5484D;
    /* Devanagari sits after the Latin faces: Latin glyphs render in
       JetBrains Mono / Barlow, Hindi falls through to Noto per-glyph. */
    --mono: 'JetBrains Mono', 'Noto Sans Devanagari', ui-monospace, SFMono-Regular, Menlo, monospace;
    --sans: 'Barlow Semi Condensed', 'Noto Sans Devanagari', 'Inter', system-ui, sans-serif;
}

.stApp {
    background:
        radial-gradient(1100px 520px at 12% -12%, #0F2237 0%, rgba(6,11,18,0) 62%),
        var(--void);
    color: var(--text);
    font-family: var(--sans);
}
.block-container { padding-top: 1.4rem; padding-bottom: 5rem; max-width: 1560px; }
section[data-testid="stSidebar"] > div { background:#080E17; border-right:1px solid var(--line); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
h1,h2,h3,h4 { font-family: var(--sans); color: var(--text); }
p,li,label,span,div { color: var(--text); }

/* ---- masthead ---------------------------------------------------------- */
.masthead { display:flex; align-items:baseline; justify-content:space-between;
    gap:18px; padding:10px 0 12px; border-bottom:1px solid var(--line); margin-bottom:8px; }
.masthead .wordmark { font-family:var(--mono); font-weight:700; font-size:22px; letter-spacing:2.5px; }
.masthead .wordmark b { color: var(--flow); font-weight:700; }
.masthead .tagline { font-size:15px; color: var(--muted); margin-top:2px; }
.masthead .org { font-size:13px; color:#44607A; margin-top:1px; font-family:var(--mono); }
.masthead .clock { font-family:var(--mono); font-size:13px; color:var(--muted); text-align:right; line-height:1.6; }
.masthead .clock b { color: var(--flow); font-weight:500; }
.factstrip { font-family:var(--mono); font-size:11.5px; color:#44607A;
    padding:0 0 16px; border-bottom:1px solid var(--line); margin-bottom:18px; }

/* ---- section headers --------------------------------------------------- */
.sec-head { display:flex; align-items:center; gap:10px; margin:6px 0 12px; }
.sec-head .rule { flex:1; height:1px; background:var(--line); }
.sec-head .txt { font-family:var(--mono); font-size:12.5px; color:var(--muted); letter-spacing:1.2px; }
.sec-head .num { font-family:var(--mono); font-size:11px; color:var(--flow);
    border:1px solid var(--line); border-radius:3px; padding:1px 6px; }

/* ---- metrics ----------------------------------------------------------- */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--panel2) 0%, var(--panel) 100%);
    border:1px solid var(--line); border-left:2px solid var(--flow);
    border-radius:4px; padding:14px 16px 12px; }
div[data-testid="stMetricLabel"] p { font-family:var(--mono) !important; font-size:11.5px !important;
    color:var(--muted) !important; letter-spacing:.6px; }
div[data-testid="stMetricValue"] { font-family:var(--mono) !important; font-weight:700;
    color:var(--text) !important; font-size:31px !important; }
div[data-testid="stMetricDelta"] { font-family:var(--mono) !important; font-size:12.5px !important; }

.panel { background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:14px 16px; }

/* ---- network map ------------------------------------------------------- */
.mapwrap { background:linear-gradient(180deg,#081320 0%, #060D16 100%);
    border:1px solid var(--line); border-radius:4px; padding:6px 8px; }
.mapwrap svg { width:100%; height:auto; display:block; }
.mapwrap text { font-family: var(--sans); }
.mapwrap text.mono { font-family: var(--mono); }
@keyframes flowdash { to { stroke-dashoffset: -44; } }
.trunkflow { animation: flowdash 1.6s linear infinite; }
@keyframes nodepulse { 0%,100% { r:9; opacity:1 } 50% { r:15; opacity:.35 } }
.node-alarm { animation: nodepulse 1.4s ease-in-out infinite; }

/* ---- priority queue ---------------------------------------------------- */
.pq-card { background:linear-gradient(90deg,var(--panel2) 0%, var(--panel) 45%);
    border:1px solid var(--line); border-left:3px solid var(--muted);
    border-radius:4px; padding:11px 13px; margin-bottom:9px; display:flex; gap:12px; }
.pq-card.sev-crit { border-left-color:var(--crit); }
.pq-card.sev-high { border-left-color:var(--warn); }
.pq-card.sev-med  { border-left-color:var(--flow); }
.pq-card.sev-low  { border-left-color:var(--muted); }
.pq-main { flex:1; min-width:0; }
.pq-top { display:flex; justify-content:space-between; align-items:baseline; gap:12px; }
.pq-type { font-size:17px; font-weight:600; }
.pq-score { font-family:var(--mono); font-size:19px; font-weight:700; }
.sev-crit .pq-score { color:var(--crit); }
.sev-high .pq-score { color:var(--warn); }
.sev-med  .pq-score { color:var(--flow); }
.sev-low  .pq-score { color:var(--muted); }
.pq-meta { font-family:var(--mono); font-size:11.5px; color:var(--muted); margin:6px 0 8px; }
.pq-meta span + span { border-left:1px solid var(--line); margin-left:9px; padding-left:9px; }
.pq-bar { height:3px; background:#0A1220; border-radius:2px; overflow:hidden; }
.pq-bar i { display:block; height:100%; }
.sev-crit .pq-bar i { background:var(--crit); }
.sev-high .pq-bar i { background:var(--warn); }
.sev-med  .pq-bar i { background:var(--flow); }
.sev-low  .pq-bar i { background:var(--muted); }
.pq-foot { display:flex; justify-content:space-between; font-family:var(--mono);
    font-size:11px; color:var(--muted); margin-top:7px; }
.pq-foot b { color:var(--text); font-weight:500; }
/* citizen photo attached to a 311 complaint */
.pq-photo { flex:0 0 74px; }
.pq-photo img, .pq-photo .ph {
    width:74px; height:74px; object-fit:cover; border-radius:3px;
    border:1px solid var(--line); display:block; }
.pq-photo .ph { background:#0A1220; display:flex; align-items:center; justify-content:center; }
.pq-photo .cap { font-family:var(--mono); font-size:9px; color:#3E5468;
    margin-top:4px; text-align:center; line-height:1.3; }

/* ---- chain-of-thought log ---------------------------------------------- */
.cot-shell { background:#050A11; border:1px solid var(--line); border-radius:4px;
    height:430px; overflow-y:auto; padding:10px 4px 10px 12px; }
.cot-shell::-webkit-scrollbar { width:7px; }
.cot-shell::-webkit-scrollbar-track { background:#050A11; }
.cot-shell::-webkit-scrollbar-thumb { background:var(--line); border-radius:4px; }
.log-row { font-family:var(--mono); font-size:12px; line-height:1.55; display:flex;
    gap:9px; padding:3px 0; border-bottom:1px solid rgba(23,41,60,.45); }
.log-ts  { color:#3E5468; flex:0 0 84px; }
.log-tag { flex:0 0 92px; color:var(--muted); }
.log-msg { color:#A9BED2; flex:1; }
.lvl-ok   .log-tag, .lvl-ok   .log-msg { color:var(--ok); }
.lvl-warn .log-tag, .lvl-warn .log-msg { color:var(--warn); }
.lvl-crit .log-tag, .lvl-crit .log-msg { color:var(--crit); font-weight:500; }
.lvl-act  .log-tag, .lvl-act  .log-msg { color:var(--flow); }
.log-empty { font-family:var(--mono); font-size:12px; color:var(--muted); padding:10px; }

/* ---- kanban ------------------------------------------------------------ */
.kb-col { background:#08101B; border:1px solid var(--line); border-radius:4px;
    padding:10px 10px 4px; min-height:190px; }
.kb-col-head { display:flex; justify-content:space-between; align-items:center;
    font-family:var(--mono); font-size:11.5px; color:var(--muted); letter-spacing:.6px;
    padding-bottom:8px; margin-bottom:9px; border-bottom:1px solid var(--line); }
.kb-col-head i { font-style:normal; color:var(--text); }
.kb-card { background:var(--panel2); border:1px solid var(--line); border-radius:3px;
    padding:9px 11px; margin-bottom:8px; }
.kb-card.hot { border-color:rgba(229,72,77,.55); background:#1A1116; }
.kb-id { font-family:var(--mono); font-size:10.5px; color:var(--flow); }
.kb-title { font-size:15px; margin:3px 0 5px; line-height:1.3; }
.kb-sub { font-family:var(--mono); font-size:10.5px; color:var(--muted); }
.kb-sub span + span { border-left:1px solid var(--line); margin-left:8px; padding-left:8px; }
.kb-empty { font-family:var(--mono); font-size:11px; color:#35485C; padding:16px 4px; text-align:center; }

/* ---- human escalation gateway ------------------------------------------ */
.gate { border:1px solid rgba(229,72,77,.6); border-left:4px solid var(--crit);
    background:linear-gradient(90deg, rgba(229,72,77,.14) 0%, rgba(229,72,77,.03) 55%, var(--panel) 100%);
    border-radius:4px; padding:16px 18px; margin-bottom:12px;
    display:flex; gap:20px; align-items:flex-start;
    animation: gate-breathe 2.6s ease-in-out infinite; }
@keyframes gate-breathe {
    0%,100% { box-shadow:0 0 0 0 rgba(229,72,77,.30); }
    50%     { box-shadow:0 0 0 9px rgba(229,72,77,0); } }
.gate-main { flex:1; min-width:0; }
.gate-kicker { font-family:var(--mono); font-size:11.5px; color:var(--crit); letter-spacing:1.2px; }
.gate-title { font-size:25px; font-weight:600; margin:5px 0 8px; line-height:1.25; }
.gate-body { font-size:15px; color:#B6C7D8; max-width:74ch; line-height:1.55; }
.gate-grid { display:flex; gap:24px; flex-wrap:wrap; margin-top:12px; }
.gate-grid div { font-family:var(--mono); font-size:11.5px; color:var(--muted); }
.gate-grid b { display:block; color:var(--text); font-size:14px; font-weight:500; margin-top:3px; }
.gate-photo { flex:0 0 150px; }
.gate-photo img, .gate-photo .ph { width:150px; height:112px; object-fit:cover;
    border-radius:3px; border:1px solid rgba(229,72,77,.45); display:block; }
.gate-photo .ph { background:#150D11; display:flex; align-items:center; justify-content:center; }
.gate-photo .cap { font-family:var(--mono); font-size:9.5px; color:#8A6A70;
    margin-top:5px; text-align:center; }

/* ---- sidebar ----------------------------------------------------------- */
.status-chip { display:flex; align-items:center; gap:10px; background:var(--panel);
    border:1px solid var(--line); border-radius:4px; padding:11px 13px; margin-bottom:6px; }
.lamp { width:11px; height:11px; border-radius:50%; flex:0 0 11px; }
.lamp.green { background:var(--ok);   animation:pulse-ok 2s infinite; }
.lamp.amber { background:var(--warn); animation:pulse-warn 1.2s infinite; }
.lamp.red   { background:var(--crit); animation:pulse-crit .8s infinite; }
.lamp.grey  { background:#2E4055; }
@keyframes pulse-ok   { 0%,100%{box-shadow:0 0 0 0 rgba(63,182,139,.55);} 50%{box-shadow:0 0 0 7px rgba(63,182,139,0);} }
@keyframes pulse-warn { 0%,100%{box-shadow:0 0 0 0 rgba(232,163,61,.6);}  50%{box-shadow:0 0 0 7px rgba(232,163,61,0);} }
@keyframes pulse-crit { 0%,100%{box-shadow:0 0 0 0 rgba(229,72,77,.7);}   50%{box-shadow:0 0 0 8px rgba(229,72,77,0);} }
.status-txt { font-family:var(--mono); font-size:12.5px; line-height:1.4; }
.status-txt i { font-style:normal; display:block; font-size:10.5px; color:var(--muted); }
.side-note { font-family:var(--mono); font-size:10.5px; color:var(--muted); line-height:1.75; }
.side-note b { color:var(--text); font-weight:500; }

/* ---- widgets ----------------------------------------------------------- */
.stSelectbox div[data-baseweb="select"] > div, .stTextInput input {
    background:var(--panel) !important; border-color:var(--line) !important;
    font-family:var(--mono) !important; font-size:13px !important; }
.stButton > button { font-family:var(--mono); font-size:12.5px; border-radius:3px;
    border:1px solid var(--line); background:var(--panel2); color:var(--text);
    transition:border-color .15s ease, background .15s ease; }
.stButton > button:hover { border-color:var(--flow); color:#fff; }
.stButton > button[kind="primary"] { background:#123240; border-color:var(--flow); color:#EAF9FC; }
.stButton > button[kind="primary"]:hover { background:#17414F; }
.stButton > button:focus-visible { outline:2px solid var(--flow); outline-offset:2px; }
div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] { font-family:var(--mono); }
hr { border-color:var(--line); }

/* Streamlit paints sliders, radios and toggles with its own primary red,
   which fights the palette. Those controls are built from unnamed emotion
   classes, so the stable way to recolour them is to rotate the hue of the
   control itself -- scoped tightly enough that no label text is caught. */
div[data-testid="stSlider"] div[role="group"],
label[data-testid="stRadioOption"] > div > div:first-child,
div[data-testid="stCheckbox"] > label > div:not([data-testid]) {
    filter: hue-rotate(168deg) saturate(.92); }

/* Streamlit chrome that has no place in a control room. */
div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display:none !important; }

/* Streamlit dims the page while a rerun is in flight. At a 2-second cadence
   that reads as a flicker across the whole dashboard, so hold it at full
   opacity and let the numbers just change. */
div[data-stale="true"], .element-container[data-stale="true"] { opacity:1 !important; }

@media (prefers-reduced-motion: reduce) {
    .lamp, .gate, .trunkflow, .node-alarm { animation:none !important; }
}
</style>
"""

# Devanagari needs more vertical room for its matras, and Hindi words are
# longer than their English labels, so the fixed columns widen in Hindi only.
HINDI_CSS = """
<style>
/* Devanagari needs more vertical room for its matras, and Hindi labels run
   longer than their English equivalents, so the fixed columns widen here.

   The font order also has to flip. In a fallback chain the SPACE glyph comes
   from the first font that has one -- JetBrains Mono, whose space is a full
   monospace advance -- which left Hindi words looking double-spaced. Putting
   Noto first fixes the spacing; the few elements that are pure digits and
   need to stay in a column get JetBrains back explicitly. */
:root {
    --mono: 'Noto Sans Devanagari', 'JetBrains Mono', ui-monospace, monospace;
    --sans: 'Noto Sans Devanagari', 'Barlow Semi Condensed', system-ui, sans-serif;
}
.log-ts, .pq-score, div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important; }
.log-row { line-height:1.85; font-size:12.5px; }
.log-tag { flex:0 0 112px; }
.pq-meta, .kb-sub, .side-note, .status-txt, .gate-grid div { line-height:1.9; }
.pq-type, .kb-title, .gate-title { line-height:1.45; }
.gate-body { line-height:1.8; }
.sec-head .txt { letter-spacing:0; font-size:13px; }
div[data-testid="stMetricLabel"] p { letter-spacing:0; font-size:12.5px !important; }
</style>
"""


def html(markup: str) -> None:
    """Render raw HTML.

    Newlines are collapsed so Streamlit's markdown parser can't slip stray <p>
    tags inside our layout divs. Lines are joined with a single space, not
    nothing: joining bare would weld the last word of one wrapped sentence
    onto the first word of the next. Every element that sits inline next to a
    sibling here is in a flex container, so the extra space collapses away.
    """
    st.markdown(" ".join(line.strip() for line in markup.splitlines()),
                unsafe_allow_html=True)


def esc(value) -> str:
    return html_lib.escape(str(value))


# =============================================================================
# SECTION 4 -- SESSION STATE
# =============================================================================

def init_state() -> None:
    ss = st.session_state
    if ss.get("_v") == STATE_VERSION:
        return
    keep_lang = ss.get("lang", "en")
    ss.clear()
    ss._v = STATE_VERSION
    ss.lang = keep_lang           # a reset must not throw away the language

    ss.tick = 0
    ss.booted_at = datetime.now()
    ss.telemetry = deque(maxlen=HISTORY_TICKS * len(ZONES))
    ss.pressure_scale = {z: 1.0 for z in ZONES}
    ss.flow_scale = {z: 1.0 for z in ZONES}
    ss.prev_kpi = {"pressure": None, "flow": None, "anomalies": 0}

    ss.cot_log = deque(maxlen=220)   # newest entry at index 0
    ss.queue = []
    ss.work_orders = []
    ss.escalation = None
    ss.incident = None
    ss.decisions = []
    ss.resolved_count = 0
    ss.tokens_used = 0
    ss.repair_active = False

    ss.streaming = False
    ss.pending_trigger = False

    for _ in range(26):
        sample_telemetry(advance_clock=True, silent=True)

    seed_background_issues()
    ss.prev_kpi["anomalies"] = len(ss.queue)
    log("boot", "lg_boot1", app=APP_NAME, n=len(ZONES), hist=HISTORY_TICKS, level="ok")
    log("boot", "lg_boot2")


def next_id(prefix: str) -> str:
    return f"{prefix}-{random.randint(1000, 9999)}"


# =============================================================================
# SECTION 5 -- PICTURES
# Two kinds, both of which do a job:
#   1. A live schematic of the Narmada feeder and the four zones, redrawn
#      every interval so zone colour tracks real pressure.
#   2. Citizen photographs. Indore 311 complaints carry geotagged photos, so
#      the agent attaches them to the issue it raised -- an operator should
#      see the flooded street before authorising a shutdown.
# Real photos are optional: drop files into ./assets and they appear.
# =============================================================================

_ASSET_CACHE: dict[str, str | None] = {}


def asset_uri(*names: str) -> str | None:
    """First matching file under ./assets as a data URI, or None.

    Data URIs are used because Streamlit's markdown cannot reference local
    file paths. Missing folder, missing file and unreadable file all return
    None, and every caller has a drawn fallback.
    """
    key = "|".join(names)
    if key in _ASSET_CACHE:
        return _ASSET_CACHE[key]
    result = None
    if ASSETS_DIR.is_dir():
        for name in names:
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                path = ASSETS_DIR / f"{name}{ext}"
                if path.is_file():
                    try:
                        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext[1:]}"
                        data = base64.b64encode(path.read_bytes()).decode("ascii")
                        result = f"data:{mime};base64,{data}"
                    except OSError:
                        result = None
                    break
            if result:
                break
    _ASSET_CACHE[key] = result
    return result


CAMERA_GLYPH = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
                'stroke="#2E4055" stroke-width="1.5"><path d="M3 8h3l1.5-2h9L18 8h3v12H3z"/>'
                '<circle cx="12" cy="13.5" r="3.5"/></svg>')


def photo_block(kind: str, zone_tag: str, caption: str) -> str:
    """<img> when a real photo exists, drawn placeholder when it doesn't."""
    uri = asset_uri(f"311/{zone_tag.lower()}-{kind}", f"311/{kind}", f"zones/{zone_tag.lower()}")
    inner = (f'<img src="{uri}" alt="">' if uri
             else f'<div class="ph">{CAMERA_GLYPH}</div>')
    cap = caption if uri else t("no_photo")
    return f'{inner}<div class="cap">{esc(cap)}</div>'


def render_city_map() -> None:
    """Live schematic: Narmada at Jalud, the 534 m lift, and the four zones.

    This is the one picture that explains why Indore's water is worth an
    agent watching it -- every litre here has been lifted half a kilometre
    and carried 70 km before it reaches a tap, so a burst main spills
    pumping energy as well as water.

    Laid out as a real schematic would be: one trunk main across the city
    with a tee down to each zone, rather than feeders fanning out from a
    single point. Nothing overlaps at any zoom, which matters on a
    projector.
    """
    readings = latest_readings()
    trunk_y = 168

    def zone_colour(tag: str) -> str:
        psi = readings.get(tag, {}).get("pressure", BASE_PRESSURE[tag])
        if psi < PRESSURE_ALARM_PSI:
            return "#E5484D"
        if psi < BASE_PRESSURE[tag] * 0.88:
            return "#E8A33D"
        return "#3FB68B"

    parts = []
    for tag, meta in ZONES.items():
        x, y = meta["xy"]
        colour = zone_colour(tag)
        top = meta["side"] == "top"
        name_y = y - 20 if top else y + 26
        esr_y = y + 26 if top else y + 42
        ring = (f'<circle cx="{x}" cy="{y}" r="9" fill="none" stroke="{colour}" '
                f'stroke-width="2" class="node-alarm"/>') if colour == "#E5484D" else ""
        psi = readings.get(tag, {}).get("pressure", BASE_PRESSURE[tag])
        note = meta["note_hi"] if lang() == "hi" else meta["note_en"]
        parts.append(f"""
          <g>
            <title>{esc(zone_name(tag))} — {esc(note)} — {psi:.1f} PSI</title>
            <line x1="{x}" y1="{trunk_y}" x2="{x}" y2="{y}" stroke="#1B3348" stroke-width="2"/>
            <circle cx="{x}" cy="{trunk_y}" r="3.5" fill="#2BC6D8" opacity=".8"/>
            {ring}
            <circle cx="{x}" cy="{y}" r="7" fill="{colour}"/>
            <text x="{x}" y="{name_y}" fill="#C9D8E6" font-size="13"
                  text-anchor="middle">{esc(zone_name(tag))}</text>
            <text x="{x}" y="{esr_y}" fill="#46617A" font-size="10.5" class="mono"
                  text-anchor="middle">{esc(meta["esr_hi"] if lang() == "hi" else meta["esr"])}</text>
          </g>""")

    svg = f"""
    <div class="mapwrap">
    <svg viewBox="0 0 1200 320" xmlns="http://www.w3.org/2000/svg" role="img"
         aria-label="{esc(t('sec_network'))}">
      <!-- legend + programme status, clear of the city boundary below -->
      <circle cx="26" cy="26" r="5" fill="#3FB68B"/>
      <text x="38" y="30" fill="#46617A" font-size="11" class="mono">{esc(t('map_legend_ok'))}</text>
      <circle cx="130" cy="26" r="5" fill="#E8A33D"/>
      <text x="142" y="30" fill="#46617A" font-size="11" class="mono">{esc(t('map_legend_watch'))}</text>
      <circle cx="238" cy="26" r="5" fill="#E5484D"/>
      <text x="250" y="30" fill="#46617A" font-size="11" class="mono">{esc(t('map_legend_crit'))}</text>
      <text x="1178" y="30" fill="#3B5570" font-size="11" class="mono" text-anchor="end">
        {esc(t('map_phase', p4=indian_number(CITY_FACTS['phase4_mld'])))}</text>

      <!-- Narmada at Jalud -->
      <path d="M22 258 C 62 246, 92 270, 130 258 C 164 247, 186 264, 208 256"
            fill="none" stroke="#1E4E63" stroke-width="7" stroke-linecap="round"/>
      <text x="24" y="288" fill="#46617A" font-size="11.5" class="mono">{esc(t('map_narmada'))}</text>
      <rect x="214" y="236" width="48" height="30" rx="3" fill="#0E2030" stroke="#25455C"/>
      <circle cx="228" cy="251" r="4.5" fill="none" stroke="#2BC6D8" stroke-width="1.6"/>
      <circle cx="248" cy="251" r="4.5" fill="none" stroke="#2BC6D8" stroke-width="1.6"/>
      <text x="206" y="228" fill="#8FA7BD" font-size="12" class="mono">{esc(t('map_jalud'))}</text>

      <!-- the lift: rising main, Jalud -> treatment -> city -->
      <path d="M262 251 C 350 246, 372 214, 424 205" fill="none" stroke="#143246"
            stroke-width="9" stroke-linecap="round"/>
      <path d="M262 251 C 350 246, 372 214, 424 205" fill="none" stroke="#2BC6D8"
            stroke-width="3" stroke-linecap="round" stroke-dasharray="14 30"
            class="trunkflow" opacity=".85"/>
      <text x="276" y="182" fill="#2BC6D8" font-size="11.5" class="mono">
        {esc(t('map_lift', lift=CITY_FACTS['lift_m'], km=CITY_FACTS['trunk_km']))}</text>
      <rect x="424" y="187" width="36" height="36" rx="3" fill="#0E2030" stroke="#25455C"/>
      <text x="442" y="244" fill="#46617A" font-size="10.5" class="mono"
            text-anchor="middle">{esc(t('map_wtp'))}</text>
      <path d="M460 205 C 520 200, 552 180, 596 {trunk_y}" fill="none" stroke="#143246"
            stroke-width="9" stroke-linecap="round"/>
      <path d="M460 205 C 520 200, 552 180, 596 {trunk_y}" fill="none" stroke="#2BC6D8"
            stroke-width="3" stroke-linecap="round" stroke-dasharray="14 30"
            class="trunkflow" opacity=".85"/>

      <!-- city boundary and the trunk main running through it -->
      <rect x="596" y="52" width="584" height="240" rx="10" fill="#070F1A"
            stroke="#12243A" stroke-width="1"/>
      <!-- letter-spacing breaks Devanagari conjuncts and matras apart, so the
           tracked-out treatment is applied to the Latin wordmark only -->
      <text x="614" y="78" fill="#3E5A76" font-size="15"
            letter-spacing="{0 if lang() == 'hi' else 4}">{esc(t('map_city'))}</text>
      <path d="M596 {trunk_y} L 1156 {trunk_y}" fill="none" stroke="#143246"
            stroke-width="7" stroke-linecap="round"/>
      <path d="M596 {trunk_y} L 1156 {trunk_y}" fill="none" stroke="#2BC6D8"
            stroke-width="2.5" stroke-dasharray="14 30" class="trunkflow" opacity=".7"/>
      {''.join(parts)}
    </svg>
    </div>
    """
    html(svg)


# =============================================================================
# SECTION 6 -- LLM / AGENT BACKEND LAYER
# =============================================================================
# >>>>>>>>>>>>>>>>>>>>  THIS IS YOUR INTEGRATION POINT  <<<<<<<<<<<<<<<<<<<<<<<
#
# Everything the agent "says" flows through AgentLLM.think(). The mock returns
# scripted keys so the demo is deterministic and works with no network. To go
# live, replace the body of _call_backend() and keep the return contract:
#
#   {"key": str | None, "text": str | None, "severity": int, "confidence": float,
#    "action": str | None, "level": "info"|"ok"|"warn"|"crit"|"act"}
#
# `key` means "look this sentence up in STRINGS and render it in whichever
# language the operator has selected". A real model returns `text` instead --
# free-form prose in the language you asked for. The log renderer handles
# both, so you can migrate one stage at a time.
# =============================================================================

SYSTEM_PROMPT_EN = """You are the reasoning core of the water network agent for
Indore Municipal Corporation. Narmada water is lifted 534 m at Jalud and carried
70 km to the city, so lost water is also lost pumping energy: weigh that in
every recommendation. You receive zone district-meter telemetry from IMC SCADA,
the hydraulic model of the Narmada feeder, and citizen complaints from the
Indore 311 app (Open311).

For each observation you must:
  1. state what changed, with the number that changed;
  2. name the evidence you cross-referenced;
  3. give a severity 0-100 and a calibrated confidence 0-1;
  4. propose exactly one next action, or null if none is warranted.
Any action that interrupts supply to a customer is HIGH IMPACT and must be
returned with "requires_approval": true. Many zones are already on
alternate-day supply, so treat a supply interruption as expensive.
Respond with JSON only. No prose, no markdown fences."""

# Appended when the operator has selected Hindi. Keep identifiers (IDZ-02,
# J-4417, RV-02, PSI, MLD) in Latin script -- that is how they appear on the
# asset itself, and a crew reading a work order needs to match the tag.
SYSTEM_PROMPT_HI_SUFFIX = """
Write the "thought" field in Hindi (Devanagari), in the plain register a
municipal engineer would use. Keep asset tags, node IDs, valve IDs and units
(PSI, MLD, L/s, m³) in Latin script. All other JSON fields stay as specified."""


class AgentLLM:
    """Thin adapter so the UI never cares which model is behind it."""

    def __init__(self, backend_label: str, temperature: float, language: str):
        self.label = backend_label
        self.cfg = LLM_BACKENDS[backend_label]
        self.model_id = self.cfg["id"]
        self.temperature = temperature
        self.language = language

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_EN + (SYSTEM_PROMPT_HI_SUFFIX if self.language == "hi" else "")

    def think(self, observation: dict, stage: str) -> dict:
        return self._call_backend(observation, stage)

    def _call_backend(self, observation: dict, stage: str) -> dict:
        # ------------------------------------------------------------------
        # MOCK IMPLEMENTATION -- delete and drop in one of the snippets below.
        # ------------------------------------------------------------------
        lo, hi = self.cfg["latency_ms"]
        st.session_state.tokens_used += random.randint(180, 520)
        _ = random.randint(lo, hi)   # pretend latency; don't block the UI
        return scripted_thought(observation, stage)

        # ------------------------------------------------------------------
        # OPTION A -- Llama 3 8B via a local Ollama server. Runs offline,
        # which matters if the venue wifi dies mid-demo.
        # ------------------------------------------------------------------
        # import requests
        # r = requests.post(
        #     "http://localhost:11434/api/chat",
        #     json={
        #         "model": self.model_id,
        #         "format": "json",
        #         "stream": False,
        #         "options": {"temperature": self.temperature},
        #         "messages": [
        #             {"role": "system", "content": self.system_prompt()},
        #             {"role": "user", "content": json.dumps(
        #                 {"stage": stage, "observation": observation},
        #                 ensure_ascii=False)},
        #         ],
        #     },
        #     timeout=30,
        # )
        # return self._parse(r.json()["message"]["content"])

        # ------------------------------------------------------------------
        # OPTION B -- Gemini 1.5 Flash
        # ------------------------------------------------------------------
        # import google.generativeai as genai
        # genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # model = genai.GenerativeModel(self.model_id,
        #                               system_instruction=self.system_prompt())
        # resp = model.generate_content(
        #     json.dumps({"stage": stage, "observation": observation},
        #                ensure_ascii=False),
        #     generation_config={"temperature": self.temperature,
        #                        "response_mime_type": "application/json"})
        # return self._parse(resp.text)
        #
        # NOTE ON HINDI: small quantised models often drift back to English
        # halfway through a Devanagari response. If that happens, generate in
        # English and translate the "thought" field in a second cheap call, or
        # use an Indic-tuned model. The UI does not care which you choose --
        # it only needs the "text" field populated.

    @staticmethod
    def _parse(raw: str) -> dict:
        """Defensive JSON parse -- small models still emit ``` fences."""
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return {"key": None, "text": f"Malformed model output: {cleaned[:140]}",
                    "severity": 0, "confidence": 0.0, "action": None, "level": "warn"}
        return {
            "key": None,
            "text": data.get("thought", ""),
            "severity": int(data.get("severity", 0)),
            "confidence": float(data.get("confidence", 0.0)),
            "action": data.get("action"),
            "level": data.get("level", "info"),
        }


def get_agent() -> AgentLLM:
    return AgentLLM(
        backend_label=st.session_state.get("llm_backend", list(LLM_BACKENDS)[0]),
        temperature=st.session_state.get("temperature", 0.2),
        language=lang(),
    )


# Scripted incident stages: (stage tag, severity, confidence, log level).
# The sentence itself lives in STRINGS under "lg_<stage>", in both languages.
INCIDENT_SCRIPT = [
    ("detect",    58, 0.62, "warn"),
    ("correlate", 71, 0.78, "warn"),
    ("diagnose",  96, 0.93, "crit"),
    ("plan",      96, 0.95, "act"),
    ("escalate",  96, 0.95, "crit"),
]
STAGE_INDEX = {s[0]: i for i, s in enumerate(INCIDENT_SCRIPT)}


def scripted_thought(obs: dict, stage: str) -> dict:
    """Deterministic stand-in for model output. Same contract as a real call."""
    if stage in STAGE_INDEX:
        _, sev, conf, level = INCIDENT_SCRIPT[STAGE_INDEX[stage]]
        return {"key": f"lg_{stage}", "text": None, "severity": sev,
                "confidence": conf, "action": obs.get("proposed_action"),
                "level": level}
    return {"key": None, "text": "", "severity": 0, "confidence": 0.5,
            "action": None, "level": "info"}


IDLE_CHATTER = [
    ("scan", "lg_scan1", "info"),
    ("scan", "lg_scan2", "info"),
    ("scan", "lg_scan3", "info"),
    ("model", "lg_model1", "info"),
    ("model", "lg_model2", "ok"),
]


# =============================================================================
# SECTION 7 -- TELEMETRY SIMULATION
# Replace with the IMC SCADA historian. Everything downstream only reads
# st.session_state.telemetry, so nothing else needs to change.
# =============================================================================

def sample_telemetry(advance_clock: bool = True, silent: bool = False) -> None:
    ss = st.session_state
    if advance_clock:
        ss.tick += 1
    stamp = datetime.now()
    for tag in ZONES:
        pressure = BASE_PRESSURE[tag] * ss.pressure_scale[tag] + random.gauss(0, 0.55)
        flow = BASE_FLOW[tag] * ss.flow_scale[tag] + random.gauss(0, 0.9)
        ss.telemetry.append({"ts": stamp, "tick": ss.tick, "zone": tag,
                             "pressure": round(max(pressure, 3.0), 2),
                             "flow": round(max(flow, 0.5), 2)})
    if not silent:
        recover_pressure()


def recover_pressure() -> None:
    """Walk a faulted zone back to nominal once the fix has been authorised."""
    ss = st.session_state
    for tag in ZONES:
        if ss.pressure_scale[tag] < 1.0 and ss.repair_active:
            ss.pressure_scale[tag] = min(1.0, ss.pressure_scale[tag] + RECOVERY_RATE)
            ss.flow_scale[tag] = max(1.0, ss.flow_scale[tag] - RECOVERY_RATE * 0.9)


def telemetry_df() -> pd.DataFrame:
    return pd.DataFrame(list(st.session_state.telemetry))


def latest_readings() -> dict:
    df = telemetry_df()
    if df.empty:
        return {}
    last = df.sort_values("tick").groupby("zone").tail(1)
    return {row.zone: {"pressure": row.pressure, "flow": row.flow}
            for row in last.itertuples()}


# =============================================================================
# SECTION 8 -- AGENT CYCLE
# =============================================================================

def log(stage_tag: str, key: str | None, level: str = "info",
        text: str | None = None, **params) -> None:
    """Append a reasoning step.

    The message is stored as a KEY plus parameters, never as a finished
    sentence, so the whole history re-renders when the language changes.
    A live model returns prose instead: pass it as `text`.
    """
    st.session_state.cot_log.appendleft({
        "ts": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "tag": stage_tag, "key": key, "text": text, "params": params, "level": level,
    })


def seed_background_issues() -> None:
    for _ in range(2):
        spawn_minor_issue(quiet=True)


def spawn_minor_issue(quiet: bool = False) -> None:
    """Routine anomaly the agent can clear without asking anyone."""
    kind = random.choice(list(MINOR_ISSUE_TYPES))
    lo, hi, has_photo = MINOR_ISSUE_TYPES[kind]
    tag = random.choice(list(ZONES))
    severity = random.randint(lo, hi)
    issue = {
        "id": next_id("ISS"), "type_key": f"ty_{kind}", "kind": kind, "zone": tag,
        "node": f"J-{random.randint(1000, 4999)}", "severity": severity,
        "confidence": round(random.uniform(0.61, 0.88), 2),
        "opened": datetime.now(), "ttl": random.randint(4, 9),
        "critical": False, "photo": has_photo,
    }
    st.session_state.queue.append(issue)
    add_work_order("wo_verify", {"type": t(issue["type_key"]).lower(),
                                 "node": issue["node"]}, issue, "eval")
    if not quiet:
        log("triage", "lg_triage", zone=zone_name(tag), node=issue["node"],
            type=t(issue["type_key"]).lower(), sev=severity, conf=issue["confidence"])


def add_work_order(title_key: str, title_params: dict, issue: dict,
                   status: str, hot: bool = False) -> str:
    """Kanban card. `status` is a stable code; the label is looked up at draw."""
    wo_id = next_id("WO")
    st.session_state.work_orders.append({
        "id": wo_id, "title_key": title_key, "title_params": title_params,
        "issue_id": issue["id"], "zone": issue["zone"], "node": issue["node"],
        "status": status, "crew": None, "hot": hot,
    })
    return wo_id


def move_work_order(issue_id: str, status: str, crew: str | None = None) -> None:
    for wo in st.session_state.work_orders:
        if wo["issue_id"] == issue_id:
            wo["status"] = status
            if crew:
                wo["crew"] = crew


def close_work_order(issue_id: str) -> None:
    st.session_state.work_orders = [
        wo for wo in st.session_state.work_orders if wo["issue_id"] != issue_id]


def trigger_critical_anomaly() -> None:
    """The demo button: collapse pressure in the old city and wake the agent."""
    ss = st.session_state
    if ss.incident is not None:
        log("triage", "lg_dup", level="warn", zone=zone_name(ss.incident["issue"]["zone"]))
        return

    tag = ANOMALY_ZONE
    before = BASE_PRESSURE[tag] * ss.pressure_scale[tag]
    ss.pressure_scale[tag] = 1.0 - ANOMALY_PRESSURE_DROP
    ss.flow_scale[tag] = ANOMALY_FLOW_SURGE
    ss.repair_active = False
    # Take a reading immediately so the cliff edge is on the chart even if
    # the stream is paused when the button is pressed.
    sample_telemetry(advance_clock=True, silent=True)
    after = BASE_PRESSURE[tag] * ss.pressure_scale[tag]

    issue = {
        "id": next_id("ISS"), "type_key": "ty_suspected_break", "kind": "main_break",
        "zone": tag, "node": f"J-{random.randint(4000, 4999)}", "severity": 58,
        "confidence": 0.62, "opened": datetime.now(), "ttl": 999,
        "critical": True, "photo": True,
    }
    ss.queue.append(issue)
    ss.incident = {
        "issue": issue, "stage_index": 0, "before": before, "after": after,
        "valve": f"RV-{tag[-2:]}", "households": random.randint(1100, 2400),
        "complaints": random.randint(5, 11), "dia": random.choice([200, 250, 300, 400]),
        "crew": f"CRW-{random.randint(10, 39)}",
        # 380 m3 of Narmada water at IMC's own cost of ~Rs 21 per 1000 L
        "loss_cost": int(380 * 1000 / 1000 * 21),
        "resolved": False,
    }
    add_work_order("wo_assess", {"node": issue["node"]}, issue, "eval", hot=True)

    agent = get_agent()
    step = agent.think(incident_observation(ss.incident), "detect")
    emit(step, "detect", ss.incident)
    ss.incident["stage_index"] = 1


def incident_observation(inc: dict) -> dict:
    """The context bundle handed to the model at every stage."""
    return {
        "zone": zone_name(inc["issue"]["zone"]),
        "zone_tag": inc["issue"]["zone"],
        "node": inc["issue"]["node"],
        "valve": inc["valve"],
        "households": indian_number(inc["households"]),
        "complaints": inc["complaints"],
        "dia": inc["dia"],
        "cost": indian_number(inc["loss_cost"]),
        "tag": t(random.choice(["cmp_dirty", "cmp_nowater", "cmp_flood"])),
        "drop": ANOMALY_PRESSURE_DROP * 100,
        "before": inc["before"],
        "after": inc["after"],
        "surge": (ANOMALY_FLOW_SURGE - 1) * 100,
        "proposed_action": t("gate_action", valve=inc["valve"],
                             zone=zone_name(inc["issue"]["zone"])),
    }


def emit(step: dict, stage: str, inc: dict) -> None:
    """Write one model step to the log and update the issue it concerns."""
    log(stage, step.get("key"), step["level"], text=step.get("text"),
        **incident_observation(inc))
    inc["issue"]["severity"] = step["severity"]
    inc["issue"]["confidence"] = step["confidence"]


def run_agent_cycle() -> None:
    """One reasoning pass, called once per interval."""
    ss = st.session_state
    agent = get_agent()
    inc = ss.incident

    if inc and not inc["resolved"]:
        idx = inc["stage_index"]
        if idx < len(INCIDENT_SCRIPT):
            stage = INCIDENT_SCRIPT[idx][0]
            step = agent.think(incident_observation(inc), stage)
            emit(step, stage, inc)
            inc["stage_index"] += 1
            if stage == "diagnose":
                inc["issue"]["type_key"] = "ty_main_break"
            if stage == "escalate":
                open_escalation(inc)
        else:
            monitor_recovery(inc)
        return

    if random.random() < 0.16 and len(ss.queue) < 6:
        spawn_minor_issue()
    age_minor_issues()

    if random.random() < 0.45:
        tag, key, level = random.choice(IDLE_CHATTER)
        zone = random.choice(list(ZONES))
        log(tag, key, level, n=len(ZONES), zone=zone_name(zone),
            v=random.uniform(9, 61), hh=random.choice([7, 8, 18, 19, 20]))


def age_minor_issues() -> None:
    """Advance and eventually close routine issues so the board keeps moving."""
    ss = st.session_state
    still_open = []
    for issue in ss.queue:
        if issue["critical"]:
            still_open.append(issue)
            continue
        issue["ttl"] -= 1
        if issue["ttl"] == 3:
            move_work_order(issue["id"], "action", crew=f"CRW-{random.randint(10, 39)}")
            log("act", "lg_autoapprove", "act", id=issue["id"],
                zone=zone_name(issue["zone"]), sev=issue["severity"],
                threshold=ss.get("approval_threshold", 70))
        elif issue["ttl"] <= 0:
            close_work_order(issue["id"])
            ss.resolved_count += 1
            log("resolve", "lg_resolve_minor", "ok", type=t(issue["type_key"]),
                node=issue["node"], zone=zone_name(issue["zone"]))
            continue
        still_open.append(issue)
    ss.queue = still_open


def open_escalation(inc: dict) -> None:
    """Raise the human-in-the-loop gate. This pauses the simulation."""
    ss = st.session_state
    obs = incident_observation(inc)
    ss.escalation = {
        "action": obs["proposed_action"], "issue": inc["issue"], "valve": inc["valve"],
        "households": obs["households"], "confidence": inc["issue"]["confidence"],
        "raised_at": datetime.now(),
    }
    move_work_order(inc["issue"]["id"], "wait")


def resolve_escalation(approved: bool) -> None:
    """Operator decided. Record it, act on it, resume the stream."""
    ss = st.session_state
    gate = ss.escalation
    if gate is None:
        return
    inc = ss.incident
    ss.decisions.append({"at": datetime.now(), "action": gate["action"],
                         "verdict": "approved" if approved else "overridden"})

    if approved:
        ss.repair_active = True
        move_work_order(gate["issue"]["id"], "action", crew=inc["crew"])
        log("human", "lg_human_ok", "ok", action=gate["action"], id=gate["issue"]["id"])
        log("execute", "lg_exec_ok", "act", valve=gate["valve"], crew=inc["crew"],
            households=gate["households"])
    else:
        ss.repair_active = False
        move_work_order(gate["issue"]["id"], "action", crew=inc["crew"])
        log("human", "lg_human_no", "warn", action=gate["action"])
        log("execute", "lg_exec_no", "warn", node=gate["issue"]["node"], crew=inc["crew"])

    ss.escalation = None
    if inc:
        inc["stage_index"] = len(INCIDENT_SCRIPT)   # move into monitoring


def monitor_recovery(inc: dict) -> None:
    """Tail end of the incident: report recovery, then close it out."""
    ss = st.session_state
    tag = inc["issue"]["zone"]
    scale = ss.pressure_scale[tag]
    psi = BASE_PRESSURE[tag] * scale

    if not ss.repair_active:
        if random.random() < 0.5:
            log("monitor", "lg_hold", "warn", zone=zone_name(tag), psi=psi,
                gap=PRESSURE_ALARM_PSI - psi)
        return

    if scale >= 0.995:
        inc["resolved"] = True
        ss.queue = [i for i in ss.queue if i["id"] != inc["issue"]["id"]]
        close_work_order(inc["issue"]["id"])
        ss.resolved_count += 1
        ss.incident = None
        log("resolve", "lg_resolved", "ok", zone=zone_name(tag), psi=psi,
            node=inc["issue"]["node"])
    else:
        log("monitor", "lg_monitor", "ok", zone=zone_name(tag), psi=psi, pct=scale * 100)


def advance_simulation() -> None:
    sample_telemetry()
    run_agent_cycle()


# =============================================================================
# SECTION 9 -- UI
# =============================================================================

def section_header(number: str, text: str) -> None:
    html(f"""
    <div class="sec-head">
      <span class="num">{esc(number)}</span>
      <span class="txt">{esc(text)}</span>
      <span class="rule"></span>
    </div>""")


def render_masthead() -> None:
    ss = st.session_state
    mins, secs = divmod(int((datetime.now() - ss.booted_at).total_seconds()), 60)
    wordmark = APP_NAME_HI if lang() == "hi" else APP_NAME
    html(f"""
    <div class="masthead">
      <div>
        <div class="wordmark">{esc(wordmark)}</div>
        <div class="tagline">{esc(t('subtitle'))}</div>
        <div class="org">{esc(t('operator'))}</div>
      </div>
      <div class="clock">
        {esc(datetime.now().strftime('%d %b %Y  %H:%M:%S'))}<br>
        {esc(t('tick'))} <b>{ss.tick:05d}</b> &nbsp; {esc(t('uptime'))}
        <b>{mins:02d}:{secs:02d}</b> &nbsp; {esc(BUILD_TAG)}
      </div>
    </div>
    <div class="factstrip">{esc(t('facts',
        mld=CITY_FACTS['supply_mld'], km=indian_number(CITY_FACTS['network_km']),
        tanks=CITY_FACTS['tanks_on_scada'], wards=CITY_FACTS['wards'],
        zones=CITY_FACTS['zones']))}</div>""")


def render_sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown(f"### {t('mission_control')}")

        # Language first: everything below it changes when this changes.
        st.radio(t("language"), options=["en", "hi"], horizontal=True, key="lang",
                 format_func=lambda c: "English" if c == "en" else "हिंदी")

        if ss.escalation is not None:
            lamp, title, sub = "red", t("st_gate"), t("st_gate_sub")
        elif ss.incident is not None:
            lamp, title = "amber", t("st_incident")
            sub = t("st_incident_sub", zone=zone_name(ss.incident["issue"]["zone"]))
        elif ss.streaming:
            lamp, title, sub = "green", t("st_nominal"), t("st_nominal_sub", n=len(ZONES))
        else:
            lamp, title, sub = "grey", t("st_idle"), t("st_idle_sub")
        html(f"""
        <div class="status-chip"><span class="lamp {lamp}"></span>
          <span class="status-txt">{esc(title)}<i>{esc(sub)}</i></span></div>""")

        st.markdown("---")
        st.markdown(f"**{t('simulation')}**")
        st.toggle(t("start_stream"), key="streaming", help=t("start_stream_help"))

        if st.button(t("trigger"), **WIDE, disabled=ss.incident is not None):
            ss.pending_trigger = True
            st.rerun()
        if st.button(t("reset"), **WIDE):
            ss._v = None
            st.rerun()

        st.markdown("---")
        st.markdown(f"**{t('agent_config')}**")
        st.selectbox(t("model"), list(LLM_BACKENDS), key="llm_backend")
        cfg = LLM_BACKENDS[ss.llm_backend]
        levels = ["supervised", "semi", "auto"]
        ss.autonomy = st.selectbox(          # language-scoped key, see render_chart
            t("autonomy"), levels,
            index=levels.index(ss.get("autonomy", "semi")),
            key=f"autonomy_{lang()}", help=t("autonomy_help"),
            format_func=lambda k: t(f"aut_{k}"))
        st.slider(t("temperature"), 0.0, 1.0, 0.2, 0.05, key="temperature")
        st.slider(t("threshold"), 0, 100, 70, 5, key="approval_threshold",
                  help=t("threshold_help"))

        html(f"""
        <div class="panel" style="margin-top:14px"><div class="side-note">
          {esc(t('lbl_runtime'))} <b>{esc(cfg['runtime'])}</b><br>
          {esc(t('lbl_model'))} <b>{esc(cfg['id'])}</b><br>
          {esc(t('lbl_cost'))} <b>{esc(cfg['cost'])}</b><br>
          {esc(t('lbl_tokens'))} <b>{indian_number(ss.tokens_used)}</b><br>
          {esc(t('lbl_resolved'))} <b>{ss.resolved_count}</b>
        </div></div>""")

        if ss.decisions:
            st.markdown(f"**{t('decisions')}**")
            for d in reversed(ss.decisions[-4:]):
                colour = "var(--ok)" if d["verdict"] == "approved" else "var(--warn)"
                html(f"""
                <div class="side-note" style="margin-bottom:7px">
                  <span style="color:{colour}">{esc(t(d['verdict']))}</span>
                  &nbsp;{esc(d['at'].strftime('%H:%M:%S'))}<br>
                  <b>{esc(d['action'])}</b></div>""")


def render_kpis() -> None:
    ss = st.session_state
    readings = latest_readings()
    if not readings:
        return
    avg_p = sum(r["pressure"] for r in readings.values()) / len(readings)
    tot_f = sum(r["flow"] for r in readings.values())
    n_anom = len(ss.queue)
    prev = ss.prev_kpi

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.3])
    with c1:
        st.metric(t("kpi_pressure"), f"{avg_p:.1f} PSI",
                  None if prev["pressure"] is None else f"{avg_p - prev['pressure']:+.1f} PSI")
    with c2:
        st.metric(t("kpi_flow"), f"{tot_f:.1f} L/s",
                  None if prev["flow"] is None else f"{tot_f - prev['flow']:+.1f} L/s",
                  delta_color="inverse")   # rising flow with falling pressure is bad
    with c3:
        st.metric(t("kpi_anomalies"), f"{n_anom}",
                  f"{n_anom - prev['anomalies']:+d}" if n_anom != prev["anomalies"] else None,
                  delta_color="inverse")
    with c4:
        worst_tag, worst = min(readings.items(), key=lambda kv: kv[1]["pressure"])
        st.metric(t("kpi_lowest", zone=zone_name(worst_tag)),
                  f"{worst['pressure']:.1f} PSI",
                  t("kpi_floor", v=f"{worst['pressure'] - PRESSURE_ALARM_PSI:+.1f}"))
    ss.prev_kpi = {"pressure": avg_p, "flow": tot_f, "anomalies": n_anom}


def render_chart() -> None:
    df = telemetry_df()
    if df.empty:
        return
    # A widget whose option LABELS change with language needs a
    # language-scoped key, or the value stored under the old labels goes
    # stale the moment the operator switches. The real choice is mirrored
    # into a plain state key so switching language never resets it.
    metrics = ["pressure", "flow"]
    col = st.radio("trend", metrics,
                   index=metrics.index(st.session_state.get("chart_metric", "pressure")),
                   horizontal=True, label_visibility="collapsed",
                   key=f"chart_metric_{lang()}",
                   format_func=lambda c: t("chart_" + c))
    st.session_state.chart_metric = col

    fig = go.Figure()
    for tag, meta in ZONES.items():
        sub = df[df["zone"] == tag]
        fig.add_trace(go.Scatter(
            x=sub["tick"], y=sub[col], name=zone_name(tag), mode="lines",
            line=dict(color=meta["color"], width=2, shape="spline", smoothing=0.6),
            hovertemplate=f"<b>{tag}</b> %{{y:.1f}}<extra></extra>"))

    if col == "pressure":
        fig.add_hline(y=PRESSURE_ALARM_PSI, line_color="#E5484D", line_width=1,
                      line_dash="dot",
                      annotation_text=t("service_floor", v=int(PRESSURE_ALARM_PSI)),
                      annotation_position="top left",
                      annotation_font=dict(color="#E5484D", size=10,
                                           family="JetBrains Mono, Noto Sans Devanagari, monospace"))

    fig.update_layout(
        height=300, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, Noto Sans Devanagari, monospace",
                  size=11, color="#617B92"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0C1522", bordercolor="#17293C",
                        font=dict(family="JetBrains Mono, Noto Sans Devanagari, monospace",
                                  size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, zeroline=False, title=None, linecolor="#17293C"),
        yaxis=dict(gridcolor="rgba(23,41,60,.75)", zeroline=False, title=None,
                   linecolor="#17293C"))
    st.plotly_chart(fig, **WIDE, config={"displayModeBar": False})


def severity_class(score: int) -> str:
    if score >= 85:
        return "sev-crit"
    if score >= 65:
        return "sev-high"
    if score >= 45:
        return "sev-med"
    return "sev-low"


def render_priority_queue() -> None:
    ss = st.session_state
    if not ss.queue:
        html(f'<div class="panel"><div class="log-empty">{esc(t("queue_empty"))}</div></div>')
        return

    for issue in sorted(ss.queue, key=lambda i: -i["severity"]):
        cls = severity_class(issue["severity"])
        age = int((datetime.now() - issue["opened"]).total_seconds())
        crew = next((wo["crew"] for wo in ss.work_orders
                     if wo["issue_id"] == issue["id"] and wo["crew"]), None)
        # Issues a resident would photograph carry a 311 attachment.
        photo = (f'<div class="pq-photo">'
                 f'{photo_block(issue["kind"], issue["zone"], t("photo_311"))}</div>'
                 if issue.get("photo") else "")
        html(f"""
        <div class="pq-card {cls}">
          {photo}
          <div class="pq-main">
            <div class="pq-top">
              <span class="pq-type">{esc(t(issue['type_key']))}</span>
              <span class="pq-score">{issue['severity']}</span>
            </div>
            <div class="pq-meta">
              <span>{esc(zone_name(issue['zone']))}</span>
              <span>{esc(t('node'))} {esc(issue['node'])}</span>
              <span>{esc(t('open_for', s=age))}</span>
            </div>
            <div class="pq-bar"><i style="width:{issue['severity']}%"></i></div>
            <div class="pq-foot">
              <span>{esc(t('confidence'))} <b>{issue['confidence']:.2f}</b></span>
              <span>{esc(issue['id'])}{f" &nbsp; {esc(t('crew'))} <b>{esc(crew)}</b>" if crew else ""}</span>
            </div>
          </div>
        </div>""")


def render_cot_log() -> None:
    """Streaming reasoning trace, newest line at the top.

    Streamlit strips <script> from markdown, so a div cannot be auto-scrolled
    to the bottom. Newest-first keeps the live line in view without fighting
    the scroll position on every rerun. Messages are rendered from their keys
    here, which is why switching language rewrites the whole history.
    """
    rows = list(st.session_state.cot_log)
    if not rows:
        body = f'<div class="log-empty">{esc(t("log_empty"))}</div>'
    else:
        parts = []
        for r in rows:
            msg = r["text"] if r.get("text") else t(r["key"], **r.get("params", {}))
            parts.append(
                f'<div class="log-row lvl-{esc(r["level"])}">'
                f'<span class="log-ts">{esc(r["ts"])}</span>'
                f'<span class="log-tag">{esc(t("tag_" + r["tag"]))}</span>'
                f'<span class="log-msg">{esc(msg)}</span></div>')
        body = "".join(parts)
    html(f'<div class="cot-shell">{body}</div>')


def render_kanban() -> None:
    ss = st.session_state
    columns = [("eval", t("kb_eval")), ("action", t("kb_action")), ("wait", t("kb_wait"))]
    cols = st.columns(3, gap="small")
    for col, (code, label) in zip(cols, columns):
        items = [wo for wo in ss.work_orders if wo["status"] == code]
        cards = "".join(
            f'<div class="kb-card {"hot" if wo["hot"] else ""}">'
            f'<div class="kb-id">{esc(wo["id"])}</div>'
            f'<div class="kb-title">{esc(t(wo["title_key"], **wo["title_params"]))}</div>'
            f'<div class="kb-sub"><span>{esc(zone_name(wo["zone"]))}</span>'
            f'<span>{esc(wo["crew"]) if wo["crew"] else esc(t("unassigned"))}</span>'
            f'</div></div>'
            for wo in items
        ) or f'<div class="kb-empty">{esc(t("kb_empty"))}</div>'
        with col:
            html(f"""
            <div class="kb-col">
              <div class="kb-col-head"><span>{esc(label)}</span><i>{len(items)}</i></div>
              {cards}
            </div>""")


def render_escalation_gate() -> bool:
    """The human-in-the-loop gate. Returns True while a decision is pending."""
    gate = st.session_state.escalation
    if gate is None:
        return False
    issue = gate["issue"]
    zone = zone_name(issue["zone"])

    st.error(t("gate_banner"), icon="🛑")
    html(f"""
    <div class="gate">
      <div class="gate-main">
        <div class="gate-kicker">{esc(t('gate_kicker'))}</div>
        <div class="gate-title">{esc(gate['action'])}</div>
        <div class="gate-body">{esc(t('gate_body', households=gate['households']))}</div>
        <div class="gate-grid">
          <div>{esc(t('gate_issue'))}<b>{esc(issue['id'])}</b></div>
          <div>{esc(t('gate_zone'))}<b>{esc(zone)}</b></div>
          <div>{esc(t('gate_severity'))}<b>{issue['severity']}</b></div>
          <div>{esc(t('gate_confidence'))}<b>{gate['confidence']:.2f}</b></div>
          <div>{esc(t('gate_raised'))}<b>{esc(gate['raised_at'].strftime('%H:%M:%S'))}</b></div>
          <div>{esc(t('gate_alt'))}<b>{esc(t('gate_alt_val', node=issue['node']))}</b></div>
        </div>
      </div>
      <div class="gate-photo">
        {photo_block('main_break', issue['zone'], t('gate_site', zone=zone))}
      </div>
    </div>""")

    c1, c2, c3 = st.columns([1.1, 1, 3])
    with c1:
        if st.button(t("approve"), type="primary", **WIDE):
            resolve_escalation(approved=True)
            st.rerun()
    with c2:
        if st.button(t("override"), **WIDE):
            resolve_escalation(approved=False)
            st.rerun()
    with c3:
        st.markdown(f'<div class="side-note" style="padding-top:9px">{esc(t("gate_audit"))}</div>',
                    unsafe_allow_html=True)
    return True


# =============================================================================
# SECTION 10 -- MAIN
# =============================================================================

def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} | IMC Water Operations",
                       page_icon="💧", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    init_state()
    ss = st.session_state
    if lang() == "hi":
        st.markdown(HINDI_CSS, unsafe_allow_html=True)

    if ss.pending_trigger:
        ss.pending_trigger = False
        trigger_critical_anomaly()

    gate_open = ss.escalation is not None
    if ss.streaming and not gate_open:
        advance_simulation()

    render_sidebar()
    render_masthead()

    # The gate outranks everything on the page. When the agent needs a
    # decision it sits directly under the masthead, so the one moment that
    # sells the whole demo can never scroll out of view.
    paused = render_escalation_gate()

    section_header("01", t("sec_network"))
    render_city_map()
    st.write("")

    section_header("02", t("sec_perception"))
    render_kpis()
    st.write("")
    render_chart()
    st.write("")

    left, right = st.columns([1, 1.3], gap="large")
    with left:
        section_header("03", t("sec_priority"))
        render_priority_queue()
    with right:
        section_header("04", t("sec_reasoning"))
        render_cot_log()
    st.write("")

    section_header("05", t("sec_execution"))
    render_kanban()

    # The heartbeat. One interval per refresh, and it stops dead while the
    # escalation gate is open -- which is what makes the human-in-the-loop
    # moment feel real: the numbers freeze until someone decides.
    if ss.streaming and not paused:
        time.sleep(TICK_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()

# =============================================================================
# OPTIONAL: REAL PHOTOGRAPHS
# =============================================================================
# The app draws placeholders when no photos are present. To use real ones,
# create an ./assets folder next to this file:
#
#   assets/
#     311/main_break.jpg        <- shown in the escalation gate + break cards
#     311/turbidity.jpg         <- shown on dirty-water complaints
#     311/chlorine_drop.jpg
#     311/idz-02-main_break.jpg <- zone-specific version, takes priority
#     zones/idz-02.jpg          <- fallback if no complaint photo matches
#
# Lookup order per card: zone-specific complaint photo, then generic
# complaint photo, then the zone photo, then the drawn placeholder. Any
# missing file simply falls through, so a half-filled folder is fine.
#
# For photographs you can legally show in a public demo, Wikimedia Commons
# has Indore material under CC licences -- search "Indore" or "Rajwada"
# there and check each file's licence, then credit it on your closing slide.
# Do not scrape press photos of the Bhagirathpura contamination; the point
# lands harder as a number in your own deck than as someone else's picture.
# =============================================================================
