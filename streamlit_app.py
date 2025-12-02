# -*- coding: utf-8 -*-
import streamlit as st
from dataclasses import dataclass
from typing import Any, Dict, List
import time, json, hashlib
import numpy as np
import pandas as pd
import random
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="PoSAC Test Demo with Competitive and Malicious Node", layout="wide")

# -----------------------------
# Helper deterministic "signature"
def sign_string(s: str, node_id: str) -> str:
    return hashlib.sha256((node_id + "|" + s).encode()).hexdigest()

def verify_signature(s: str, node_id: str, signature: str) -> bool:
    return sign_string(s, node_id) == signature

# -----------------------------
# Useful compute
def useful_compute_train_predict(data_x: List[float], data_y: List[float], query_x: float) -> float:
    X = np.array(data_x, dtype=float)
    Y = np.array(data_y, dtype=float)
    if X.size == 0:
        return 0.0
    A = np.vstack([X, np.ones_like(X)]).T
    try:
        w, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    except Exception:
        w, b = 0.0, float(np.mean(Y))
    return float(w * query_x + b)

def make_deterministic_sample(seed_str: str, n=10):
    s = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(s)
    x = rng.normal(0.0, 1.0, n).tolist()
    y = (2.0 * np.array(x) + rng.normal(0, 0.1, n)).tolist()
    return x, y

# -----------------------------
# Message types
@dataclass
class PrePrepare:
    view: int
    seq: int
    digest: str
    proposer: str
    signature: str
    payload: Dict[str, Any]

@dataclass
class Prepare:
    view: int
    seq: int
    digest: str
    node: str
    signature: str

@dataclass
class Commit:
    view: int
    seq: int
    digest: str
    node: str
    signature: str

# -----------------------------
# Block
@dataclass
class Block:
    index: int
    seq: int
    proposer: str
    digest: str
    payload: Dict[str, Any]
    committed: bool = False
    committed_at: float = None

# -----------------------------
# Node
class Node:
    def __init__(self, node_id: str, all_node_ids: List[str]):
        self.id = node_id
        self.chain: List[Block] = []
        self.view = 0
        self.log_preprepare: Dict[int, PrePrepare] = {}
        self.log_prepare: Dict[int, List[Prepare]] = {}
        self.log_commit: Dict[int, List[Commit]] = {}
        self.executed: List[Block] = []
        self.peers = [nid for nid in all_node_ids if nid != node_id]
        self.all_nodes = all_node_ids
        self.conflicts = 0

    def on_receive_preprepare(self, msg: PrePrepare):
        core = f"{msg.view}|{msg.seq}|{msg.digest}"
        if not verify_signature(core, msg.proposer, msg.signature):
            return {"status": "invalid_signature"}
        if msg.seq in self.log_preprepare:
            self.conflicts += 1
        self.log_preprepare[msg.seq] = msg
        self.log_prepare.setdefault(msg.seq, [])
        self.log_commit.setdefault(msg.seq, [])
        return {"status": "accepted", "seq": msg.seq}

    def on_receive_prepare(self, msg: Prepare):
        core = f"{msg.view}|{msg.seq}|{msg.digest}"
        if not verify_signature(core, msg.node, msg.signature):
            return {"status": "invalid_signature"}
        plist = self.log_prepare.setdefault(msg.seq, [])
        if any(p.node == msg.node for p in plist):
            return {"status": "duplicate"}
        plist.append(msg)
        return {"status": "ok", "count": len(plist)}

    def on_receive_commit(self, msg: Commit):
        core = f"{msg.view}|{msg.seq}|{msg.digest}"
        if not verify_signature(core, msg.node, msg.signature):
            return {"status": "invalid_signature"}
        clist = self.log_commit.setdefault(msg.seq, [])
        if any(c.node == msg.node for c in clist):
            return {"status": "duplicate"}
        clist.append(msg)
        if len(clist) >= self._commit_required(len(self.all_nodes)):
            pp = self.log_preprepare.get(msg.seq)
            if pp and not any(b.seq == msg.seq for b in self.chain):
                if not verify_payload_compute(pp.payload):
                    return {"status": "payload_verify_failed"}
                block = Block(index=len(self.chain), seq=msg.seq, proposer=pp.proposer,
                              digest=pp.digest, payload=pp.payload, committed=True, committed_at=time.time())
                self.chain.append(block)
                self.executed.append(block)
                return {"status": "committed", "seq": msg.seq}
        return {"status": "ok", "commits": len(clist)}

    def _commit_required(self, n):
        f = (n - 1) // 3
        return 2 * f + 1

# -----------------------------
# Cluster
class Cluster:
    def __init__(self, n_nodes: int):
        if n_nodes < 4:
            raise ValueError("The number of nodes should be >= 4")
        self.n = n_nodes
        self.node_ids = [f"node-{i+1}" for i in range(n_nodes)]
        self.nodes: Dict[str, Node] = {nid: Node(nid, self.node_ids) for nid in self.node_ids}
        self.seq_counter = 0
        self.view = 0
        self.validators = list(self.node_ids)
        self.malicious_logs: List[Dict[str, Any]] = []

    def expected_proposer(self, seq):
        n_attempt = random.randint(1, len(self.validators))
        return random.sample(self.validators, n_attempt)

    def propose_block_by_client(self, client_payload: Dict[str, Any], malicious_node=None, tamper_prob=0) -> Dict[str, Any]:
        self.seq_counter += 1
        seq = self.seq_counter
        selected_nodes = self.expected_proposer(seq)
        proposer = selected_nodes[0]

        seed_str = json.dumps(client_payload, sort_keys=True) + f"|{seq}|{proposer}"
        x, y = make_deterministic_sample(seed_str, n=12)
        query_x = float(client_payload.get("query_x", 1.0))
        prediction = useful_compute_train_predict(x, y, query_x)
        payload = {"client_payload": client_payload, "sample_x": x, "sample_y": y,
                   "query_x": query_x, "prediction": prediction}

        tampered = False
        if malicious_node is not None and proposer == malicious_node:
            if random.random() < tamper_prob / 100.0:
                payload["prediction"] += random.uniform(5, 10)
                tampered = True
                self.malicious_logs.append({
                    "time": time.strftime("%H:%M:%S"),
                    "seq": seq,
                    "node": malicious_node,
                    "behavior": "Tampered prediction",
                    "payload": payload["prediction"]
                })

        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        core = f"{self.view}|{seq}|{digest}"

        conflicts = 0
        preprepare = PrePrepare(view=self.view, seq=seq, digest=digest, proposer=proposer,
                                signature=sign_string(core, proposer), payload=payload)
        for nid in selected_nodes:
            r = self.nodes[nid].on_receive_preprepare(preprepare)
            if r["status"] != "accepted":
                conflicts += 1

        for nid in selected_nodes:
            sig = sign_string(core, nid)
            prepare = Prepare(view=self.view, seq=seq, digest=digest, node=nid, signature=sig)
            commit = Commit(view=self.view, seq=seq, digest=digest, node=nid, signature=sig)
            for node in self.nodes.values():
                if tampered and node.id == malicious_node:
                    continue
                node.on_receive_prepare(prepare)
                node.on_receive_commit(commit)

        alert = None
        if tampered:
            alert = f"⚠️ Malicious node {malicious_node} attempted tampering at seq {seq}!"

        return {"seq": seq, "proposer": proposer, "prediction": prediction,
                "conflicts": conflicts, "alert": alert}

    def reset_chain(self):
        for n in self.nodes.values():
            n.chain = []
            n.log_preprepare = {}
            n.log_prepare = {}
            n.log_commit = {}
            n.executed = []
            n.conflicts = 0
        self.seq_counter = 0
        self.malicious_logs = []

# -----------------------------
def verify_payload_compute(payload: Dict[str, Any]) -> bool:
    try:
        x = payload["sample_x"]
        y = payload["sample_y"]
        qx = float(payload["query_x"])
        recomputed = useful_compute_train_predict(x, y, qx)
        return abs(recomputed - float(payload["prediction"])) < 1e-6
    except Exception:
        return False

# -----------------------------
def compute_node_metrics(cluster: Cluster):
    data = []
    for nid, node in cluster.nodes.items():
        executed_preds = [b.payload.get("prediction",0) for b in node.executed[-20:]]
        data.append({"node": nid, "chain_height": len(node.chain),
                     "executed_count": len(node.executed), "predictions": executed_preds,
                     "conflicts": node.conflicts})
    return data

# -----------------------------
# Sidebar
with st.sidebar:
    st.header("Auto Simulation Settings")
    n_nodes = st.number_input("Number of nodes", min_value=4, max_value=12, value=5)
    auto_interval = st.number_input("Auto propose interval (s)", min_value=0.1, max_value=5.0, value=0.8, step=0.1)
    query_x = st.number_input("Query x", value=1.0)
    st.header("Malicious Node Settings")
    malicious_node = st.selectbox("Select malicious node (optional)",
                                  options=[None] + [f"node-{i+1}" for i in range(n_nodes)])
    tamper_prob = st.slider("Tamper probability (%) per block", 0, 100, 50)


    # Pause / Resume
    if "paused" not in st.session_state:
        st.session_state["paused"] = False

    def toggle_pause():
        st.session_state["paused"] = not st.session_state["paused"]

    pause_label = "▶️ Resume Auto Run" if st.session_state["paused"] else "⏸️ Pause Auto Run"
    st.button(pause_label, on_click=toggle_pause)


# -----------------------------
# Initialize cluster
if "cluster" not in st.session_state or st.session_state.get("n_nodes") != n_nodes:
    st.session_state["cluster"] = Cluster(n_nodes)
    st.session_state["n_nodes"] = n_nodes
cluster: Cluster = st.session_state["cluster"]

# -----------------------------
# Auto propose + refresh
REFRESH_INTERVAL = 1000
st_autorefresh(interval=REFRESH_INTERVAL, key="competitive_auto_refresh")
if "auto_timer" not in st.session_state:
    st.session_state["auto_timer"] = 0.0
st.session_state["auto_timer"] += REFRESH_INTERVAL / 1000.0

if not st.session_state["paused"]:
    if st.session_state["auto_timer"] >= auto_interval:
        result = cluster.propose_block_by_client(
            {"query_x": query_x},
            malicious_node=malicious_node,
            tamper_prob=tamper_prob
        )
        if result["alert"]:
            st.warning(result["alert"])
        st.session_state["auto_timer"] = 0.0

    if result["alert"]:
        st.warning(result["alert"])
    st.session_state["auto_timer"] = 0.0

# -----------------------------
# Display Metrics
st.title("PoSAC Test Demo with Competitive Demo and Malicious Node Detection")
node_metrics = compute_node_metrics(cluster)

# 1. Predictions line chart
fig_line = go.Figure()
for m in node_metrics:
    seqs = list(range(len(m["predictions"])))
    line_color = "red" if m["node"] == malicious_node else "blue"
    fig_line.add_trace(go.Scatter(x=seqs, y=m["predictions"], mode="lines+markers",
                                  name=m["node"], line=dict(color=line_color)))
st.plotly_chart(fig_line, use_container_width=True)

# 2. Chain height bar chart with red star for malicious
fig_height = go.Figure()
for m in node_metrics:
    color = "red" if m["node"] == malicious_node else "blue"
    text = f"★ {m['chain_height']}" if m["node"] == malicious_node else m['chain_height']
    fig_height.add_trace(go.Bar(x=[m["node"]], y=[m["chain_height"]],
                                marker_color=color, text=text, textposition="auto"))
st.plotly_chart(fig_height, use_container_width=True)

# 3. Executed blocks
df_exec = pd.DataFrame({"node": [m["node"] for m in node_metrics],
                        "executed": [m["executed_count"] for m in node_metrics]})
fig_exec = px.bar(df_exec, x="node", y="executed", color="node", title="Executed Blocks per Node", text="executed")
st.plotly_chart(fig_exec, use_container_width=True)

# 4. Conflicts
df_conflict = pd.DataFrame({"node": [m["node"] for m in node_metrics],
                            "conflicts": [m["conflicts"] for m in node_metrics]})
fig_conflict = px.bar(df_conflict, x="node", y="conflicts", color="node", title="Conflict / Rollback Counts", text="conflicts")
st.plotly_chart(fig_conflict, use_container_width=True)

# 5. Conflicts proportion
total_conflicts = sum(m["conflicts"] for m in node_metrics)
sizes = [m["conflicts"]/total_conflicts if total_conflicts>0 else 0 for m in node_metrics]
df_pie = pd.DataFrame({"node": [m["node"] for m in node_metrics], "conflict_ratio": sizes})
fig_pie = px.pie(df_pie, names="node", values="conflict_ratio", title="Conflict Ratio per Node")
st.plotly_chart(fig_pie, use_container_width=True)

# 6. Last 10 blocks per node
with st.expander("Show Last 10 Blocks per Node"):
    for m in node_metrics:
        node = cluster.nodes[m["node"]]
        if node.chain:
            df = pd.DataFrame([{"seq": b.seq, "proposer": b.proposer,
                                "prediction": round(b.payload.get("prediction",0),4)}
                               for b in node.chain[-10:]])
            st.markdown(f"**{m['node']}**")
            st.table(df)
        else:
            st.write(f"{m['node']}: No blocks yet")

# 7. Malicious behavior log table
if cluster.malicious_logs:
    st.subheader("Detected Malicious Behaviors")
    df_malicious = pd.DataFrame(cluster.malicious_logs)
    st.table(df_malicious)
