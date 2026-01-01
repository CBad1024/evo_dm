import streamlit as st
import subprocess
import sys
import os
import time
import fcntl
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(page_title="EvoDM Playground", layout="wide")

# Custom CSS for Playground feel
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        border-radius: 20px;
    }
    .highlight-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        text-align: center;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
NUM_TABS = 5
MODES = {
    "Simple SSWM": "simple_sswm",
    "RL Training (Legacy)": "sswm",
    "Wright-Fisher Landscapes": "wf_ls",
    "Wright-Fisher Seascapes": "wf_ss",
    "MDP Simulation": "mdp"
}

# Initialize session state for simulations
if "sims" not in st.session_state:
    st.session_state.sims = {}
    for i in range(NUM_TABS):
        st.session_state.sims[i] = {
            "mode": "Simple SSWM",
            "train": True,
            "process": None,
            "logs": "No logs yet. Click 'Run' to start.\n",
            "running": False,
            "exit_code": None,
            "hp": {
                "lr": 0.0001,
                "epochs": 50,
                "batch_size": 128,
                "n_mut": 4,
                "sigma": 0.5,
                "pop_size": 10000,
                "mutation_rate": 1e-5,
                "gen_per_step": 500
            }
        }

def start_simulation(tab_id):
    sim = st.session_state.sims[tab_id]
    mode_arg = MODES[sim["mode"]]
    train_arg = "--train" if sim["train"] else "--no-train"
    
    # Use the same python executable as current process
    py_path = sys.executable
    script_path = str(project_root / "examples" / "run.py")
    
    cmd = [
        py_path, script_path, 
        "--mode", mode_arg, 
        train_arg,
        "--lr", str(sim["hp"]["lr"]),
        "--epochs", str(sim["hp"]["epochs"]),
        "--batch-size", str(sim["hp"]["batch_size"]),
        "--n-mut", str(sim["hp"]["n_mut"]),
        "--sigma", str(sim["hp"]["sigma"]),
    ]

    # Mode-specific args
    if mode_arg in ["wf_ls", "wf_ss"]:
        cmd += [
            "--pop-size", str(sim["hp"]["pop_size"]),
            "--mutation-rate", str(sim["hp"]["mutation_rate"]),
            "--gen-per-step", str(sim["hp"]["gen_per_step"])
        ]

    if sim["train"] and sim.get("signature"):
        cmd += ["--signature", sim["signature"]]
    elif not sim["train"] and sim.get("selected_policy"):
        cmd += ["--filename", sim["selected_policy"]]
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(project_root),
        env=env,
        bufsize=0,
        text=False
    )
    
    if sys.platform != "win32":
        fd = process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    sim["process"] = process
    sim["running"] = True
    sim["logs"] = f"--- STARTED: {' '.join(cmd)} ---\n\n"
    sim["exit_code"] = None

def update_logs_for_sim(tab_id):
    sim = st.session_state.sims[tab_id]
    changed = False
    if sim["process"]:
        fd = sim["process"].stdout.fileno()
        try:
            while True:
                chunk = os.read(fd, 8192)
                if not chunk: break
                sim["logs"] += chunk.decode('utf-8', errors='replace')
                changed = True
        except (BlockingIOError, IOError):
            pass
        
        exit_code = sim["process"].poll()
        if exit_code is not None:
            sim["running"] = False
            sim["exit_code"] = exit_code
            try:
                while True:
                    chunk = os.read(fd, 8192)
                    if not chunk: break
                    sim["logs"] += chunk.decode('utf-8', errors='replace')
            except: pass
            sim["logs"] += f"\n\n--- PROCESS EXITED with code {exit_code} ---\n"
            sim["process"] = None
            changed = True
            
            if exit_code == 0:
                st.toast(f"✅ Simulation {tab_id+1} Successful!", icon="🎉")
            else:
                st.toast(f"❌ Simulation {tab_id+1} Failed (Code: {exit_code})", icon="⚠️")
    return changed

# Header
st.title("🔬 EvoDM Playground")
st.markdown("Tinker with Evolutionary Dynamics and Reinforcement Learning right in your browser.")

# UI Layout
tab_labels = [f"Simulation {i+1}" for i in range(NUM_TABS)]
tabs = st.tabs(tab_labels)

@st.fragment(run_every=2)
def render_tab_content(tab_id):
    update_logs_for_sim(tab_id)
    sim = st.session_state.sims[tab_id]
    
    # Top Control Bar
    with st.container():
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        cols_top = st.columns([2, 1, 1, 1, 1, 1])
        
        with cols_top[0]:
            sim["mode"] = st.selectbox(
                "Evolutionary Regime", 
                list(MODES.keys()), 
                index=list(MODES.keys()).index(sim["mode"]),
                key=f"mode_sel_{tab_id}"
            )
        with cols_top[1]:
            sim["hp"]["lr"] = st.selectbox(
                "Learning rate", 
                [0.1, 0.01, 0.001, 0.0001, 0.00001], 
                index=3,
                key=f"lr_{tab_id}"
            )
        with cols_top[2]:
            sim["hp"]["epochs"] = st.number_input("Epochs", 1, 1000, sim["hp"]["epochs"], key=f"epochs_{tab_id}")
        with cols_top[3]:
            sim["hp"]["batch_size"] = st.selectbox("Batch size", [16, 32, 64, 128, 256], index=3, key=f"batch_{tab_id}")
            
        with cols_top[4]:
             can_run = not sim["train"] or (sim["train"] and sim.get("signature", "").strip() != "")
             if not sim["running"]:
                if st.button("🚀 RUN", key=f"run_btn_{tab_id}", use_container_width=True, type="primary", disabled=not can_run):
                    start_simulation(tab_id)
                    st.rerun()
             else:
                if st.button("🛑 STOP", key=f"stop_btn_{tab_id}", use_container_width=True, type="secondary"):
                    if sim["process"]:
                        sim["process"].terminate()
                    st.rerun()
        
        with cols_top[5]:
            if st.button("🗑️ CLEAR", key=f"clear_tab_btn_{tab_id}", use_container_width=True):
                sim["logs"] = ""
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Main Area
    col_env, col_regime, col_log = st.columns([1, 1, 2])
    
    with col_env:
        st.subheader("Landscape Config")
        sim["hp"]["n_mut"] = st.slider("Number of mutations (N)", 1, 10, sim["hp"]["n_mut"], key=f"n_mut_{tab_id}")
        sim["hp"]["sigma"] = st.slider("Sigma (Noise)", 0.0, 1.0, sim["hp"]["sigma"], 0.1, key=f"sigma_{tab_id}")
        sim["train"] = st.checkbox("Enable Training", value=sim["train"], key=f"train_cb_{tab_id}")
        
        if sim["train"]:
            sim["signature"] = st.text_input("Run Signature", value=sim.get("signature", ""), key=f"sig_input_{tab_id}")
            if not sim["signature"]:
                st.caption("⚠️ :red[Run Signature is mandatory for training]")
        else:
             mode_arg = MODES[sim["mode"]]
             log_dir = "RL" if mode_arg in ["wf_ls", "wf_ss", "sswm"] else "sswm_dqn"
             path = project_root / "log" / log_dir
             policies = sorted([f.name for f in path.glob("*.pth")]) if path.exists() else []
             if policies:
                sim["selected_policy"] = st.selectbox("Select Policy", policies, key=f"policy_sel_{tab_id}")
             else:
                st.warning("No policies found.")

    with col_regime:
        st.subheader("Regime Specific")
        if MODES[sim["mode"]] in ["wf_ls", "wf_ss"]:
            sim["hp"]["pop_size"] = st.number_input("Population Size", 100, 1000000, sim["hp"]["pop_size"], key=f"pop_{tab_id}")
            sim["hp"]["mutation_rate"] = st.number_input("Mutation Rate", 0.0, 1.0, sim["hp"]["mutation_rate"], format="%.1e", key=f"mut_{tab_id}")
            sim["hp"]["gen_per_step"] = st.number_input("Gens per Step", 1, 10000, sim["hp"]["gen_per_step"], key=f"gps_{tab_id}")
        else:
            st.info("No specific parameters for this regime.")
            
        if sim["running"]:
            st.success(f"Running (PID: {sim['process'].pid})")
        elif sim["exit_code"] is not None:
            if sim["exit_code"] == 0: st.success("Finished")
            else: st.error(f"Failed ({sim['exit_code']})")
        else:
            st.warning("Idle")

    with col_log:
        st.subheader("Execution Logs")
        st.code(sim["logs"] if sim["logs"] else " ", language="text")

for i, tab in enumerate(tabs):
    with tab:
        render_tab_content(i)

st.divider()
st.caption("EvoDM - Evolutionary Dynamics Modeling with RL | support for concurrent executions")
