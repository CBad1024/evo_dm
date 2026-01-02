import streamlit as st
import subprocess
import sys
import os
import time
import fcntl
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from evodm.envs import define_mira_landscapes

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

DATASETS = {
    "Synthetic (NK)": {"N": "any", "description": "NK landscapes with tunable epistasis."},
    "Mira (E. Coli)": {"N": 4, "description": "Empirical fitness landscapes from Mira et al. (2015)."}
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
                "dataset": "Synthetic (NK)",
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
    
    # Add dataset/system info if Mira is selected
    if sim["hp"]["dataset"] == "Mira (E. Coli)":
        cmd += ["--n-mut", "4"] # Force N=4 for Mira
    
    # Cleanup old live files
    signature = sim.get("signature")
    if signature:
        for folder in ["trajectories", "policies"]:
            fpath = project_root / "log" / folder / f"{signature}_live.{'csv' if folder == 'trajectories' else 'json'}"
            if fpath.exists():
                fpath.unlink()

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

def plot_landscape_heatmap(tab_id):
    sim = st.session_state.sims[tab_id]
    if sim["hp"]["dataset"] == "Mira (E. Coli)":
        data = define_mira_landscapes()
        drug_names = ['AMP', 'AM', 'CEC', 'CTX', 'ZOX', 'CXM', 'CRO', 'AMC', 'CAZ', 'CTT', 'SAM', 'CPR', 'CPD', 'TZP', 'FEP']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, aspect='auto', cmap="viridis")
        fig.colorbar(im, ax=ax, label="Fitness")
        ax.set_xticks(range(16))
        ax.set_xticklabels([bin(i)[2:].zfill(4) for i in range(16)], rotation=45)
        ax.set_yticks(range(15))
        ax.set_yticklabels(drug_names)
        ax.set_title("Mira et al. (2015) Fitness Landscape")
        ax.set_xlabel("Genotypes")
        ax.set_ylabel("Drugs")
        st.pyplot(fig)
    else:
        st.info("Synthetic landscape visualization coming soon (requires generating a sample for N={}).".format(sim["hp"]["n_mut"]))

def plot_live_trajectory(tab_id):
    sim = st.session_state.sims[tab_id]
    signature = sim.get("signature")
    if not signature:
        return

    traj_file = project_root / "log" / "trajectories" / f"{signature}_live.csv"
    if traj_file.exists():
        try:
            df = pd.read_csv(traj_file)
            if not df.empty:
                st.subheader("Live Trajectory (Testing)")
                
                # Get the latest episode
                latest_ep = df['episode'].max()
                ep_df = df[df['episode'] == latest_ep]
                
                # Plot the path on a heatmap-like background or simple line plot
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(ep_df['step'], ep_df['genotype'], 'o-', label=f"Ep {latest_ep}")
                ax.set_yticks(range(2**sim["hp"]["n_mut"]))
                ax.set_yticklabels([bin(i)[2:].zfill(sim["hp"]["n_mut"]) for i in range(2**sim["hp"]["n_mut"])])
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Genotype")
                ax.set_title(f"Tumor Mutation Path (Mean Fitness: {np.mean(ep_df['fitness']):.4f})")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Also show fitness over time
                st.line_chart(ep_df.set_index('step')['fitness'], height=150)
        except Exception as e:
            st.error(f"Error loading trajectory: {e}")

def plot_live_policy(tab_id):
    sim = st.session_state.sims[tab_id]
    signature = sim.get("signature")
    if not signature:
        return

    policy_file = project_root / "log" / "policies" / f"{signature}_live.json"
    if not policy_file.exists():
        return
        
    try:
        with open(policy_file, "r") as f:
            data = json.load(f)
        
        q_values = np.array(data["q_values"]) # Shape (n_states, n_actions)
        
        # Get the MIRA landscape for comparison
        if sim["hp"]["dataset"] == "Mira (E. Coli)":
            mira_data = define_mira_landscapes()  # Shape (15 drugs, 16 genotypes)
            # Transpose to match Q-values orientation (genotypes x drugs)
            mira_landscape = mira_data.T  # Now (16 genotypes, 15 drugs)
            
            st.subheader("Q-Value vs MIRA Landscape Comparison")
            
            # Calculate correlation
            from scipy.stats import pearsonr
            # Flatten both arrays for correlation
            q_flat = q_values.flatten()
            mira_flat = mira_landscape.flatten()
            correlation, p_value = pearsonr(q_flat, mira_flat)
            
            # Display correlation metric prominently
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Q-MIRA Correlation", f"{correlation:.3f}")
            with col2:
                st.metric("P-value", f"{p_value:.2e}")
            with col3:
                # Interpretation
                if abs(correlation) > 0.7:
                    st.metric("Convergence", "Strong", delta="✓")
                elif abs(correlation) > 0.4:
                    st.metric("Convergence", "Moderate", delta="~")
                else:
                    st.metric("Convergence", "Weak", delta="✗")
            
            # Side-by-side heatmaps
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # MIRA Landscape
            im1 = axes[0].imshow(mira_landscape.T, aspect='auto', cmap="viridis")
            fig.colorbar(im1, ax=axes[0], label="Fitness")
            axes[0].set_title("True MIRA Landscape")
            axes[0].set_xticks(range(data["n_states"]))
            axes[0].set_xticklabels([bin(i)[2:].zfill(int(np.log2(data["n_states"]))) for i in range(data["n_states"])], rotation=45)
            axes[0].set_ylabel("Drugs")
            axes[0].set_xlabel("Genotypes")
            
            # Learned Q-values
            im2 = axes[1].imshow(q_values.T, aspect='auto', cmap="magma")
            fig.colorbar(im2, ax=axes[1], label="Q-Value")
            axes[1].set_title("Learned Q-Values")
            axes[1].set_xticks(range(data["n_states"]))
            axes[1].set_xticklabels([bin(i)[2:].zfill(int(np.log2(data["n_states"]))) for i in range(data["n_states"])], rotation=45)
            axes[1].set_ylabel("Actions (Drugs)")
            axes[1].set_xlabel("Genotypes")
            
            # Difference map (Q - Fitness)
            # Normalize both to 0-1 range first for fair comparison
            mira_norm = (mira_landscape - mira_landscape.min()) / (mira_landscape.max() - mira_landscape.min())
            q_norm = (q_values - q_values.min()) / (q_values.max() - q_values.min())
            diff = q_norm - mira_norm
            
            im3 = axes[2].imshow(diff.T, aspect='auto', cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(im3, ax=axes[2], label="Difference (Normalized)")
            axes[2].set_title("Difference Map (Q - MIRA)")
            axes[2].set_xticks(range(data["n_states"]))
            axes[2].set_xticklabels([bin(i)[2:].zfill(int(np.log2(data["n_states"]))) for i in range(data["n_states"])], rotation=45)
            axes[2].set_ylabel("Actions (Drugs)")
            axes[2].set_xlabel("Genotypes")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.caption("💡 High correlation means the agent is learning the true fitness structure. Divergence may indicate strategic optimization beyond immediate fitness.")
        else:
            # For synthetic landscapes, just show Q-values
            st.subheader("Learned Value Landscape (Live)")
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(q_values.T, aspect='auto', cmap="magma")
            fig.colorbar(im, ax=ax, label="Q-Value / Advantage")
            
            ax.set_xticks(range(data["n_states"]))
            ax.set_xticklabels([bin(i)[2:].zfill(int(np.log2(data["n_states"]))) for i in range(data["n_states"])], rotation=45)
            
            ax.set_xlabel("Genotypes")
            ax.set_ylabel("Actions (Drugs)")
            ax.set_title("Agent's Internal Model of the Landscape")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading policy snapshot: {e}")


def plot_training_progress(tab_id):
    sim = st.session_state.sims[tab_id]
    signature = sim.get("signature")
    if not signature and not sim["train"] and sim.get("selected_policy"):
        # Try to extract signature from policy filename
        # best_policy_sswm_testing.pth -> testing
        fname = sim["selected_policy"]
        if "sswm_" in fname:
            signature = fname.split("sswm_")[-1].replace(".pth", "")
        elif "policy_" in fname:
            signature = fname.split("policy_")[-1].replace(".pth", "")

    if not signature:
        st.info("No signature available to load training metrics.")
        return

    metrics_file = project_root / "log" / "metrics" / f"{signature}.csv"
    corr_file = project_root / "log" / "metrics" / f"{signature}_correlation.csv"
    
    # Rewards plot
    if metrics_file.exists():
        try:
            df = pd.read_csv(metrics_file)
            if not df.empty:
                st.subheader("Training Progress")
                
                # Reward plot
                plot_df = pd.DataFrame({
                    "Epoch": df["epoch"],
                    "Mean Reward": df["mean_reward"],
                    "Upper Bound": df["mean_reward"] + df["std_reward"],
                    "Lower Bound": df["mean_reward"] - df["std_reward"]
                }).set_index("Epoch")
                
                st.line_chart(plot_df, color=["#1f77b4", "#aec7e8", "#aec7e8"])
                st.caption("Blue: Mean Reward | Shaded: ±1 Std Dev")
                
                # Loss plot (if available)
                if "loss" in df.columns and df["loss"].notna().any():
                    st.subheader("Training Loss")
                    loss_df = df[["epoch", "loss"]].dropna().set_index("epoch")
                    if not loss_df.empty:
                        st.line_chart(loss_df, color="#ff7f0e")
                        st.caption("Orange: Training Loss (lower is better)")
            else:
                st.info("Metrics file is empty.")
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
    else:
        if sim["running"]:
            st.info("Waiting for first training metrics...")
        else:
            st.info(f"No metrics found for signature: {signature}")
    
    # Correlation plot (for MIRA only)
    if sim["hp"]["dataset"] == "Mira (E. Coli)" and corr_file.exists():
        try:
            corr_df = pd.read_csv(corr_file)
            if not corr_df.empty:
                st.subheader("Q-MIRA Correlation Over Time")
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(corr_df["epoch"], corr_df["correlation"], 'o-', color='#ff7f0e')
                ax.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Strong (0.7)')
                ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.4)')
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Correlation")
                ax.set_ylim([-1, 1])
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                st.caption("Tracking how well Q-values align with true MIRA fitness over training")
        except Exception as e:
            pass  # Silently skip if correlation data isn't available yet


@st.dialog("📜 Execution Logs", width="large")
def show_logs(tab_id):
    # Nested fragment to update logs without closing the dialog
    @st.fragment(run_every=1)
    def log_viewer():
        # Update logs from process if running
        update_logs_for_sim(tab_id)
        sim = st.session_state.sims[tab_id]
        st.code(sim["logs"] if sim["logs"] else "No logs available.", language="text")
        if sim["running"]:
            st.caption("🔄 Auto-updating logs...")
    log_viewer()

def render_status_logic(tab_id):
    @st.fragment(run_every=2)
    def status_fragment():
        update_logs_for_sim(tab_id)
        sim = st.session_state.sims[tab_id]
        
        # Status indicator at top
        st.divider()
        if sim["running"]:
            st.success(f"🔄 Running (PID: {sim['process'].pid})")
        elif sim["exit_code"] is not None:
            if sim["exit_code"] == 0: st.success("✅ Finished Successfully")
            else: st.error(f"❌ Failed (Exit Code: {sim['exit_code']})")
        else:
            st.info("⏸️ Idle - Configure and click RUN to start")
        
        st.divider()
        
        # Visualization Section
        if sim["train"] or sim.get("selected_policy"):
             if sim["train"]:
                plot_live_policy(tab_id)
                st.divider()
             plot_training_progress(tab_id)
             st.divider()
             
             if not sim["train"]:
                plot_live_trajectory(tab_id)
                st.divider()
    status_fragment()

# Header
st.title("🔬 EvoDM Playground")
st.markdown("Tinker with Evolutionary Dynamics and Reinforcement Learning right in your browser.")

# UI Layout
tab_labels = [f"Simulation {i+1}" for i in range(NUM_TABS)]
tabs = st.tabs(tab_labels)

def render_tab_content(tab_id):
    sim = st.session_state.sims[tab_id]
    
    # Top Control Bar
    with st.container():
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        cols_top = st.columns([1.5, 1, 1, 1, 1, 1, 1])
        
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
            if st.button("📜 LOGS", key=f"logs_btn_{tab_id}", use_container_width=True):
                show_logs(tab_id)

        with cols_top[6]:
            if st.button("🗑️ CLEAR", key=f"clear_tab_btn_{tab_id}", use_container_width=True):
                sim["logs"] = ""
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Main Area
    col_env, col_regime = st.columns([1, 1])
    
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
        st.subheader("Dataset & Regime")
        
        # Dataset Selection with Filtering Logic
        available_datasets = [name for name, meta in DATASETS.items() if meta["N"] == "any" or meta["N"] == sim["hp"]["n_mut"]]
        
        sim["hp"]["dataset"] = st.selectbox(
            "Select Dataset", 
            available_datasets, 
            index=0 if sim["hp"]["dataset"] not in available_datasets else available_datasets.index(sim["hp"]["dataset"]),
            key=f"dataset_sel_{tab_id}",
            help=DATASETS[sim["hp"]["dataset"]]["description"] if sim["hp"]["dataset"] in DATASETS else ""
        )
        
        if sim["hp"]["dataset"] == "Mira (E. Coli)" and sim["hp"]["n_mut"] != 4:
            st.warning("Mira dataset is strictly for N=4. Please adjust the mutation slider.")
            
        if MODES[sim["mode"]] in ["wf_ls", "wf_ss"]:
            sim["hp"]["pop_size"] = st.number_input("Population Size", 100, 1000000, sim["hp"]["pop_size"], key=f"pop_{tab_id}")
            sim["hp"]["mutation_rate"] = st.number_input("Mutation Rate", 0.0, 1.0, sim["hp"]["mutation_rate"], format="%.1e", key=f"mut_{tab_id}")
            sim["hp"]["gen_per_step"] = st.number_input("Gens per Step", 1, 10000, sim["hp"]["gen_per_step"], key=f"gps_{tab_id}")
        else:
            st.info("No specific parameters for this regime.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Status indicator
    render_status_logic(tab_id)
    
    # Visualizations removed from here - they're now in render_status_logic



for i, tab in enumerate(tabs):
    with tab:
        render_tab_content(i)

st.divider()
st.caption("EvoDM - Evolutionary Dynamics Modeling with RL | support for concurrent executions")
