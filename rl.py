"""
RL Traffic Light Control – Bougara El Biar Intersection, Alger, Algeria
=======================================================================
This script trains a Proximal Policy Optimization (PPO) agent using
stable-baselines3 to control the traffic lights at a real-world inspired
4-arm intersection simulated in SUMO.

Intersection structure:
  - Main road (bidirectional, 2 lanes each direction): North-South axis
  - Secondary road (bidirectional, 1 lane): joins from East (right side)
  - Third road: exit for Northbound only, going West (slightly further south)
  - 3 pedestrian crossings: at secondary road start, middle of main road, before third road

Junctions controlled:
  - J_main_sec : Main x Secondary junction (TL with 2 green phases)
  - J_third    : Third road exit junction (TL with 1 green phase)

RL Formulation:
  State  : queue lengths + waiting times on each approach lane + current phases
  Action : choose next green phase index for J_main_sec (agent controls main junction;
           J_third follows a coordinated fixed offset to avoid conflicts)
  Reward : negative total cumulative waiting time across all vehicles (lower is better)

Dependencies:
  pip install stable-baselines3[extra] gymnasium traci
  SUMO must be installed and SUMO_HOME must be set.
"""

import os
import sys
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
# pyrefly: ignore [missing-import]
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
# pyrefly: ignore [missing-import]
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import traffic_generator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the SUMO binary (sumo for headless, sumo-gui for visual)
SUMO_BINARY = "sumo"  # Change to "sumo-gui" to watch the simulation

# ---------------------------------------------------------------------------
# PATH CONFIGURATION
# Set KAGGLE_DATASET_NAME to your Kaggle dataset slug if running on Kaggle.
# Example: if your dataset URL is kaggle.com/datasets/yourname/bougara-sumo
#          then set KAGGLE_DATASET_NAME = "bougara-sumo"
# Leave as None to use the local sumo_files/ folder.
# ---------------------------------------------------------------------------
KAGGLE_DATASET_NAME = os.environ.get("KAGGLE_DATASET_NAME", "rldatasett")   # ← CHANGE THIS on Kaggle, e.g. "bougara-sumo"

ON_KAGGLE = os.path.exists("/kaggle/input")

if ON_KAGGLE and KAGGLE_DATASET_NAME:
    SUMO_DIR = f"/kaggle/input/datasets/rayantribeche/{KAGGLE_DATASET_NAME}"
else:
    _script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    SUMO_DIR = os.path.join(_script_dir, "sumo_files")

# ---------------------------------------------------------------------------
# On Kaggle: copy SUMO files to a writable directory and patch add.xml.
# The Kaggle SUMO version requires 'freq' instead of 'period' in detectors.
# ---------------------------------------------------------------------------
if ON_KAGGLE:
    import shutil
    _working_sumo = "/kaggle/working/sumo_files"
    os.makedirs(_working_sumo, exist_ok=True)
    for _f in os.listdir(SUMO_DIR):
        shutil.copy(os.path.join(SUMO_DIR, _f), _working_sumo)
    # Patch bougara.add.xml: replace 'period' with 'freq'
    _add_xml = os.path.join(_working_sumo, "bougara.add.xml")
    with open(_add_xml, "r") as _fh:
        _content = _fh.read()
    _content = _content.replace('period="30"', 'freq="30"')
    with open(_add_xml, "w") as _fh:
        _fh.write(_content)
    SUMO_DIR = _working_sumo
    print(f"[INFO] SUMO files copied and patched → {_working_sumo}")

SUMO_CFG = os.path.join(SUMO_DIR, "bougara.sumocfg")


# Simulation parameters
SIM_STEP_LENGTH = 1        # seconds per simulation step
EPISODE_DURATION = 3600    # seconds per episode (1 hour of traffic)
YELLOW_DURATION  = 3       # seconds for yellow phase transition

# RL agent controls J_main_sec phases:
#   Phase 0: Main road (NB + SB) green  → state "GgGrrGGG"
#   Phase 2: Secondary road green       → state "rrrGGGrr"
# (odd phases are yellow transitions, handled automatically)
MAIN_GREEN_PHASE = 0   # index in J_main_sec tlLogic
SEC_GREEN_PHASE  = 2   # index in J_main_sec tlLogic

# J_third is coordinated: when main road is green, J_third also allows Northbound + Southbound
# When main road switches, J_third shortly switches too.
THIRD_NB_PHASE   = 0   # J_third phase: full NB+SB green "GGGgG"

# Minimum green time to prevent rapid switching (seconds)
MIN_GREEN_TIME = 10

# Approach lanes for state observation (in order)
APPROACH_LANES_JM = [
    "main_NB_pc_jm_0",   # NB lane 0 → J_main_sec
    "main_NB_pc_jm_1",   # NB lane 1 → J_main_sec
    "main_SB_pb_jm_0",   # SB lane 0 → J_main_sec
    "main_SB_pb_jm_1",   # SB lane 1 → J_main_sec
    "sec_in_2_0",         # Secondary inbound → J_main_sec
]

APPROACH_LANES_JT = [
    "main_NB_s_jt_0",    # NB lane 0 → J_third
    "main_NB_s_jt_1",    # NB lane 1 → J_third
    "main_SB_pc_jt_0",   # SB lane 0 → J_third
    "main_SB_pc_jt_1",   # SB lane 1 → J_third
]

ALL_LANES = APPROACH_LANES_JM + APPROACH_LANES_JT
N_LANES = len(ALL_LANES)  # 9 lanes total

# State: [queue_len per lane (9), avg_wait per lane (9), current_phase_JM (1), phase_elapsed (1)]
OBS_DIM = N_LANES * 2 + 2

# ---------------------------------------------------------------------------
# SUMO Environment
# ---------------------------------------------------------------------------

class BougaraIntersectionEnv(gym.Env):
    """
    Custom Gymnasium environment for the Bougara El Biar intersection.

    Observation space:
        - Normalised queue length per lane (vehicles, capped at 30)
        - Normalised mean waiting time per lane (seconds, capped at 300s)
        - Current green phase index of J_main_sec (0 or 1, normalised)
        - Time elapsed in current green phase (normalised 0-1 over MIN_GREEN_TIME+)

    Action space:
        Discrete(2): 0 = keep/set Main-road green, 1 = keep/set Secondary green
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, sumo_binary: str = SUMO_BINARY, use_gui: bool = False):
        super().__init__()
        self.sumo_binary = "sumo-gui" if use_gui else sumo_binary
        self.use_gui = use_gui
        self._sumo_running = False

        # Observation space (all values normalised to [0, 1])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        # Action space: 0=main road green, 1=secondary green
        self.action_space = spaces.Discrete(2)

        # Internal state
        self._step = 0
        self._current_phase_jm = MAIN_GREEN_PHASE
        self._phase_elapsed = 0          # steps in current phase
        self._in_yellow = False
        self._pending_phase = None       # phase to switch to after yellow
        self._yellow_elapsed = 0
        self._episode_reward = 0.0
        self._episode_rewards = []       # track per-episode total reward

    # ------------------------------------------------------------------
    def _start_sumo(self):
        """Launch SUMO via TraCI."""
        if self._sumo_running:
            traci.close()
            self._sumo_running = False

        # Defensively close any lingering connection from a previous env instance
        # (e.g. left open by check_env or a crashed episode)
        try:
            traci.close()
        except Exception:
            pass

        # Generate new random traffic scenario before starting
        traffic_generator.generate_traffic(SUMO_DIR, scenario="random")

        sumo_cmd = [
            self.sumo_binary,
            "-c", SUMO_CFG,
            "--no-step-log", "true",
            "--log", os.devnull,
            "--random",               # randomise departure times each episode
        ]
        traci.start(sumo_cmd)
        self._sumo_running = True

        # Set initial phases programmatically
        traci.trafficlight.setPhase("J_main_sec", MAIN_GREEN_PHASE)
        traci.trafficlight.setPhase("J_third",     THIRD_NB_PHASE)

    # ------------------------------------------------------------------
    def _get_lane_info(self, lane_id: str):
        """Return (queue_len, mean_wait) for a lane."""
        try:
            queue_len = traci.lane.getLastStepHaltingNumber(lane_id)
            wait_time = traci.lane.getWaitingTime(lane_id)
            # Mean waiting time: distribute over waiting vehicles
            veh_count = traci.lane.getLastStepVehicleNumber(lane_id)
            mean_wait = wait_time / max(veh_count, 1)
        except traci.exceptions.TraCIException:
            queue_len = 0
            mean_wait = 0.0
        return queue_len, mean_wait

    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        """Build the normalised observation vector."""
        queue_norms = []
        wait_norms  = []

        for lane in ALL_LANES:
            q, w = self._get_lane_info(lane)
            queue_norms.append(min(q,  30)  / 30.0)    # normalise queue 0-30
            wait_norms.append(min(w, 300.0) / 300.0)   # normalise wait 0-300s

        # Current phase and elapsed phase time
        phase_norm   = self._current_phase_jm / max(SEC_GREEN_PHASE, 1)
        elapsed_norm = min(self._phase_elapsed, 120) / 120.0

        obs = np.array(queue_norms + wait_norms + [phase_norm, elapsed_norm],
                       dtype=np.float32)
        return obs

    # ------------------------------------------------------------------
    def _compute_reward(self) -> float:
        """
        Reward = negative sum of halting vehicles across all approach lanes.
        A smaller number of halting vehicles is better → higher (less negative) reward.
        We also add a small fairness bonus if no lane is completely starved.
        """
        total_halting = 0
        max_lane_wait = 0.0

        for lane in ALL_LANES:
            q, w = self._get_lane_info(lane)
            total_halting += q
            max_lane_wait = max(max_lane_wait, w)

        # Exponential penalty for starvation (e.g. w=10 -> 1, w=60 -> 36, w=120 -> 144)
        wait_penalty = (max_lane_wait / 10.0) ** 2

        reward = -float(total_halting) - wait_penalty

        return reward

    # ------------------------------------------------------------------
    def _apply_action(self, action: int):
        """
        Apply RL action to J_main_sec.
        Handles yellow phase transition before switching green phase.
        J_third is coordinated: mirrors the main junction phase.
        """
        target_phase = MAIN_GREEN_PHASE if action == 0 else SEC_GREEN_PHASE
        is_valid = True

        if self._in_yellow:
            self._yellow_elapsed += 1
            if self._yellow_elapsed >= YELLOW_DURATION:
                traci.trafficlight.setPhase("J_main_sec", self._pending_phase)
                self._current_phase_jm = self._pending_phase
                self._in_yellow = False
                self._yellow_elapsed = 0
                self._phase_elapsed = 0
                self._set_jthird_phase(self._pending_phase)
            else:
                if target_phase != self._pending_phase:
                    is_valid = False
        else:
            self._phase_elapsed += 1
            if target_phase != self._current_phase_jm:
                if self._phase_elapsed >= MIN_GREEN_TIME:
                    yellow_phase = self._current_phase_jm + 1
                    traci.trafficlight.setPhase("J_main_sec", yellow_phase)
                    self._in_yellow = True
                    self._pending_phase = target_phase
                    self._yellow_elapsed = 0
                else:
                    is_valid = False

        return is_valid

    # ------------------------------------------------------------------
    def _set_jthird_phase(self, jm_phase: int):
        """
        Coordinate J_third with J_main_sec.
        When main road (NB+SB) is green → J_third full green (all traffic through + NB exit)
        When secondary is green → J_third red (brief all-red, phase 2 in auto-generated logic)
        """
        if jm_phase == MAIN_GREEN_PHASE:
            traci.trafficlight.setPhase("J_third", THIRD_NB_PHASE)   # full green
        else:
            # Use the all-red phase at J_third (index 2 from netconvert)
            traci.trafficlight.setPhase("J_third", 2)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._start_sumo()

        # Reset internal counters
        self._step           = 0
        self._current_phase_jm = MAIN_GREEN_PHASE
        self._phase_elapsed  = 0
        self._in_yellow      = False
        self._pending_phase  = None
        self._yellow_elapsed = 0
        self._episode_reward = 0.0

        obs = self._get_observation()
        info = {}
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action: int):
        """Execute one simulation step with the given action."""
        # Apply action (handles yellow/green transitions)
        is_valid = self._apply_action(action)

        # Advance simulation by one second
        traci.simulationStep()
        self._step += 1

        # Get new state
        obs    = self._get_observation()
        reward = self._compute_reward()
        
        # Penalize invalid action choice (agent trying to switch too fast)
        if not is_valid:
            reward -= 10.0

        self._episode_reward += reward

        # Episode ends when simulation time is up
        terminated = self._step >= EPISODE_DURATION
        truncated  = False

        if terminated:
            self._episode_rewards.append(self._episode_reward)

        info = {
            "step": self._step,
            "episode_reward": self._episode_reward,
            "current_phase": self._current_phase_jm,
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        """Rendering is handled by SUMO GUI (if use_gui=True)."""
        pass

    # ------------------------------------------------------------------
    def close(self):
        """Close the SUMO simulation."""
        if self._sumo_running:
            traci.close()
            self._sumo_running = False

    # ------------------------------------------------------------------
    def get_episode_rewards(self):
        """Return list of per-episode total rewards."""
        return self._episode_rewards


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    total_timesteps: int = 500_000,
    n_eval_episodes: int = 5,
    save_dir: str = "models",
    log_dir: str = "logs",
):
    """
    Train a PPO agent on the Bougara intersection.

    Args:
        total_timesteps : Total environment steps to train for.
        n_eval_episodes : Episodes used for periodic evaluation.
        save_dir        : Directory to save model checkpoints.
        log_dir         : Directory for TensorBoard logs.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    print("=" * 60)
    print("  Bougara El Biar Intersection – RL Training")
    print("  Algorithm : PPO (stable-baselines3)")
    print(f"  Timesteps : {total_timesteps:,}")
    print("=" * 60)

    # --- Training environment ---
    env = BougaraIntersectionEnv(use_gui=False)
    env = Monitor(env, filename=os.path.join(log_dir, "train_monitor"))
    vec_env = DummyVecEnv([lambda: env])
    train_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)


    # --- PPO model ---
    model = RecurrentPPO(
        policy           = "MlpLstmPolicy",
        env              = train_env,
        learning_rate    = 3e-4,
        n_steps          = 2048,
        batch_size       = 64,
        n_epochs         = 10,
        gamma            = 0.99,
        gae_lambda       = 0.95,
        clip_range       = 0.2,
        ent_coef         = 0.01,
        vf_coef          = 0.5,
        max_grad_norm    = 0.5,
        tensorboard_log  = log_dir,
        verbose          = 1,
        device           = "auto",
    )

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = save_dir,
        name_prefix = "bougara_lstm",
        verbose     = 1,
    )

    eval_env = BougaraIntersectionEnv(use_gui=False)
    eval_env = Monitor(eval_env)
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)
    eval_vec_env.obs_rms = train_env.obs_rms

    eval_cb  = EvalCallback(
        eval_vec_env,
        best_model_save_path = os.path.join(save_dir, "best"),
        log_path             = log_dir,
        eval_freq            = 50_000,
        n_eval_episodes      = n_eval_episodes,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    # --- Train ---
    print("[INFO] Starting training...\n")
    start_time = time.time()

    model.learn(
        total_timesteps  = total_timesteps,
        callback         = [checkpoint_cb, eval_cb],
        tb_log_name      = "PPO_bougara",
        progress_bar     = True,
    )

    elapsed = time.time() - start_time
    print(f"\n[INFO] Training complete in {elapsed/60:.1f} minutes.")

    # --- Save final model ---
    final_path = os.path.join(save_dir, "bougara_lstm_final")
    model.save(final_path)
    train_env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    print(f"[INFO] Final model saved to: {final_path}.zip")

    # --- Plot reward curve ---
    _plot_reward_curve(log_dir)

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_path: str, n_episodes: int = 5, use_gui: bool = False):
    """
    Evaluate a trained RecurrentPPO model.
    """
    print(f"\n[INFO] Loading model from: {model_path}")
    model = RecurrentPPO.load(model_path)

    env = BougaraIntersectionEnv(use_gui=use_gui)
    vec_env = DummyVecEnv([lambda: env])
    
    save_dir = os.path.dirname(model_path)
    vec_norm_path = os.path.join(save_dir, "vec_normalize.pkl")
    if os.path.exists(vec_norm_path):
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    episode_rewards = []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        total_reward = 0.0

        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, reward, done_vec, info = vec_env.step(action)
            episode_starts = done_vec
            done = done_vec[0]
            total_reward += reward[0]

        episode_rewards.append(total_reward)
        print(f"  Episode {ep+1}/{n_episodes} | Total Reward: {total_reward:.1f}")

    vec_env.close()

    mean_r = np.mean(episode_rewards)
    std_r  = np.std(episode_rewards)
    print(f"\n[EVAL] Mean reward: {mean_r:.1f} ± {std_r:.1f}")
    return episode_rewards


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_reward_curve(log_dir: str):
    """Parse monitor CSV and plot the training reward curve."""
    monitor_file = os.path.join(log_dir, "train_monitor.monitor.csv")
    if not os.path.exists(monitor_file):
        print("[WARN] Monitor file not found, skipping plot.")
        return

    rewards    = []
    timesteps  = []
    cumsteps   = 0

    with open(monitor_file, "r") as f:
        lines = f.readlines()
    # Skip header lines (first 2 lines are metadata)
    for line in lines[2:]:
        parts = line.strip().split(",")
        if len(parts) >= 3:
            try:
                r  = float(parts[0])
                ep_len = int(parts[1])
                rewards.append(r)
                cumsteps += ep_len
                timesteps.append(cumsteps)
            except ValueError:
                continue

    if not rewards:
        print("[WARN] No episode data found in monitor file.")
        return

    # Smoothed reward
    window = max(1, len(rewards) // 20)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timesteps, rewards, alpha=0.3, color="steelblue", label="Episode reward")
    ax.plot(
        timesteps[window - 1:],
        smoothed,
        color="orangered",
        linewidth=2,
        label=f"Smoothed (window={window})",
    )
    ax.set_xlabel("Training Timesteps", fontsize=13)
    ax.set_ylabel("Total Episode Reward", fontsize=13)
    ax.set_title(
        "PPO Training Reward – Bougara El Biar Intersection\n"
        "(Alger, Algeria | SUMO Simulation)",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(log_dir, "reward_curve.png")
    fig.savefig(out_path, dpi=150)
    print(f"[INFO] Reward curve saved to: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(total_timesteps=500_000)
