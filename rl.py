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
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
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
# Automatically detects Kaggle vs local environment.
# ---------------------------------------------------------------------------
ON_KAGGLE = os.path.exists("/kaggle")

if ON_KAGGLE:
    _working_sumo = "/kaggle/working/sumo_files"
    os.makedirs(_working_sumo, exist_ok=True)
    
    # If /kaggle/working/sumo_files is empty, search /kaggle/input for SUMO config
    if not any(f.endswith(".sumocfg") for f in os.listdir(_working_sumo)):
        import glob, shutil
        cfg_candidates = glob.glob("/kaggle/input/**/bougara.sumocfg", recursive=True)
        if cfg_candidates:
            source_dir = os.path.dirname(cfg_candidates[0])
            for _f in os.listdir(source_dir):
                if _f.endswith((".xml", ".sumocfg")):
                    shutil.copy(os.path.join(source_dir, _f), _working_sumo)

    # Patch bougara.add.xml for Linux SUMO if necessary
    _add_xml = os.path.join(_working_sumo, "bougara.add.xml")
    if os.path.exists(_add_xml):
        with open(_add_xml, "r") as _fh:
            _content = _fh.read()
        if 'period="30"' in _content:
            _content = _content.replace('period="30"', 'freq="30"')
            with open(_add_xml, "w") as _fh:
                _fh.write(_content)

    SUMO_DIR = _working_sumo
else:
    _script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    SUMO_DIR = os.path.join(_script_dir, "sumo_files")

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
DELTA_TIME     = 5   # decision interval: seconds to maintain green when keeping phase

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

# State: [queue_len (9), avg_wait (9), mean_speed (9), current_phase_JM (1), phase_elapsed (1)]
OBS_DIM = N_LANES * 3 + 2

# ---------------------------------------------------------------------------
# SUMO Environment
# ---------------------------------------------------------------------------

class BougaraIntersectionEnv(gym.Env):
    """
    Custom Gymnasium environment for the Bougara El Biar intersection.

    Observation space:
        - Normalised queue length per lane (vehicles, capped at 30)
        - Normalised mean waiting time per lane (seconds, capped at 300s)
        - Normalised mean vehicle speed per lane (speed / max_speed, [0.0, 1.0])
        - Current green phase index of J_main_sec (0 or 1, normalised)
        - Time elapsed in current green phase (normalised 0-1 over MIN_GREEN_TIME+)

    Action space:
        Discrete(2):
            0 = Prioritize Main road green
            1 = Prioritize Secondary road green

        Macro-Action Execution:
            - If action matches current green: extends green by DELTA_TIME (5s).
            - If action requests a phase switch: runs YELLOW_DURATION (3s) yellow,
              switches traffic lights, and guarantees MIN_GREEN_TIME (10s) of green.
            Every action selected by the agent ALWAYS takes effect immediately and
            reliably, completely eliminating ignored actions or policy confusion.
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
        self._sim_seconds = 0
        self._current_phase_jm = MAIN_GREEN_PHASE
        self._phase_elapsed = MIN_GREEN_TIME
        self._episode_reward = 0.0
        self._episode_rewards = []       # track per-episode total reward

    # ------------------------------------------------------------------
    def _start_sumo(self):
        """Launch SUMO via TraCI."""
        if self._sumo_running:
            traci.close()
            self._sumo_running = False

        # Defensively close any lingering connection from a previous env instance
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
            "--no-warnings", "true",
            "--log", os.devnull,
            "--error-log", os.devnull,
            "--random",               # randomise departure times each episode
        ]
        if self.use_gui:
            sumo_cmd.extend(["--start", "true", "--delay", "60"])
            gui_cfg = os.path.join(SUMO_DIR, "gui-settings.xml")
            if os.path.exists(gui_cfg):
                sumo_cmd.extend(["--gui-settings-file", gui_cfg])
        traci.start(sumo_cmd)
        self._sumo_running = True

        # Set initial phases programmatically
        traci.trafficlight.setPhase("J_main_sec", MAIN_GREEN_PHASE)
        traci.trafficlight.setPhase("J_third",     THIRD_NB_PHASE)

    # ------------------------------------------------------------------
    def _get_lane_info(self, lane_id: str):
        """Return (queue_len, mean_wait, speed_norm) for a lane."""
        try:
            queue_len = traci.lane.getLastStepHaltingNumber(lane_id)
            wait_time = traci.lane.getWaitingTime(lane_id)
            veh_count = traci.lane.getLastStepVehicleNumber(lane_id)
            mean_wait = wait_time / max(veh_count, 1)

            # Normalized speed: ratio of current mean speed to lane speed limit [0.0 - 1.0]
            max_speed = traci.lane.getMaxSpeed(lane_id)
            if veh_count > 0:
                raw_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                speed_norm = float(np.clip(raw_speed / max(max_speed, 0.1), 0.0, 1.0))
            else:
                speed_norm = 1.0  # Empty lane has no bottleneck -> treated as free-flowing
        except traci.exceptions.TraCIException:
            queue_len = 0
            mean_wait = 0.0
            speed_norm = 1.0
        return queue_len, mean_wait, speed_norm

    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        """Build the normalised observation vector."""
        queue_norms = []
        wait_norms  = []
        speed_norms = []

        for lane in ALL_LANES:
            q, w, s = self._get_lane_info(lane)
            queue_norms.append(min(q,  30)  / 30.0)    # normalise queue 0-30
            wait_norms.append(min(w, 300.0) / 300.0)   # normalise wait 0-300s
            speed_norms.append(s)                      # normalise speed 0.0-1.0

        # Current phase and elapsed phase time
        phase_norm   = self._current_phase_jm / max(SEC_GREEN_PHASE, 1)
        elapsed_norm = min(self._phase_elapsed, 120) / 120.0

        obs = np.array(queue_norms + wait_norms + speed_norms + [phase_norm, elapsed_norm],
                       dtype=np.float32)
        return obs

    # ------------------------------------------------------------------
    def _compute_reward(self) -> float:
        """
        Comprehensive traffic reward combining:
        1. Queued / halting vehicles (queue penalty)
        2. Total delay & waiting time across vehicles
        3. Speed deficit (penalizing slow crawling cars when lanes are loaded)
        4. Starvation penalty for long-waiting side roads
        """
        total_halting = 0
        total_wait = 0.0
        max_lane_wait = 0.0
        speed_deficit = 0.0

        for lane in ALL_LANES:
            q, w, s = self._get_lane_info(lane)
            total_halting += q
            total_wait += w
            max_lane_wait = max(max_lane_wait, w)

            try:
                veh_count = traci.lane.getLastStepVehicleNumber(lane)
            except Exception:
                veh_count = 0
            if veh_count > 0:
                # If cars are on this lane, penalize how far their speed is below speed limit
                speed_deficit += (1.0 - s) * veh_count

        # Starvation penalty (fairness: prevent minor road vehicles waiting > 30s)
        wait_penalty = (max_lane_wait / 20.0) ** 2

        # Reward formulation (negative cost: closer to 0 is better)
        reward = -(
            1.0 * total_halting + 
            0.05 * total_wait + 
            0.5 * speed_deficit + 
            wait_penalty
        )
        return float(reward)

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
        self._sim_seconds = 0
        self._current_phase_jm = MAIN_GREEN_PHASE
        self._phase_elapsed = MIN_GREEN_TIME
        self._episode_reward = 0.0

        obs = self._get_observation()
        info = {}
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Execute one RL decision step:
        - If action matches current green: extends green by DELTA_TIME (5 seconds).
        - If action requests a switch: executes YELLOW_DURATION (3s) yellow, switches,
          and simulates MIN_GREEN_TIME (10s) green on the new approach.
        Control returns to the agent only when the next legitimate decision can be made.
        Zero actions are ignored; zero 'WTF' moments for the policy.
        """
        target_phase = MAIN_GREEN_PHASE if action == 0 else SEC_GREEN_PHASE
        step_rewards = []

        if target_phase == self._current_phase_jm:
            # Maintain current green phase for DELTA_TIME (5 seconds)
            for _ in range(DELTA_TIME):
                traci.simulationStep()
                self._sim_seconds += 1
                self._phase_elapsed += 1
                step_rewards.append(self._compute_reward())
                if self._sim_seconds >= EPISODE_DURATION:
                    break
        else:
            # Phase Switch:
            # 1. Yellow transition phase (3 seconds)
            yellow_phase = self._current_phase_jm + 1
            traci.trafficlight.setPhase("J_main_sec", yellow_phase)
            for _ in range(YELLOW_DURATION):
                traci.simulationStep()
                self._sim_seconds += 1
                step_rewards.append(self._compute_reward())
                if self._sim_seconds >= EPISODE_DURATION:
                    break

            # 2. Switch to target green phase
            traci.trafficlight.setPhase("J_main_sec", target_phase)
            self._current_phase_jm = target_phase
            self._phase_elapsed = 0
            self._set_jthird_phase(target_phase)

            # 3. Mandatory Minimum Green duration (10 seconds)
            for _ in range(MIN_GREEN_TIME):
                traci.simulationStep()
                self._sim_seconds += 1
                self._phase_elapsed += 1
                step_rewards.append(self._compute_reward())
                if self._sim_seconds >= EPISODE_DURATION:
                    break

        reward = float(np.mean(step_rewards)) if step_rewards else 0.0
        self._episode_reward += reward

        obs = self._get_observation()
        terminated = self._sim_seconds >= EPISODE_DURATION
        truncated = False

        if terminated:
            self._episode_rewards.append(self._episode_reward)

        info = {
            "sim_seconds": self._sim_seconds,
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
# Normalization & Evaluation Callbacks
# ---------------------------------------------------------------------------

class SaveVecNormalizeCallback(BaseCallback):
    """
    Saves the VecNormalize statistics synchronously alongside model checkpoints.
    Also continuously saves 'vec_normalize.pkl' so evaluate() always finds it,
    even if training is interrupted early (e.g. laptop closed or timeout).
    """
    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if isinstance(self.training_env, VecNormalize):
                step_file = os.path.join(self.save_path, f"vec_normalize_{self.num_timesteps}_steps.pkl")
                latest_file = os.path.join(self.save_path, "vec_normalize.pkl")
                self.training_env.save(step_file)
                self.training_env.save(latest_file)
                if self.verbose > 0:
                    print(f"[INFO] Normalization stats saved to {latest_file}")
        return True


class SyncEvalCallback(EvalCallback):
    """
    Subclass of EvalCallback that:
    1. Synchronizes running observation statistics from the training environment
       into the evaluation environment before running evaluation.
    2. Whenever a new best model is saved, automatically saves the matching
       'vec_normalize.pkl' directly into the 'best/' folder alongside best_model.zip.
    """
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sync_envs_normalization(self.training_env, self.eval_env)

        continue_training = super()._on_step()

        # Save normalization stats alongside the best model
        if self.best_model_save_path is not None and hasattr(self, "best_mean_reward"):
            if self.last_mean_reward == self.best_mean_reward:
                best_norm_path = os.path.join(self.best_model_save_path, "vec_normalize.pkl")
                if isinstance(self.training_env, VecNormalize):
                    self.training_env.save(best_norm_path)

        return continue_training


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
        total_timesteps : Maximum environment steps to train for (acts as upper bound).
        n_eval_episodes : Episodes used for periodic evaluation.
        save_dir        : Directory to save model checkpoints.
        log_dir         : Directory for TensorBoard logs.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    print("=" * 60)
    print("  Bougara El Biar Intersection – RL Training")
    print("  Algorithm : PPO with LSTM (RecurrentPPO)")
    print(f"  Max Steps : {total_timesteps:,}")
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

    norm_save_cb = SaveVecNormalizeCallback(
        save_freq   = 25_000,
        save_path   = save_dir,
        verbose     = 1,
    )

    eval_env = BougaraIntersectionEnv(use_gui=False)
    eval_env = Monitor(eval_env)
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)

    # Early Stopping Callback: stops training if the model stops improving across 10 consecutive evaluations
    # (10 * 25,000 = 250,000 steps without improvement). Provides ample exploration breathing room!
    stop_train_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals = 10,
        min_evals                = 5,
        verbose                  = 1,
    )

    eval_cb = SyncEvalCallback(
        eval_vec_env,
        best_model_save_path = os.path.join(save_dir, "best"),
        log_path             = log_dir,
        eval_freq            = 25_000,
        n_eval_episodes      = n_eval_episodes,
        callback_after_eval  = stop_train_cb,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    # --- Train ---
    print("[INFO] Starting training with Early Stopping active...\n")
    start_time = time.time()

    model.learn(
        total_timesteps  = total_timesteps,
        callback         = [checkpoint_cb, norm_save_cb, eval_cb],
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
    print(f"[INFO] Final normalization stats saved to: {save_dir}/vec_normalize.pkl")

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
    
    # If generic vec_normalize.pkl is missing, check for step-specific files
    if not os.path.exists(vec_norm_path):
        import glob
        candidates = glob.glob(os.path.join(save_dir, "vec_normalize*.pkl"))
        if candidates:
            vec_norm_path = sorted(candidates)[-1]
            print(f"[INFO] Found checkpoint normalization file: {vec_norm_path}")

    if os.path.exists(vec_norm_path):
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"[INFO] Normalization statistics loaded successfully from: {vec_norm_path}")
    else:
        print(f"\n[WARNING] No 'vec_normalize.pkl' found in '{save_dir}'!")
        print("[WARNING] The model was trained with normalized inputs. Evaluating without normalization")
        print("[WARNING] may lead to degraded or erratic behavior.\n")

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
