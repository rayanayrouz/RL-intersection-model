import os
import sys
import time
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

# Set SUMO paths
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
sumo_tools = os.path.join(os.environ["SUMO_HOME"], "tools")
if sumo_tools not in sys.path:
    sys.path.append(sumo_tools)

import traci
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
import rl
import traffic_generator

def watch_model(
    model_path: str = "models/bougara_lstm_100000_steps.zip",
    norm_path: str = "models/vec_normalize_100000_steps.pkl",
    scenario: str = "rush_hour_main",
    duration_seconds: int = 1800,
    delay_ms: int = 60,
):
    """
    Launch SUMO GUI and watch the trained RL agent control the Bougara intersection live!
    
    Args:
        model_path: Path to the trained model zip file.
        norm_path: Path to the matching VecNormalize pickle file.
        scenario: Traffic scenario ('uniform', 'rush_hour_main', 'rush_hour_sec', 'asymmetric_surge', 'platoon_bursts').
        duration_seconds: How long to run the visual simulation (default: 1800s / 30 mins).
        delay_ms: Delay in milliseconds per simulation step (e.g., 50-80ms makes traffic watchable at realistic speed).
    """
    if not os.path.isabs(model_path):
        model_path = os.path.join(WORKSPACE_DIR, model_path)
    if not os.path.isabs(norm_path):
        norm_path = os.path.join(WORKSPACE_DIR, norm_path)

    sumo_dir = os.path.join(WORKSPACE_DIR, "sumo_files")
    sumo_cfg = os.path.join(sumo_dir, "bougara.sumocfg")

    print("=" * 70)
    print("  LIVE SUMO VISUALIZATION – BOUARA INTERSECTION (RL AGENT)")
    print(f"  Scenario : {scenario.upper()}")
    print(f"  Model    : {os.path.basename(model_path)}")
    print(f"  Step Lag : {delay_ms} ms (smooth human viewing speed)")
    print("=" * 70)

    # 1. Generate the chosen traffic scenario
    print(f"[INFO] Generating traffic for scenario: {scenario}...")
    traffic_generator.generate_traffic(sumo_dir, scenario=scenario)

    # 2. Check model and normalization files
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return
    if not os.path.exists(norm_path):
        print(f"[WARNING] Normalization file not found: {norm_path}. Using unnormalized inputs.")

    # 3. Load Model
    print("[INFO] Loading trained RecurrentPPO model into memory...")
    model = RecurrentPPO.load(model_path, device="cpu")

    # 4. Start SUMO-GUI
    print("[INFO] Opening SUMO-GUI desktop window...")
    gui_cfg = os.path.join(sumo_dir, "gui-settings.xml")
    sumo_cmd = [
        "sumo-gui",
        "-c", sumo_cfg,
        "--start", "true",
        "--delay", str(delay_ms),
        "--no-warnings", "true",
        "--no-step-log", "true",
    ]
    if os.path.exists(gui_cfg):
        sumo_cmd.extend(["--gui-settings-file", gui_cfg])

    traci.start(sumo_cmd)

    # Initial light phases
    traci.trafficlight.setPhase("J_main_sec", rl.MAIN_GREEN_PHASE)
    traci.trafficlight.setPhase("J_third",     rl.THIRD_NB_PHASE)

    # Load Normalization Stats
    obs_rms = None
    if os.path.exists(norm_path):
        import pickle
        with open(norm_path, "rb") as f:
            norm_data = pickle.load(f)
        obs_rms = norm_data.obs_rms
        print("[INFO] Input normalization statistics loaded successfully!")

    def get_obs(current_phase, phase_elapsed):
        queue_norms, wait_norms, speed_norms = [], [], []
        for lane in rl.ALL_LANES:
            try:
                q = traci.lane.getLastStepHaltingNumber(lane)
                w = traci.lane.getWaitingTime(lane)
                v = traci.lane.getLastStepVehicleNumber(lane)
                mean_w = w / max(v, 1)
                max_s = traci.lane.getMaxSpeed(lane)
                raw_s = traci.lane.getLastStepMeanSpeed(lane) if v > 0 else max_s
                s = float(np.clip(raw_s / max(max_s, 0.1), 0.0, 1.0))
            except Exception:
                q, mean_w, s = 0, 0.0, 1.0
            queue_norms.append(min(q, 30) / 30.0)
            wait_norms.append(min(mean_w, 300.0) / 300.0)
            speed_norms.append(s)
        p_norm = current_phase / max(rl.SEC_GREEN_PHASE, 1)
        el_norm = min(phase_elapsed, 120) / 120.0
        raw_obs = np.array(queue_norms + wait_norms + speed_norms + [p_norm, el_norm], dtype=np.float32)
        if obs_rms is not None:
            raw_obs = np.clip((raw_obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)
        return raw_obs

    sim_seconds = 0
    current_phase = rl.MAIN_GREEN_PHASE
    phase_elapsed = rl.MIN_GREEN_TIME
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    print("\n[LIVE] Simulation running! Look at your taskbar for the SUMO-GUI window.\n")
    print(f"{'SIM TIME':<10} | {'CURRENT GREEN':<18} | {'ACTION TAKEN':<24} | {'QUEUE':<8} | {'SPEED':<8}")
    print("-" * 75)

    try:
        while sim_seconds < duration_seconds:
            obs = get_obs(current_phase, phase_elapsed)
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            episode_start = np.zeros((1,), dtype=bool)
            act = int(action)

            target_phase = rl.MAIN_GREEN_PHASE if act == 0 else rl.SEC_GREEN_PHASE
            action_desc = "KEEP GREEN (+5s)" if target_phase == current_phase else "SWITCH PHASE (Yellow+MinGreen)"

            # Print live dashboard update
            total_q = sum(traci.lane.getLastStepHaltingNumber(l) for l in rl.ALL_LANES)
            spds = [traci.lane.getLastStepMeanSpeed(l) / max(traci.lane.getMaxSpeed(l), 0.1) 
                    for l in rl.ALL_LANES if traci.lane.getLastStepVehicleNumber(l) > 0]
            avg_spd = np.mean(spds) * 100 if spds else 100.0
            phase_name = "MAIN ROAD" if current_phase == rl.MAIN_GREEN_PHASE else "SECONDARY ROAD"

            print(f"{sim_seconds:5d}s      | {phase_name:<18} | {action_desc:<24} | {total_q:3d} cars  | {avg_spd:4.1f}%")

            if target_phase == current_phase:
                for _ in range(rl.DELTA_TIME):
                    traci.simulationStep()
                    sim_seconds += 1
                    phase_elapsed += 1
                    if sim_seconds >= duration_seconds:
                        break
            else:
                # Yellow (3s)
                yellow = current_phase + 1
                traci.trafficlight.setPhase("J_main_sec", yellow)
                for _ in range(rl.YELLOW_DURATION):
                    traci.simulationStep()
                    sim_seconds += 1
                    if sim_seconds >= duration_seconds:
                        break
                # Green (10s min green)
                traci.trafficlight.setPhase("J_main_sec", target_phase)
                current_phase = target_phase
                phase_elapsed = 0
                if current_phase == rl.MAIN_GREEN_PHASE:
                    traci.trafficlight.setPhase("J_third", rl.THIRD_NB_PHASE)
                else:
                    traci.trafficlight.setPhase("J_third", 2)
                for _ in range(rl.MIN_GREEN_TIME):
                    traci.simulationStep()
                    sim_seconds += 1
                    phase_elapsed += 1
                    if sim_seconds >= duration_seconds:
                        break

    except traci.exceptions.FatalTraCIError:
        print("\n[INFO] SUMO GUI window was closed by user.")
    finally:
        try:
            traci.close()
        except Exception:
            pass
        print("\n[INFO] Live simulation session finished.\n")

if __name__ == "__main__":
    watch_model()
