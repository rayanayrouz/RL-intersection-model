import os
import pandas as pd
import numpy as np
from evaluate_baselines import evaluate_baseline
from rl import BougaraIntersectionEnv, SUMO_DIR
import traffic_generator

def evaluate_rl(scenario="random", seed=42, model_path="models/best/best_model.zip"):
    print(f"Evaluating RL agent on {scenario} with seed {seed}")
    
    traffic_generator.generate_traffic(SUMO_DIR, scenario=scenario)
    
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
    except Exception as e:
        try:
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(model_path)
        except Exception as e2:
            from stable_baselines3 import RecurrentPPO
            model = RecurrentPPO.load(model_path)

    env = BougaraIntersectionEnv(use_gui=False)
    
    original_sumo_cmd = env._start_sumo
    def _patched_start_sumo():
        import traci
        if env._sumo_running:
            traci.close()
            env._sumo_running = False
        try:
            traci.close()
        except:
            pass
        from rl import SUMO_CFG
        sumo_cmd = [
            env.sumo_binary,
            "-c", SUMO_CFG,
            "--no-step-log", "true",
            "--log", os.devnull,
            "--seed", str(seed)
        ]
        traci.start(sumo_cmd)
        env._sumo_running = True
        from rl import MAIN_GREEN_PHASE, THIRD_NB_PHASE
        traci.trafficlight.setPhase("J_main_sec", MAIN_GREEN_PHASE)
        traci.trafficlight.setPhase("J_third", THIRD_NB_PHASE)
    
    env._start_sumo = _patched_start_sumo
    
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    vec_env = DummyVecEnv([lambda: env])
    vec_norm_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), "vec_normalize.pkl")
    if os.path.exists(vec_norm_path):
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        
    obs = vec_env.reset()
    done = False
    info = None
    
    is_lstm = hasattr(model, "policy") and "Recurrent" in type(model.policy).__name__
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    while not done:
        if is_lstm:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            episode_starts = done_vec if 'done_vec' in locals() else np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs, deterministic=True)
            
        obs, reward, done_vec, infos = vec_env.step(action)
        done = done_vec[0]
        info = infos[0]
        episode_starts = done_vec
        
    total_wait_time = -info["episode_reward"] * 100.0
    
    return {
        "Agent": "PPO",
        "Scenario": scenario,
        "Seed": seed,
        "Total_Cars_Processed": info["cars_processed"],
        "Total_Wait_Time": total_wait_time,
        "Average_Queue_Length": info["avg_queue_length"]
    }

if __name__ == "__main__":
    scenarios = ["uniform", "rush_hour_main", "rush_hour_sec", "quiet_night", "accident_main"]
    seeds = [10, 20, 30, 40, 50]
    
    results = []
    for scenario in scenarios:
        for seed in seeds:
            print(f"Running Baseline: {scenario} (Seed {seed})")
            baseline_res = evaluate_baseline(scenario=scenario, seed=seed)
            results.append(baseline_res)
            
            print(f"Running RL Agent: {scenario} (Seed {seed})")
            rl_res = evaluate_rl(scenario=scenario, seed=seed)
            results.append(rl_res)
            
    df = pd.DataFrame(results)
    df.to_csv("final_validation_report.csv", index=False)
    print("Robustness testing complete. Saved to final_validation_report.csv")
