import os
import traci
import pandas as pd
import numpy as np
import traffic_generator
from rl import SUMO_CFG, SUMO_DIR, MAIN_GREEN_PHASE, SEC_GREEN_PHASE, THIRD_NB_PHASE, ALL_LANES

def evaluate_baseline(scenario="random", seed=42, use_gui=False):
    """
    Evaluates a fixed-time controller (40s Main Green, 3s Yellow, 20s Sec Green, 3s Yellow).
    """
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    traffic_generator.generate_traffic(SUMO_DIR, scenario=scenario)
    
    sumo_cmd = [
        sumo_binary,
        "-c", SUMO_CFG,
        "--no-step-log", "true",
        "--log", os.devnull,
        "--seed", str(seed)
    ]
    
    traci.start(sumo_cmd)
    
    # 0 = Main Green, 1 = Yellow, 2 = Sec Green, 3 = Yellow
    state_times = [40, 3, 20, 3] 
    
    step = 0
    total_steps = 3600
    
    queue_lengths = []
    total_wait_time = 0
    cars_processed = 0 
    
    # Initialize phases
    traci.trafficlight.setPhase("J_main_sec", MAIN_GREEN_PHASE)
    traci.trafficlight.setPhase("J_third", THIRD_NB_PHASE)
    
    current_state = 0
    time_in_state = 0
    
    while step < total_steps:
        traci.simulationStep()
        
        halted = 0
        for lane in ALL_LANES:
            try:
                halted += traci.lane.getLastStepHaltingNumber(lane)
                total_wait_time += traci.lane.getWaitingTime(lane)
            except:
                pass
        
        queue_lengths.append(halted)
        cars_processed += traci.simulation.getArrivedNumber()
        
        step += 1
        time_in_state += 1
        
        if time_in_state >= state_times[current_state]:
            current_state = (current_state + 1) % 4
            time_in_state = 0
            
            if current_state == 0:
                traci.trafficlight.setPhase("J_main_sec", MAIN_GREEN_PHASE)
                traci.trafficlight.setPhase("J_third", THIRD_NB_PHASE)
            elif current_state == 1:
                traci.trafficlight.setPhase("J_main_sec", MAIN_GREEN_PHASE + 1)
                traci.trafficlight.setPhase("J_third", 1) # Yellow
            elif current_state == 2:
                traci.trafficlight.setPhase("J_main_sec", SEC_GREEN_PHASE)
                traci.trafficlight.setPhase("J_third", 2) # Red
            elif current_state == 3:
                traci.trafficlight.setPhase("J_main_sec", SEC_GREEN_PHASE + 1)
                # J_third remains red
                
    traci.close()
    
    avg_queue = np.mean(queue_lengths)
    
    return {
        "Agent": "Fixed-Time",
        "Scenario": scenario,
        "Seed": seed,
        "Total_Cars_Processed": cars_processed,
        "Total_Wait_Time": total_wait_time,
        "Average_Queue_Length": avg_queue
    }

if __name__ == "__main__":
    print("Evaluating Fixed-Time Baseline on 'uniform' scenario...")
    res = evaluate_baseline(scenario="uniform", seed=10)
    print(res)
    df = pd.DataFrame([res])
    df.to_csv("baseline_metrics.csv", index=False)
    print("Saved to baseline_metrics.csv")
