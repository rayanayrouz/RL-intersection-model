import os
import random

def generate_traffic(sumo_dir: str, scenario: str = "random"):
    """
    Generates a new bougara.rou.xml file with randomized traffic volumes.
    
    scenarios:
      - "uniform": Base traffic.
      - "rush_hour_main": Heavy traffic on the North/South main road.
      - "rush_hour_sec": Heavy traffic on the secondary road.
      - "quiet_night": Very low traffic everywhere.
      - "random": Randomly picks one of the above.
    """
    scenarios = ["uniform", "rush_hour_main", "rush_hour_sec", "quiet_night", "accident_main"]
    if scenario == "random":
        scenario = random.choice(scenarios)

    # Base volumes
    fNB_straight = 480
    fNB_moto = 120
    fNB_truck = 40
    fNB_to_third = 60
    fNB_to_sec = 80

    fSB_straight = 460
    fSB_moto = 100
    fSB_truck = 30
    fSB_to_sec = 70

    fSec_NB = 100
    fSec_SB = 100
    fSec_moto = 40

    # Apply multipliers based on scenario
    if scenario == "rush_hour_main":
        main_mult = 2.0
        sec_mult = 1.0
    elif scenario == "rush_hour_sec":
        main_mult = 1.0
        sec_mult = 4.0
    elif scenario == "quiet_night":
        main_mult = 0.2
        sec_mult = 0.2
    elif scenario == "accident_main":
        main_mult = 1.2
        sec_mult = 0.8
    else: # uniform
        main_mult = 1.0
        sec_mult = 1.0

    # Add +/- 15% random noise to everything to ensure it's never exactly the same
    def apply_noise(val, mult):
        base = val * mult
        noise = random.uniform(0.85, 1.15)
        return int(base * noise)

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- Vehicle types -->
    <vType id="car"    accel="2.6" decel="4.5" sigma="0.5" length="4.5"  minGap="2.5" maxSpeed="50" guiShape="passenger"/>
    <vType id="moto"   accel="3.0" decel="5.0" sigma="0.6" length="2.0"  minGap="1.5" maxSpeed="55" guiShape="moped"/>
    <vType id="truck"  accel="1.5" decel="3.5" sigma="0.4" length="10.0" minGap="3.0" maxSpeed="40" guiShape="truck"/>
    <vType id="pedestrian" vClass="pedestrian" length="0.25" minGap="0.25" maxSpeed="1.5" guiShape="pedestrian"/>

    <!-- ===== VEHICLE ROUTES ===== -->
    <route id="rNB_straight" edges="main_NB_s_jt main_NB_jt_pc main_NB_pc_jm main_NB_jm_pb main_NB_pb_n"/>
    <route id="rNB_to_third" edges="main_NB_s_jt third_out"/>
    <route id="rNB_to_sec"   edges="main_NB_s_jt main_NB_jt_pc main_NB_pc_jm sec_out_1 sec_out_2"/>
    <route id="rSB_straight" edges="main_SB_n_pb main_SB_pb_jm main_SB_jm_pc main_SB_pc_jt main_SB_jt_s"/>
    <route id="rSB_to_sec"   edges="main_SB_n_pb main_SB_pb_jm sec_out_1 sec_out_2"/>
    <route id="rSec_to_NB"   edges="sec_in_1 sec_in_2 main_NB_jm_pb main_NB_pb_n"/>
    <route id="rSec_to_SB"   edges="sec_in_1 sec_in_2 main_SB_jm_pc main_SB_pc_jt main_SB_jt_s"/>

    <!-- ===== PEDESTRIAN ROUTES ===== -->
    <route id="rPed_A_fwd" edges="sec_out_1"/>
    <route id="rPed_A_bwd" edges="sec_in_2"/>

    <!-- ===== VEHICLE FLOWS (Scenario: {scenario}) ===== -->
    <!-- Main Northbound -->
    <flow id="fNB_straight" type="car"  route="rNB_straight" begin="0" end="3600" vehsPerHour="{apply_noise(fNB_straight, main_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fNB_moto"     type="moto" route="rNB_straight" begin="0" end="3600" vehsPerHour="{apply_noise(fNB_moto, main_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fNB_truck"    type="truck" route="rNB_straight" begin="0" end="3600" vehsPerHour="{apply_noise(fNB_truck, main_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fNB_to_third" type="car"  route="rNB_to_third" begin="0" end="3600" vehsPerHour="{apply_noise(fNB_to_third, main_mult)}"  departSpeed="max" departLane="best"/>
    <flow id="fNB_to_sec"   type="car"  route="rNB_to_sec"   begin="0" end="3600" vehsPerHour="{apply_noise(fNB_to_sec, main_mult)}"  departSpeed="max" departLane="best"/>

    <!-- Main Southbound -->
    <flow id="fSB_straight" type="car"  route="rSB_straight" begin="0" end="3600" vehsPerHour="{apply_noise(fSB_straight, main_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fSB_moto"     type="moto" route="rSB_straight" begin="0" end="3600" vehsPerHour="{apply_noise(fSB_moto, main_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fSB_truck"    type="truck" route="rSB_straight" begin="0" end="3600" vehsPerHour="{apply_noise(fSB_truck, main_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fSB_to_sec"   type="car"  route="rSB_to_sec"   begin="0" end="3600" vehsPerHour="{apply_noise(fSB_to_sec, main_mult)}"  departSpeed="max" departLane="best"/>

    <!-- Secondary road inbound -->
    <flow id="fSec_NB"      type="car"  route="rSec_to_NB"   begin="0" end="3600" vehsPerHour="{apply_noise(fSec_NB, sec_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fSec_SB"      type="car"  route="rSec_to_SB"   begin="0" end="3600" vehsPerHour="{apply_noise(fSec_SB, sec_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fSec_moto"    type="moto" route="rSec_to_NB"   begin="0" end="3600" vehsPerHour="{apply_noise(fSec_moto, sec_mult)}"  departSpeed="max" departLane="best"/>

    {f'''<!-- ===== ACCIDENT VEHICLE ===== -->
    <!-- Simulates a vehicle breaking down in the right lane (lane 0) of the Northbound approach -->
    <vehicle id="accident_veh" type="car" route="rNB_straight" depart="300" departLane="0">
        <stop lane="main_NB_pc_jm_0" endPos="30" duration="2700"/>
    </vehicle>''' if scenario == 'accident_main' else ''}
</routes>
"""
    route_file = os.path.join(sumo_dir, "bougara.rou.xml")
    with open(route_file, "w") as f:
        f.write(xml_content)

if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _sumo_dir = os.path.join(_script_dir, "sumo_files")
    generate_traffic(_sumo_dir, "random")
    print(f"Random traffic generated in {_sumo_dir}/bougara.rou.xml")
