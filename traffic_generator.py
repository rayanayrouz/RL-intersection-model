import os
import random

def generate_traffic(sumo_dir: str, scenario: str = "random"):
    """
    Generates a new bougara.rou.xml file with randomized traffic volumes,
    heterogeneous vehicle driver behaviors (varying speeds, accelerations, gaps),
    and diverse realistic traffic scenarios.
    """
    scenarios = [
        "uniform",            # Balanced normal traffic
        "rush_hour_main",     # Heavy North/South corridor
        "rush_hour_sec",      # Heavy inbound from secondary road
        "quiet_night",        # Very light sparse traffic
        "accident_main",      # Broken down vehicle blocking main NB lane
        "accident_sec",       # Broken down vehicle blocking secondary road
        "heavy_freight",      # Surge of slow trucks and commercial trailers
        "asymmetric_surge",   # Massive Northbound flow, Southbound nearly empty
        "morning_commute",    # Heavy inbound flow towards city center
        "evening_rush",       # Heavy outbound flow towards suburbs
        "midday_delivery",    # Fast mopeds and delivery vans buzzing through
        "platoon_bursts",     # Traffic arriving in packed clusters / waves
    ]
    if scenario == "random":
        scenario = random.choice(scenarios)

    # Base hourly flow volumes
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

    truck_mult = 1.0
    moto_mult = 1.0

    # Apply multipliers based on scenario
    if scenario == "rush_hour_main":
        main_mult = 2.1
        sec_mult = 1.0
    elif scenario == "rush_hour_sec":
        main_mult = 1.0
        sec_mult = 3.6
    elif scenario == "quiet_night":
        main_mult = 0.20
        sec_mult = 0.20
    elif scenario == "accident_main":
        main_mult = 1.3
        sec_mult = 0.9
    elif scenario == "accident_sec":
        main_mult = 1.1
        sec_mult = 1.8
    elif scenario == "heavy_freight":
        main_mult = 1.2
        sec_mult = 1.0
        truck_mult = 3.2
    elif scenario == "asymmetric_surge":
        main_mult = 1.9
        sec_mult = 0.8
    elif scenario == "morning_commute":
        main_mult = 1.8
        sec_mult = 2.2
        moto_mult = 1.4
    elif scenario == "evening_rush":
        main_mult = 2.0
        sec_mult = 1.4
    elif scenario == "midday_delivery":
        main_mult = 1.1
        sec_mult = 1.2
        moto_mult = 2.5
    elif scenario == "platoon_bursts":
        main_mult = 1.5
        sec_mult = 1.3
    else:  # uniform
        main_mult = 1.0
        sec_mult = 1.0

    # Add +/- 15% random noise to ensure every generated file is unique
    def apply_noise(val, mult):
        base = val * mult
        noise = random.uniform(0.85, 1.15)
        return max(5, int(base * noise))

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- ================================================================= -->
    <!-- Heterogeneous Vehicle Distributions: Realistic Varying Speeds     -->
    <!-- ================================================================= -->

    <!-- Cars: 55% normal, 25% aggressive speeders, 20% slow/cautious drivers -->
    <vTypeDistribution id="car_mix">
        <!-- Normal passenger car: average speed, realistic reaction -->
        <vType id="car_normal" probability="0.55" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" maxSpeed="50" speedFactor="normc(1.00,0.10,0.80,1.20)" guiShape="passenger"/>
        <!-- Fast / aggressive driver: quick acceleration, tailgates, drives 15-35% faster -->
        <vType id="car_fast" probability="0.25" accel="3.4" decel="5.2" sigma="0.2" length="4.5" minGap="1.8" maxSpeed="60" speedFactor="normc(1.18,0.15,0.95,1.40)" guiShape="passenger"/>
        <!-- Slow / cautious driver: slow to accelerate, keeps big safety gap, drives 15-30% slower -->
        <vType id="car_slow" probability="0.20" accel="1.7" decel="3.5" sigma="0.8" length="4.5" minGap="3.5" maxSpeed="40" speedFactor="normc(0.80,0.10,0.60,0.95)" guiShape="passenger"/>
    </vTypeDistribution>

    <!-- Motorcycles / Scooters: fast acceleration, lane filtering behavior -->
    <vTypeDistribution id="moto_mix">
        <vType id="moto_normal" probability="0.60" accel="3.0" decel="5.0" sigma="0.5" length="2.0" minGap="1.5" maxSpeed="55" speedFactor="normc(1.05,0.12,0.85,1.25)" guiShape="moped"/>
        <vType id="moto_agile"  probability="0.40" accel="3.8" decel="5.8" sigma="0.3" length="2.0" minGap="1.0" maxSpeed="65" speedFactor="normc(1.25,0.18,0.95,1.45)" guiShape="moped"/>
    </vTypeDistribution>

    <!-- Trucks / Commercial Vehicles: slow accelerating, heavy braking, lower top speed -->
    <vTypeDistribution id="truck_mix">
        <vType id="truck_standard" probability="0.70" accel="1.4" decel="3.5" sigma="0.4" length="9.5"  minGap="3.0" maxSpeed="40" speedFactor="normc(0.85,0.10,0.65,1.00)" guiShape="truck"/>
        <vType id="truck_heavy"    probability="0.30" accel="1.0" decel="2.8" sigma="0.6" length="12.0" minGap="4.0" maxSpeed="35" speedFactor="normc(0.70,0.08,0.50,0.85)" guiShape="truck/semitrailer"/>
    </vTypeDistribution>

    <!-- Pedestrians -->
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
    <flow id="fNB_straight" type="car_mix"   route="rNB_straight" begin="0" end="3600" vehsPerHour="{apply_noise(fNB_straight, main_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fNB_moto"     type="moto_mix"  route="rNB_straight" begin="0" end="3600" vehsPerHour="{apply_noise(fNB_moto, main_mult * moto_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fNB_truck"    type="truck_mix" route="rNB_straight" begin="0" end="3600" vehsPerHour="{apply_noise(fNB_truck, main_mult * truck_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fNB_to_third" type="car_mix"   route="rNB_to_third" begin="0" end="3600" vehsPerHour="{apply_noise(fNB_to_third, main_mult)}"  departSpeed="max" departLane="best"/>
    <flow id="fNB_to_sec"   type="car_mix"   route="rNB_to_sec"   begin="0" end="3600" vehsPerHour="{apply_noise(fNB_to_sec, main_mult)}"  departSpeed="max" departLane="best"/>

    <!-- Main Southbound -->
    <flow id="fSB_straight" type="car_mix"   route="rSB_straight" begin="0" end="3600" vehsPerHour="{apply_noise(fSB_straight, main_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fSB_moto"     type="moto_mix"  route="rSB_straight" begin="0" end="3600" vehsPerHour="{apply_noise(fSB_moto, main_mult * moto_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fSB_truck"    type="truck_mix" route="rSB_straight" begin="0" end="3600" vehsPerHour="{apply_noise(fSB_truck, main_mult * truck_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fSB_to_sec"   type="car_mix"   route="rSB_to_sec"   begin="0" end="3600" vehsPerHour="{apply_noise(fSB_to_sec, main_mult)}"  departSpeed="max" departLane="best"/>

    <!-- Secondary road inbound -->
    <flow id="fSec_NB"      type="car_mix"   route="rSec_to_NB"   begin="0" end="3600" vehsPerHour="{apply_noise(fSec_NB, sec_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fSec_SB"      type="car_mix"   route="rSec_to_SB"   begin="0" end="3600" vehsPerHour="{apply_noise(fSec_SB, sec_mult)}" departSpeed="max" departLane="best"/>
    <flow id="fSec_moto"    type="moto_mix"  route="rSec_to_NB"   begin="0" end="3600" vehsPerHour="{apply_noise(fSec_moto, sec_mult * moto_mult)}"  departSpeed="max" departLane="best"/>

    {f'''<!-- ===== ACCIDENT ON MAIN NORTHBOUND ===== -->
    <vehicle id="accident_main_veh" type="car_normal" route="rNB_straight" depart="250" departLane="0">
        <stop lane="main_NB_pc_jm_0" endPos="30" duration="2600"/>
    </vehicle>''' if scenario == 'accident_main' else ''}

    {f'''<!-- ===== ACCIDENT ON SECONDARY ROAD ===== -->
    <vehicle id="accident_sec_veh" type="car_slow" route="rSec_to_NB" depart="350" departLane="0">
        <stop lane="sec_in_2_0" endPos="25" duration="2400"/>
    </vehicle>''' if scenario == 'accident_sec' else ''}
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
