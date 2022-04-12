"""
Author: André Ulrich
Test EMO and GLO together
"""

from EMO import *
from optimization import GridLineOptimizer as GLO
from battery_electric_vehicle import BatteryElectricVehicle as BEV
from household import Household as HH

import matplotlib.pyplot as plt

#### GridLineOptimizer ####################################################
resolution = 15
buses = 6
bevs = 2
bev_lst = list(range(bevs))
bus_lst = list(range(buses))
s_trafo = 15 #kVA

# BEVs
#home_buses = [i for i in range(bevs)]
#start_socs = [30 - random.randint(-10, 10) for _ in range(bevs)]
#target_socs = [80 - random.randint(-20, 20) for _ in range(bevs)]
#target_times = [19 - random.randint(-4, 4) for _ in range(bevs)]
#start_times = [2 for _ in range(bevs)]
#bat_energies = [50 for _ in range(bevs)]

home_buses = [1, 5]
start_socs = [20, 20]
target_socs = [100, 100]
target_times = [20, 20]
start_times = [12, 12]
bat_energies = [50, 50]


# Households
ann_dems = [3500 for _ in range(buses)]

# BEVs erzeugen
bev_list = []
for car in bev_lst:
    bev = BEV(soc_start=start_socs[car], soc_target=target_socs[car],
              t_target=target_times[car], e_bat=bat_energies[car],
              resolution=resolution, home_bus=home_buses[car],
              t_start=start_times[car])
    bev_list.append(bev)

# Households erzeugen
household_list = []
for bus in bus_lst:
    household = HH(home_bus=bus, annual_demand=ann_dems[bus], resolution=resolution)
    household.raise_demand(11, 19, 1800)
    household_list.append(household)

test = GLO(number_buses=buses, bevs=bev_list, resolution=resolution, s_trafo_kVA=s_trafo,
           households=household_list, horizon_width=24, impedance=0.004)

#test.run_optimization_single_timestep(tee=True)
#test.plot_all_results(marker='x', legend=True, save=True, usetex=True)
#test.plot_SOC_results(marker=None, legend=False, save=True, usetex=True, compact_x=True)

# export grid as excel
grid_excel_file = 'optimized_grid'
test.export_grid(grid_excel_file)
grid_specs = test.get_grid_specs()
hh_data = test.export_household_profiles()
#wb_data = test.export_I_results()


system_1 = Low_Voltage_System(line_type='NAYY 4x120 SE', transformer_type="0.25 MVA 10/0.4 kV")
system_1.grid_from_GLO('grids/optimized_grid.xlsx', grid_specs)

sim_handler_1 = Simulation_Handler(system_1,
                                    start_minute=60 * 12,
                                    end_minute=60 * 12 + 24 * 60,
                                    rapid=False)

#sim_handler_1.run_GLO_sim(hh_data, wb_data, int(24*60/resolution), parallel=False)
sim_handler_1.run_unoptimized_sim(hh_data, bev_list, 96)

sim_handler_1.plot_EMO_sim_results(freq=resolution, element='buses', legend=False, marker=None,
                                   save=True, usetex=True, compact_x=True)
sim_handler_1.plot_EMO_sim_results(freq=resolution, element='lines', legend=False, marker=None,
                                   save=True, usetex=True, compact_x=True)
sim_handler_1.plot_EMO_sim_results(freq=resolution, element='trafo', legend=False, marker=None,
                                   save=True, usetex=True, compact_x=True)