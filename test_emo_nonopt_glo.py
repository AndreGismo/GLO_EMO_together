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
resolution = 5
buses = 6
bevs = 6
bev_lst = list(range(bevs))
bus_lst = list(range(buses))
s_trafo = 25 #kVA

# BEVs
#home_buses = [i for i in range(bevs)]
#start_socs = [30 - random.randint(-10, 10) for _ in range(bevs)]
#target_socs = [80 - random.randint(-20, 20) for _ in range(bevs)]
#target_times = [19 - random.randint(-4, 4) for _ in range(bevs)]
#start_times = [2 for _ in range(bevs)]
#bat_energies = [50 for _ in range(bevs)]

home_buses = [0, 1, 2, 3, 4, 5]
start_socs = [20, 20, 20, 20, 20, 20]
target_socs = [100, 100, 100, 100, 100, 100]
target_times = [14, 16, 18, 20, 18, 20]
start_times = [10, 12, 14, 14, 12, 14]
bat_energies = [50, 50, 50, 50, 50, 50]

final_socs_unopt = []
final_socs_opt = []


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
    #household.raise_demand(11, 19, 1800)
    household_list.append(household)

GLO.set_options('equal SOCs', 0)

test = GLO(number_buses=buses, bevs=bev_list, resolution=resolution, s_trafo_kVA=s_trafo,
           households=household_list, horizon_width=24, impedance=0.004)

test.run_optimization_single_timestep(tee=True)
test.plot_I_results(marker=None, legend=True, save=True, usetex=True, compact_x=True)
test.plot_SOC_results(marker=None, legend=True, save=True, usetex=True, compact_x=True)

# export grid as excel
grid_excel_file = 'optimized_grid'
test.export_grid(grid_excel_file)
grid_specs = test.get_grid_specs()
hh_data = test.export_household_profiles()
wb_data = test.export_I_results()


system_1 = Low_Voltage_System(line_type='NAYY 4x120 SE', transformer_type="0.25 MVA 10/0.4 kV")
system_1.grid_from_GLO('grids/optimized_grid.xlsx', grid_specs)

sim_handler_1 = Simulation_Handler(system_1,
                                    start_minute=60 * 12,
                                    end_minute=60 * 12 + 24 * 60,
                                    rapid=False)

# run the simulation with the optimized results for the bev loading
sim_handler_1.run_GLO_sim(hh_data, wb_data, int(24*60/resolution), parallel=False)
# store the results of the optimized
sim_handler_1.store_sim_results()

for bev in bev_list:
    final_socs_opt.append(test.optimization_model.SOC[int(24*60/resolution)-1, bev.home_bus].value)

# run the simulation without the optimization of the bev loading
# but first reset the bevs
for num, bev in enumerate(bev_list):
    bev.current_soc = start_socs[num]

# and create a fresh sim_handler instance
system_2 = Low_Voltage_System(line_type='NAYY 4x120 SE', transformer_type="0.25 MVA 10/0.4 kV")
system_2.grid_from_GLO('grids/optimized_grid.xlsx', grid_specs)

sim_handler_2 = Simulation_Handler(system_2,
                                    start_minute=60 * 12,
                                    end_minute=60 * 12 + 24 * 60,
                                    rapid=False)

#sim_handler_1.reset_GLO_sim_results()

sim_handler_2.run_unoptimized_sim(hh_data, bev_list, int(24*60/resolution))
# load the results from the previous simulation
sim_handler_2.load_sim_results()

sim_handler_2.plot_EMO_sim_results(freq=resolution, element='buses', legend=False, marker=None,
                                   save=True, usetex=True, compact_x=True, name_ext='_unopt')
sim_handler_2.plot_EMO_sim_results(freq=resolution, element='lines', legend=False, marker=None,
                                   save=True, usetex=True, compact_x=True, name_ext='_unopt')
sim_handler_2.plot_EMO_sim_results(freq=resolution, element='trafo', legend=True, marker=None,
                                   save=True, usetex=True, compact_x=True, compare=True)

for bev in bev_list:
    final_socs_unopt.append(bev.current_soc)

print('SOCs nach optimiertem Szenario', final_socs_opt)
print('SOCs nach unoptimiertem Szenario', final_socs_unopt)

