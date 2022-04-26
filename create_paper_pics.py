"""
Author: André Ulrich
Test EMO and GLO together
"""
import random

import pandas as pd
from EMO import *
from optimization import GridLineOptimizer as GLO
from battery_electric_vehicle import BatteryElectricVehicle as BEV
from household import Household as HH

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#### GridLineOptimizer ####################################################
resolution = 6
buses = 40
bevs = 40
bev_lst = list(range(bevs))
bus_lst = list(range(buses))
s_trafo = 250 #kVA

runs = 7

for run in range(runs):
    random.seed(run)
    # BEVs
    home_buses = [i for i in range(bevs)]
    start_socs = [30 - random.randint(-10, 10) for _ in range(bevs)]
    target_socs = [80 - random.randint(-20, 20) for _ in range(bevs)]
    target_times = [19 - random.randint(-4, 4) for _ in range(bevs)]
    start_times = [12 - random.randint(-2, 2) for _ in range(bevs)]
    bat_energies = [50 for _ in range(bevs)]


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

    #GLO.set_options('equal SOCs', 0.05)

    test = GLO(number_buses=buses, bevs=bev_list, resolution=resolution, s_trafo_kVA=s_trafo,
               households=household_list, horizon_width=24)

    test.run_optimization_single_timestep(tee=False)

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
    sim_handler_1.store_sim_results(name_extension=f'_optimized_{run}')


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

    sim_handler_2.run_unoptimized_sim(hh_data, bev_list, int(24*60/resolution), control=False)
    sim_handler_2.store_sim_results(name_extension=f'_unoptimized_{run}')

    # run the simulation without the optimization of the bev loading
    # but first reset the bevs
    for num, bev in enumerate(bev_list):
        bev.current_soc = start_socs[num]


    system_3 = Low_Voltage_System(line_type='NAYY 4x120 SE', transformer_type="0.25 MVA 10/0.4 kV")
    system_3.grid_from_GLO('grids/optimized_grid.xlsx', grid_specs)

    sim_handler_3 = Simulation_Handler(system_3,
                                        start_minute=60 * 12,
                                        end_minute=60 * 12 + 24 * 60,
                                        rapid=False)

    #sim_handler_1.reset_GLO_sim_results()

    sim_handler_3.run_unoptimized_sim(hh_data, bev_list, int(24*60/resolution), control=True)
    sim_handler_3.store_sim_results(name_extension=f'_unoptimized-controlled_{run}')

# alles plotten lassen: erstmal die Daten reinladen und Einstellungen für plots
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['font.size'] = 11

x_fmt = mdates.DateFormatter('%H')

#### optimized versions ##################################################################
opt_buses_df = []
opt_lines_df = []
opt_trafo_df = []

for run in range(runs):
    opt_buses_df.append(pd.read_csv(f'results/res_buses_optimized_{run}.csv'))
    opt_lines_df.append(pd.read_csv(f'results/res_lines_optimized_{run}.csv'))
    opt_trafo_df.append(pd.read_csv(f'results/res_trafo_optimized_{run}.csv'))

# concat each list of dfs to one big df
big_opt_trafo_df = pd.DataFrame()
for df in opt_trafo_df:
    big_opt_trafo_df = pd.concat([big_opt_trafo_df, df], axis=0)

big_opt_buses_df = pd.DataFrame()
for df in opt_buses_df:
    big_opt_buses_df = pd.concat([big_opt_buses_df, df], axis=0)

#### unoptimized versions ################################################################
unopt_buses_df = []
unopt_trafo_df = []

for run in range(runs):
    unopt_buses_df.append(pd.read_csv(f'results/res_buses_unoptimized_{run}.csv'))
    unopt_trafo_df.append(pd.read_csv(f'results/res_trafo_unoptimized_{run}.csv'))

big_unopt_buses_df = pd.DataFrame()
for df in unopt_buses_df:
    big_unopt_buses_df = pd.concat([big_unopt_buses_df, df], axis=0)

big_unopt_trafo_df = pd.DataFrame()
for df in unopt_trafo_df:
    big_unopt_trafo_df = pd.concat([big_unopt_trafo_df, df], axis=0)

#### unoptimized but controlled versions #################################################
contr_buses_df = []
contr_trafo_df = []

for run in range(runs):
    contr_buses_df.append(pd.read_csv(f'results/res_buses_unoptimized-controlled_{run}.csv'))
    contr_trafo_df.append(pd.read_csv(f'results/res_trafo_unoptimized-controlled_{run}.csv'))

big_contr_trafo_df = pd.DataFrame()
for df in contr_trafo_df:
    big_contr_trafo_df = pd.concat([big_contr_trafo_df, df], axis=0)

big_contr_buses_df = pd.DataFrame()
for df in contr_buses_df:
    big_contr_buses_df = pd.concat([big_contr_buses_df, df], axis=0)

#### compare grid loading for different scenarios #############################
big_opt_trafo_df_sorted = big_opt_trafo_df.sort_values(by='0', ascending=False, ignore_index=True)
big_contr_trafo_df_sorted = big_contr_trafo_df.sort_values(by='0', ascending=False, ignore_index=True)
load_diff_trafo = big_opt_trafo_df_sorted - big_contr_trafo_df_sorted
#load_diff_trafo = load_diff_trafo.sort_values(by='0', ascending=False)

fig0, ax0 = plt.subplots(1, 1, figsize=(6.5, 1.8))
ax0.plot(range(len(load_diff_trafo)), load_diff_trafo.sort_values(by='0', ascending=False).loc[:, '0'])
ax0.set_xlabel('Timesteps [hours]')
ax0.set_ylabel('$\Delta$ transformer loading [\%]')
mean_load_diff = load_diff_trafo.loc[:, '0'].mean()
print(mean_load_diff)
ax0.grid()
ax0.axhline(y=mean_load_diff, linewidth=0.7, color='black', linestyle='--')
pos = ax0.get_xticks()[1:-1]
ax0.set_xticks(pos, [int(val*resolution/60) for val in pos])
fig0.savefig('trafo-loading-comparison.pdf', bbox_inches='tight')



fig, ax = plt.subplots(2, 1, figsize=(6.5, 4), sharex=True)
#sort_values(by='0', ascending=False)
ax[0].plot(range(len(big_opt_trafo_df)), big_opt_trafo_df.sort_values(by='0', ascending=False).loc[:, '0'],
        label='optimised')
ax[0].plot(range(len(big_opt_trafo_df)), big_unopt_trafo_df.sort_values(by='0', ascending=False).loc[:, '0'],
        label='unoptimised')
ax[0].plot(range(len(big_opt_trafo_df)), big_contr_trafo_df.sort_values(by='0', ascending=False).loc[:, '0'],
        label='unoptimised, controlled')
#ax[0].set_xlabel('Number of minutes')
ax[0].set_ylabel('Transformer loading [\%]')
# big_opt_buses_df.sort_values(by='41', ascending=True).loc[:, '41']
ax[1].plot(range(len(big_opt_trafo_df)), big_opt_buses_df.sort_values(by='41', ascending=True).loc[:, '41'],
        label='optimised')
ax[1].plot(range(len(big_opt_trafo_df)), big_unopt_buses_df.sort_values(by='41', ascending=True).loc[:, '41'],
        label='unoptimised')
ax[1].plot(range(len(big_opt_trafo_df)), big_contr_buses_df.sort_values(by='41', ascending=True).loc[:, '41'],
        label='unoptimised, controlled')
ax[1].set_xlabel('Timesteps [hours]')
ax[1].set_ylabel('Node voltage [V]')
ax[0].legend()
ax[1].legend()
ax[0].grid()
ax[1].grid()
pos= ax[1].get_xticks()[1:-1]
ax[1].set_xticks(pos, [int(val*resolution/60) for val in pos])

ax[0].axhline(y=100, color='red', linestyle='--', linewidth=0.7)
ax[1].axhline(y=360, color='red', linestyle='--', linewidth=0.7)

#how many minutes under 360V?
crit_v_unopt = len(big_unopt_buses_df.loc[big_unopt_buses_df['41']<360, '41'])
crit_v_contr = len(big_contr_buses_df.loc[big_contr_buses_df['41']<360, '41'])
crit_p_unopt = len(big_unopt_trafo_df.loc[big_unopt_trafo_df['0']>100, '0'])

ax[0].axvline(x=crit_p_unopt, ymax=0.59, color='red', linestyle='--', linewidth=0.7)

ax[1].axvline(x=crit_v_unopt, ymax=0.63, color='red', linestyle='--', linewidth=0.7)
ax[1].axvline(x=crit_v_contr, ymax=0.63, color='red', linestyle='--', linewidth=0.7)

fig.savefig('trafo-bus_comparison.pdf', bbox_inches='tight')

#
# fig1, ax1 = plt.subplots(1, 1, figsize=(6.5, 1.75))
# #sort_values(by='0', ascending=False)
# ax1.plot(range(len(big_opt_trafo_df)), big_opt_buses_df.sort_values(by='41', ascending=True).loc[:, '41'],
#         label='optimiert')
# ax1.plot(range(len(big_opt_trafo_df)), big_unopt_buses_df.sort_values(by='41', ascending=True).loc[:, '41'],
#         label='unoptimiert')
# ax1.plot(range(len(big_opt_trafo_df)), big_contr_buses_df.sort_values(by='41', ascending=True).loc[:, '41'],
#         label='unoptimiert, geregelt')
# ax1.set_xlabel('Anzahl Minuten')
# ax1.set_ylabel('Spannung [V]')
# ax1.legend()
# ax1.grid()
#
# fig1.savefig('buses_comparison.pdf', bbox_inches='tight')
