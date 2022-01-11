"""
Author: André Ulrich
Test EMO and GLO together
"""

from EMO import *


system_1 = Low_Voltage_System(line_type='NAYY 4x120 SE',transformer_type="0.25 MVA 10/0.4 kV")
system_1.grid_from_GLO('grids/selfmade_grid.xlsx', {'S transformer': 240,
                                                            'line impedance': 0.004,
                                                            'i line max': 140
                                                            })

sim_handler_1 = Simulation_Handler(system_1,
                                           start_minute=60 * 12,
                                           end_minute=60 * 12 + 24 * 60,
                                           rapid=False)