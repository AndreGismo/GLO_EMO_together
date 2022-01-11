"""
TH Koeln, PROGRESSUS

Module Electrics Management and Optimization (EMO)

Created August 2020

@author: Hotz
00==============================================
Erweitert von André Ulrich für die Masterarbeit
Zusammenspiel von GLO und EMO
What has to be done:
- create grid according to GLO
- pass the household load profiles to GLO
  (then GLO calculates optmized wallbox currents)
- receive the optimized wallbox currents from GLO
"""

#plt.rcParams['figure.dpi'] = 300
import pandas as pd

if True:  # Import
    import pandapower as pp          # grid calculation tool
    import pandapower.plotting       # plotting
    from   scipy.io import loadmat   # load matlab database (household loads)
    import random                    # random numbers
    import numpy                     # vectors, matrices, arrays, gaussians, corrcoeff
    import math                      # various maths stuff
    import cmath                     # complex maths
    import pickle                    # save and load python objects
    import pandas                    # algebra stuff
    import copy
    import sys
    #sys.path.append('C:/Users/Hotz/Dropbox/Weitere/Programmierung')
    import PSL
    
    
class Household(): 
    # class household
    def __init__(self,system,position_load,position_bus,position_households,position_main_branch):
        self.position_load = position_load  # position in list "loads" of panda_grid
        self.position_bus = position_bus  # position in list "bus" of panda_grid
        self.position_households = position_households
        self.position_main_branch = position_main_branch
        self.timeline = []
        self.wallbox = None

class Household_Timeline_Handler(): 
    # class for household load data handling, data from IÖW-Verbrauchsdaten
    def __init__(self, system, path, month=1):
        self.system = system
        self.load_data(path)
        self.month = month

    def load_data(self, path): 
        # load and parse household_load_data from MAT-file
        # Pflugradt should be used. 
        tmp = loadmat(path)
        tmp = tmp.get("Verbrauchsdaten").tolist()
        tmp = tmp[0];tmp = tmp[0];tmp = tmp[0]
        tmp = tmp.tolist()  # foo
        tmp = tmp[0];tmp = tmp[0];tmp = tmp[0]
        tmp = tmp.tolist()  # bar
        tmp = tmp[0];tmp = tmp[0];tmp = tmp[0]
        tmp = tmp.tolist()  # tf mate
        tmp = tmp[0]
        self.data = tmp

    def get_random_start_time(self): 
        # gets random day out of specified month to make sure household timelines are different
        day = round(random.random() * 30)
        offset = round(random.random() * 75) # few minutes offset in the time of day for the same reason
        start_frame = ((self.month - 1) * 30 + day) * 1440 + offset
        end_frame = ((self.month - 1) * 30 + day + 1) * 1440 + offset
        return [start_frame, end_frame]

    def make_village_household_loads(self, start_minute, end_minute):  
        # creates timelines and writes them into household instances
        for household_item in self.system.households:
            for index in range(0, int(end_minute / 1430) + 1):  
                # make as many day-timelines as needed plus a litte slack
                data_range = self.get_random_start_time()
                household_item.timeline += self.data[data_range[0] : data_range[1]]
            household_item.timeline = household_item.timeline[start_minute:end_minute]  # crop to required size
            
class Wallbox():  
    # e-vehicle-charger. handles load timelines, SOC, charging. later also controls, communication
    
    WB_index = 0
    
    def __init__(self,system,household,position_load,position_wallboxes,position_bus,
                 rated_power_kW=-1,battery_capacity_kWh=-1,battery_SOC=-1):
        self.index=Wallbox.WB_index
        Wallbox.WB_index+=1
        
        self.rated_power_kW = rated_power_kW
        if (rated_power_kW == -1):  # rated power of charger, if not specified: random value
            self.rated_power_kW = PSL.gaussian_limited(20, 20, 10, 35)
            
        self.battery_capacity_kWh = battery_capacity_kWh  # capacity of battery
        if battery_capacity_kWh == -1:  # if not specified: random value
            self.battery_capacity_kWh = PSL.gaussian_limited(50, 50, 15, 90)  # Tesla Modell S: 100 kWh
        
        self.battery_SOC = battery_SOC
        if battery_SOC == -1:  # initial state of charge; if not specified: random SOC
            self.set_random_SOC()    
        
        self.system             = system  # reference to the system instance it is part of
        self.position_wallboxes = (position_wallboxes)  # position in list "wallboxes" of low_voltage_grid
        self.household          = household  # instance of household it is connected to
        self.household.wallbox  = self
        self.position_load      = position_load  # position in list "loads" of panda_grid
        self.position_bus       = position_bus  # position in list "bus" of panda_grid
        self.ctrl_power_cap     = self.system.max_bus_power  # power cap for voltage controls


    def adapt_ctrl_power_cap(self): 
        # voltage-power-controller. each wallbox has own cap, they're all broadcastet and the global min used
        if min([x for x in self.system.panda_grid.res_bus.vm_pu]) < 0.98:
            self.ctrl_power_cap -= 2
        
        if min([x for x in self.system.panda_grid.res_bus.vm_pu]) > 0.985:
            self.ctrl_power_cap += 0.2
        
        if self.ctrl_power_cap > self.system.max_bus_power:
            self.ctrl_power_cap = self.system.max_bus_power
        
        if self.ctrl_power_cap < 0:
            self.ctrl_power_cap = 0

    def set_random_SOC(self):
        self.battery_SOC = PSL.gaussian_limited(0.3, 0.5, 0, 1)

    def make_random_connection_timeline(self, start_minute, end_minute):  
        # create timeline of when car is connected to charger . to be replaced with work of Sir Marian Sprünken
        
        tmp = []
        for index in range(0, round(end_minute / 1438) + 2):  # number of days the sim runs plus some slack
            # random disconnection time: agent leaves house in the morning    
            disconnect_frame    = int(PSL.gaussian_limited(60 * 7.5, 150, 60 * 4, 60 * 11))  
            # random connection time: agent returns home in the evening
            connect_frame       = int(PSL.gaussian_limited(60 * 17.5, 60 * 4, 60 * 13.5, 60 * 23))  
            tmp += (numpy.ones(disconnect_frame).tolist() 
                    + numpy.zeros(connect_frame - disconnect_frame).tolist() 
                    + numpy.ones(1441 - connect_frame).tolist())
            
        self.connection_timeline = tmp[start_minute:end_minute]  # cut to required length
        
        
    def make_random_connection_timeline_rapid(self, start_minute, end_minute):  
        # create timeline of when car is connected to charger . to be replaced with work of Sir Marian Sprünken
                
        no_boxes=self.system.no_wallboxes
        index=self.index
        
        tmp = []
        tmp += numpy.zeros(index*2+1).tolist()+[1]+numpy.zeros((no_boxes-index-1)*2).tolist()
        loops=math.ceil((end_minute-start_minute)/(len(tmp)))+1
        tmp=tmp*loops
        self.connection_timeline = tmp[0:(end_minute-start_minute+1)]  # cut to required length
        
    def get_power(self, frame, voltage_control):  
        # get power output for frame based on rated power and control algorithm
        
        if (self.connection_timeline[frame] == 1 and self.connection_timeline[frame - 1] == 0): # car arrives at home
                self.set_random_SOC()
        
        if (self.connection_timeline[frame] == 1 and self.battery_SOC < 1):  # car connected and not fully charged
            global_power_cap = min([x.ctrl_power_cap for x in self.system.wallboxes])  # global power cap 
            if voltage_control:
                power = min(self.rated_power_kW, global_power_cap)
            else:
                power = self.rated_power_kW
            self.battery_SOC = (self.battery_SOC + power / 60 / self.battery_capacity_kWh)  # update battery SOC
            return power
        else:
            return 0

class Low_Voltage_System():  
    # class to wrap them all: panda-grid, wallboxes, households
    def __init__(self,line_type,
                 transformer_type):
        self.line_type=line_type
        self.transformer_type=transformer_type
        self.max_bus_power=35
        self.grid = None


    def make_system_from_excel_file(self, file):
        # read stuff from Excel file and format
        lines=pandas.read_excel(file, sheet_name="Lines")
        busses=pandas.read_excel(file, sheet_name="Busses")
        lines=pandas.DataFrame(lines, columns= ['From Bus','To Bus','Length'])
        busses=pandas.DataFrame(busses, columns= ['X','Y','Household','Wallbox'])
        
        household_list     = [pos for (pos,x) in enumerate(busses['Household'],start=0) if x=='Yes']
        wallbox_list       = [pos for (pos,x) in enumerate(busses['Wallbox'],start=0) if x=='Yes']
        
        #wallbox_list= [x for x in sim_handler_1.system.wallboxes if random.random()>0.8]
        
        self.no_households = len(household_list)
        self.no_wallboxes  = len(wallbox_list)
        
        bus_link_list = [list(x) for x in zip(list(lines['From Bus']),list(lines['To Bus']),list(lines['Length']))]
        coordinates   = [list(x) for x in zip(list(busses['X']),list(busses['Y']))]
        
        grid = pp.create_empty_network()
        pp.create_bus(grid, name="ONS_MS", vn_kv=10, type="b")  # bus0 for infinite grid, Medium Voltage
        pp.create_ext_grid(grid, 0, name="starres_Netz", vm_pu=1, va_degree=0)  # define bus as infinite grid
        pp.create_bus(grid, name="ONS_NS", vn_kv=0.4, type="b")  # ONS bus1

        for index in range(2, max([x[1] for x in bus_link_list]) + 1):  # buses on main axes
            pp.create_bus(grid, vn_kv=0.4, name="Bus No. " + str(index))

        for item in bus_link_list: 
            # main grid lines
            pp.create_line(grid,from_bus=item[0],to_bus=item[1],length_km=item[2] * 0.001,
                           std_type=self.line_type,name="Main branch " + str(item))

        self.households = []
        self.wallboxes  = []

        for item in household_list: 
            #create households, wallboxes, connectors etc.
            index_bus_household = pp.create_bus(grid, vn_kv=0.4, name="Bus Household " + str(item))  # bus for house
            pp.create_line(grid,from_bus=item,to_bus=index_bus_household, # line from main axis to house
                           length_km = 10 * 0.001,std_type=self.line_type,
                           name="Main branch " + str(item) + " to household "+ str(index_bus_household))  
            index_load_household = pp.create_load(grid,name="Load household " + str(item),bus=index_bus_household,
                                                  p_mw=0,q_mvar=0)  # create load
            self.households.append(Household(system=self,position_load=index_load_household,
                                             position_bus=index_bus_household,position_households=len(self.households),
                                             position_main_branch=item))

            if item in wallbox_list:
                index_bus_wallbox = pp.create_bus(grid, vn_kv=0.4, name="Bus Wallbox " + str(item))  # bus for WB
                pp.create_line(grid,from_bus=index_bus_household,to_bus=index_bus_wallbox,length_km=1 * 0.001,
                    std_type=self.line_type,name="Household " + str(index_bus_household)+ " to Wallbox "
                    + str(index_bus_wallbox))  # createline from main axis to house
                wallbox_load = pp.create_load(grid,name="Load Wallbox " + str(item),bus=index_bus_wallbox,
                    p_mw=0,q_mvar=0)
                self.wallboxes.append(Wallbox(system=self,household=self.households[-1],position_load=wallbox_load,
                        position_wallboxes=len(self.wallboxes),position_bus=index_bus_wallbox))
                self.households[-1].wallbox = self.wallboxes[-1]
        
        #pp.plotting.create_generic_coordinates(grid)
        pp.runpp(grid)
        
        # for (index,item) in enumerate(coordinates,start=0):
        #     grid.bus_geodata.x[index]=item[1]
        #     grid.bus_geodata.y[index]=item[0]
        #
        # for item in self.households:
        #     grid.bus_geodata.x[item.position_bus]=grid.bus_geodata.x[item.position_main_branch]+0.2
        #     grid.bus_geodata.y[item.position_bus]=grid.bus_geodata.y[item.position_main_branch]
        #
        # for item in self.wallboxes:
        #     grid.bus_geodata.x[item.position_bus]=grid.bus_geodata.x[item.household.position_main_branch]+0.2
        #     grid.bus_geodata.y[item.position_bus]=grid.bus_geodata.y[item.household.position_main_branch]+0.2
        
        pp.create_transformer(grid, 0, 1, name="Transformator", std_type=self.transformer_type)
        return grid


    def grid_from_GLO(self, GLO_grid_file, GLO_grid_params, ideal=True):
        """
        reads in the GLO_grid_file (which contains the information about the
        grid the GLO is optimizing for) and turns it into a pandapower grid
        set as attribute of class
        :param GLO_grid_file: excel-file
        :return: None
        """
        grid_data = pd.read_excel(GLO_grid_file, sheet_name=['Lines', 'Busses'])
        #buses = pd.read_excel(GLO_grid_file, sheet_name='Busses')
        v_mv = 20
        v_lv = 0.4
        s_trafo = GLO_grid_params['S transformer']
        line_impedance = GLO_grid_params['line impedance']
        i_max_line = GLO_grid_params['i line max']
        # assume all lines to be 15m for calculating R/l
        r_spec = line_impedance*1000/15 # /1000 since pandas expects length in km

        if ideal:
            vkr = 0
            pfe = 0
            i0 = 0
            vk = 0

        else:
            vkr = 1.5
            pfe = 0.4
            i0 = 0.4
            vk = 6

        # empty grid
        grid = pp.create_empty_network()
        # buses for attaching the transformer
        pp.create_bus(grid, name='transformer mv', vn_kv=v_mv)
        pp.create_bus(grid, name='transformer lv', vn_kv=v_lv)
        # create transformer
        pp.create_transformer_from_parameters(grid, hv_bus=0, lv_bus=1, sn_mva=s_trafo/1000,
                                              vn_hv_kv=v_mv, vn_lv_kv=v_lv, vkr_percent=vkr,
                                              pfe_kw=pfe, i0_percent=i0, vk_percent=vk)
        # slack
        pp.create_ext_grid(grid, bus=0)
        # create all the other busses
        for i in range(2, len(grid_data['Lines'])):
            pp.create_bus(grid, name='bus'+str(i), vn_kv=0.4)
            if grid_data['Busses'].loc[i, 'Household'] == 'Yes':
                pp.create_load(grid, bus=i, p_mw=0, name='Load at bus'+str(i))

        # create all the lines
        for i in range(3, len(grid_data['Lines'])):
            pp.create_line_from_parameters(grid, from_bus=i-1, to_bus=i, length_km=0.015,
                                           r_ohm_per_km=r_spec, name='line '+str(i-1)+'-'+str(i),
                                           x_ohm_per_km=0, c_nf_per_km=0, max_i_ka=i_max_line)

        self.grid = grid



#
class Simulation_Handler():  
    # executes timeseries sim step by step, logs results, handles global voltage-power-controller
    def __init__(self, system, start_minute=0, end_minute=0, rapid=False):
        self.system = system
        self.results = {}    # logged as a dict so it can easily be pickled, to be used e.g. in Topology Estimator
        self.start_minute = start_minute
        self.end_minute = end_minute
        self.rapid=rapid
        # uncomment for data loading from GLO
        #self.create_timelines(self.rapid)  # timelines for households from .mat-file
        
    def create_timelines(self,rapid):
        if rapid:
            self.flat_timelines()
        else:
            self.noisy_timelines()

    def noisy_timelines(self): 
    # timelines with household loads as noise and realistic charging timelines
        for wallbox_item in self.system.wallboxes:
            wallbox_item.make_random_connection_timeline(self.start_minute, self.end_minute)

        self.household_timeline_handler = Household_Timeline_Handler(self.system, household_load_file)
        self.household_timeline_handler.make_village_household_loads(start_minute=self.start_minute, 
                                                                     end_minute=self.end_minute)
    def flat_timelines(self): 
        # noise free timelines for quick and clean CL-matrix generation
        no_of_frames = len(self.system.wallboxes)*2+3 # number of days to be simulated
        for wallbox_item in self.system.wallboxes:
            wallbox_item.make_random_connection_timeline_rapid(self.start_minute, self.end_minute)
    
        self.start_minute = 0
        self.end_minute   = no_of_frames
               
        for item in self.system.households:
            item.timeline = numpy.zeros(no_of_frames ).tolist()
    
    def set_loads(self, voltage_control):  # gets loads from households and wallboxes, writes to PandaGrid
        for household_item in self.system.households:
            self.system.panda_grid.load.p_mw[household_item.position_load] = (
                household_item.timeline[self.frame] * 1e-6)
            self.system.panda_grid.load.q_mvar[household_item.position_load] = (
                household_item.timeline[self.frame] * 0.3 * 1e-6)

        for wallbox_item in self.system.wallboxes:
            self.system.panda_grid.load.p_mw[wallbox_item.position_load] = (
                wallbox_item.get_power(self.frame, voltage_control=voltage_control)* 0.001)
            wallbox_item.adapt_ctrl_power_cap()

    def log_data(self, data, name):  # logs sim results into self.results
        
        if name not in self.results.keys():  # if no item of that name exists yet (first iteration)
            self.results[name] = [] # add new result item to list
        self.results[name].append(data)  # add value to data list of results object, find results item by name

    # hier würden dann die Ergebnisse der Optimierung mit reinkommen => die mal überschreiben => run_sim_stepwise oder sowas
    def run_sim(self, voltage_control=False):  # main loop for timeseries sim
        
        for self.frame in range(1, self.end_minute - self.start_minute - 1):
            #if self.frame / 100 == round(self.frame / 100):
            if self.frame % 100 == 0:
                print("Iteration no. "+ str(self.frame)+ " of "+ str(self.end_minute - self.start_minute))
            
            self.set_loads(voltage_control)
            pp.runpp(self.system.panda_grid)  # LFNR from PandaPower
            
            # gather data that is recorded as simulation results, e.g. bus powers and voltages
            caps    = [x.ctrl_power_cap for x in self.system.wallboxes] # Wallbox power caps from P(U)-controls:
            S       = [x+1j*y for x,y in zip(self.system.panda_grid.load.p_mw, # Power on all busses
                                             self.system.panda_grid.load.q_mvar)] 
            U       = [cmath.rect(self.system.panda_grid.res_bus.vm_pu[x], # Voltages on all busses
                                  self.system.panda_grid.res_bus.va_degree[x]*math.pi/180) 
                       for x in range(len(self.system.panda_grid.res_bus.vm_pu))] 
            u_min   = min([abs(x) for x in U]) # lowest voltage magnitude in the system (for the P(U)-controls)
            I_lines = [1000*sim_handler_1.system.panda_grid.res_line['i_from_ka'][index] # Line current magnitudes
                       for index in range(len(sim_handler_1.system.panda_grid.res_line))]
            
            
            #self.log_data(data= caps,   name="wallbox_caps")  # cap of P(U)-controls
            self.log_data(data= U,       name="U")             # complex voltages on all busses
            self.log_data(data= u_min,   name="u_min")         # complex powers on all busses
            self.log_data(data= S,       name="S")             # complex powers on all busses
            self.log_data(data= I_lines, name="I_lines")       # complex powers on all busses
            
        self.results['pos_U_WB'] = [x.position_bus      # indexes of Wallboxes Voltages in voltages list
                                    for x in self.system.wallboxes]  
        self.results['pos_U_HH'] = [x.position_bus      # indexes of Households Voltages in voltages list 
                                    for x in self.system.households] 
        self.results['pos_S_WB'] = [x.position_load     # indexes of Wallboxes Powers in voltages list 
                                    for x in self.system.wallboxes]  
        self.results['pos_S_HH'] = [x.position_load     # indexes of Households Powers in voltages list 
                                    for x in self.system.households] 
        self.results['Y']        = PSL.get_Y(self.system.panda_grid)
        
        print("Simulation finished (voltage controls="+str(voltage_control)+', rapid='+str(self.rapid)+')')


    def run_GLO_sim(self, household_data, wallbox_data, timesteps):
        for step in range(timesteps):
            # set household loads
            for bus in self.system.grid.load.index:
                self.system.grid.load.loc[bus, 'p_mw'] = household_data[bus][step]

            # add wallbox loads (current*voltage(assumed))
            for wallbox_bus in wallbox_data.keys():
                self.system.grid.load.loc[wallbox_bus, 'p_mw'] += wallbox_data[wallbox_bus][step]

            # all loads set => start simulation
            pp.runpp(self.system.grid)

            # store results



    def plot_powers(self):
        for bus_no in range(len(sim_handler_1.results['S'][0])):
            PSL.plot_list([sim_handler_1.results['S'][frame][bus_no] for frame in range(len(sim_handler_1.results['S']))],
                  title='timeline power bus no. '+str(bus_no))
        
# MAIN-Routine==================================================================================================
if __name__ == '__main__':
    #grid_name='Example_Grid'
    grid_name='Example_Grid_Simple'
    #grid_name='selfmade_grid'

    #grid_name='Example_Grid_Current_Estimation'

    excel_file='grids/'+grid_name+'.xlsx'
    save_file='sav/'+grid_name+'.pic'
    results_dict_file='sav/'+grid_name+'_results.pic'
    household_load_file="zeitreihen/Verbrauchsdaten_IOW"
    handler_file='sav/'+grid_name+'_handler.pic'
    panda_file='sav/'+grid_name+'_panda.pic'
    rapid=False

    if rapid==True:
        results_dict_file='sav/'+grid_name+'_results_rapid.pic'

    if True:  # Simulation rather than loading from file
        system_1 = Low_Voltage_System(line_type='NAYY 4x120 SE',transformer_type="0.25 MVA 10/0.4 kV")
        system_1.panda_grid = system_1.make_system_from_excel_file(file=excel_file)
        system_1.grid_from_GLO('grids/selfmade_grid.xlsx', {'S transformer': 240,
                                                            'line impedance': 0.004,
                                                            'i line max': 140
                                                            })
        # simulate uncontrolled system:
        sim_handler_1 = Simulation_Handler(system_1,
                                           start_minute=60 * 12,
                                           end_minute=60 * 12 + 24 * 60,
                                           rapid=rapid)

        if False: # Simulation w/ voltage controls
            sim_handler_2 = copy.deepcopy(sim_handler_1) # run voltage controlled system simulation
            sim_handler_2.run_sim(voltage_control=True)  # run voltage controlled system simulation
        sim_handler_1.run_sim(voltage_control=False)

        if True: # Save simulation results
            with open(results_dict_file, "wb") as f:  # save to file
                pickle.dump(sim_handler_1.results, f)     # write to file



            with open(panda_file, "wb") as f:  # save to file
                pickle.dump(system_1.panda_grid, f)     # write to file
            if True:
                with open(handler_file, "wb") as f:  # save to file
                    #pickle.dump(sim_handler_1, f)     # write to file
                    pass # uncomment above to avoid error

    else:  # Load simulation results from file rather than run simulation
        with open(results_dict_file,    "rb")   as f:
            res = pickle.load(f)    # read from file
        with open(panda_file,           "rb")   as f:  # save to file
            pg_orig = pickle.load(f)     # write to file
        if True:
            with open(handler_file,         "rb")   as f:
                sim_handler_1 = pickle.load(f)    # read from file

    sim_handler_1.plot_powers()
    # x=[sim_handler_1.results['u_min'], sim_handler_2.results['u_min']]
    # import numpy as np
    # x=np.array(x).T.tolist()
    # PSL.plot_list(x)

    #[1000*sim_handler_1.system.panda_grid.res_line['i_from_ka'][index] for index in range(len(sim_handler_1.system.panda_grid.res_line))]


    # S_WB=sim_handler_1.results['pos_S_WB']
    # S_HH=sim_handler_1.results['pos_S_HH']

    # PSL.plot_list([sum([frame[WB]*1000 for WB in sim_handler_1.results['pos_S_WB']]) for frame in sim_handler_1.results['S']],title='E-Mobilitäts-Lastprofil, kumuliert, 30 Fahrzeuge')
    # PSL.plot_list([sum([frame[WB]*1000 for WB in sim_handler_2.results['pos_S_WB']]) for frame in sim_handler_2.results['S']],title='E-Mobilitäts-Lastprofil, kumuliert, 30 Fahrzeuge, geregelt')
    # PSL.plot_list([[frame[HH] for HH in sim_handler_1.results['pos_S_HH']] for frame in sim_handler_1.results['S']])
    # PSL.plot_list([[frame[HH]*1000 for HH in [6,8]] for frame in sim_handler_1.results['S']],title="Haushaltslast")















