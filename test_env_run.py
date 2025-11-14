from SumoEnv import SumoEnv
print('Creating env')
env = SumoEnv(use_gui=False, sumocfg_file='SUMO_Trinity_Traffic_sim/osm.sumocfg')
print('Resetting')
obs, info = env.reset()
print('Initial obs:', obs)
env.close()
print('Done')
