from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import os
import sys
import glob

try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import carla

import numpy as np
from time import sleep
from time import time
from sklearn.metrics import auc
import copy
from graph_tool import load_graph

from metrics import metrics
from opt_utils_multy import Evolution, Optimization
from load_carla import loading_carla, place_car
from scipy.spatial.transform import Rotation as R

start_location = carla.Location(250.,379.,-12.5)

def rt_matrix(a,b,c):
    c_p = np.cos(np.radians(a))
    s_p = -np.sin(np.radians(a))
    c_y = np.cos(np.radians(b))
    s_y = np.sin(np.radians(b))
    c_r = np.cos(np.radians(c))
    s_r = -np.sin(np.radians(c))
    matrix = np.array([[c_y*c_p, c_y*s_p*s_r - s_y*c_r, c_y*s_p*c_r + s_y*s_r],
                       [s_y*c_p, s_y*s_p*s_r + c_y*c_r, s_y*s_p*c_r + c_y*s_r],
                       [-s_p, c_p*s_r, c_p*c_r]])
    return matrix

def get_transform(start_loc, loc_shift, rotation_shift):
    loc = start_loc + carla.Location(loc_shift[0], loc_shift[1], loc_shift[2])
    rot = carla.Rotation(float(rotation_shift[0]), float(rotation_shift[1]), float(rotation_shift[2]))
    #rot = carla.Rotation(33,0,45)
    return carla.Transform(loc, rot)
    
def check_location(real_loc, trans_loc):
    return (np.abs(real_loc.x - trans_loc.x) < 0.001) and \
           (np.abs(real_loc.y - trans_loc.y) < 0.001) and \
           (np.abs(real_loc.z - trans_loc.z) < 0.001)

def check_rotation(real_rot, trans_rot):
    return (np.abs(real_rot.pitch - trans_rot.pitch) < 0.001) and \
           (np.abs(real_rot.yaw - trans_rot.yaw) < 0.001) and \
           (np.abs(real_rot.roll - trans_rot.roll) < 0.001)

def create_sensor(world, channels = 16, upper_fov=15, lower_fov=-15, points_per_second = 300000, loc_shift = [0,0,0], rot_shift = [0,0,0]):
        print("Spawn LiDAR")
        puck_bp = None
        puck_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        puck_bp.set_attribute('channels',str(channels))
        puck_bp.set_attribute('points_per_second',str(points_per_second))
        puck_bp.set_attribute('rotation_frequency',str(20))
        puck_bp.set_attribute('range',str(50))
        puck_bp.set_attribute('upper_fov',str(upper_fov))
        puck_bp.set_attribute('lower_fov',str(lower_fov))
        puck_bp.set_attribute('dropoff_general_rate',str(0))
        puck_bp.set_attribute('dropoff_intensity_limit',str(0))
        puck_bp.set_attribute('dropoff_zero_intensity',str(0))
        puck_bp.set_attribute('atmosphere_attenuation_rate',str(0))
        
        return world.spawn_actor(puck_bp, get_transform(start_location, loc_shift, rot_shift), attach_to=None)
   
def fitness(opt, sensor, graph, epoch):
    global lidar_global
    global lidar_listend
    lidar = []
    lidar_number = []
    
    n_sensors = opt.value.shape[1]
    lidar_listend = np.zeros(n_sensors)
    for i in range(n_sensors):
        lidar_global = 1
        sensor_transformation = get_transform(start_location, graph.vertex_properties["pos"][graph.vertex(opt.value[0,i])], opt.value[1:4,i])
        #sensor_transformation = get_transform(start_location, [i,0,0], opt.value[1:4,i])
        sensor.set_transform(sensor_transformation)
        
        time_start = time()
        while (not check_location(sensor.get_location(), sensor_transformation.location)) or \
              (not check_rotation(sensor.get_transform().rotation, sensor_transformation.rotation)):
            # Wait till the sensor is at the correct spot
              if  time() -time_start > 10:
                print("Problem with the placing of the sensor.\n",
                       sensor.get_location(), sensor_transformation.location,
                       sensor.get_transform().rotation, sensor_transformation.rotation)
                break
        sensor.listen(lambda point_cloud: save_pc(point_cloud, epoch, i))
        time_start = time()
        while type(lidar_global) == int:
            # Wait till the sensor has measured the scene
            if  time() - time_start > 10:
                print("Problem with the listening of the sensor.")
                break
        
        sensor.stop()
        rot_mat = R.from_euler("yzx",[sensor.get_transform().rotation.pitch,
                                      sensor.get_transform().rotation.yaw,
                                      sensor.get_transform().rotation.roll],
                                      degrees=True).as_matrix()
        lidar_temp = copy.deepcopy(lidar_global)
        lidar_temp = np.dot(np.linalg.inv(rot_mat), lidar_temp.T).T
        lidar.append(lidar_temp + [sensor.get_location().x - start_location.x,
                                   sensor.get_location().y - start_location.y,
                                   sensor.get_location().z - start_location.z])      
        lidar_number.append([i]*lidar_temp.shape[0])
    lidar = np.concatenate(lidar)
    lidar_number = np.concatenate(lidar_number)

    lidar = np.concatenate((lidar, lidar_number.reshape(-1,1)), 1)
    
    m1 = []
    m2 = []
    m3 = []
    grid_sizes = np.arange(120,1260,60)
    for gs in grid_sizes:
        _, _, r3 = metrics(lidar, n_azimuth = gs, n_z = int(gs/15), n_lidar = n_sensors)#32)
        #m1.append(r1.sum()/(gs*int(gs/15)))
        #m2.append(r2.sum()/(gs*int(gs/15)))
        m3.append(r3.sum()/(gs*int(gs/15)))
    #auc(grid_sizes, m1)
    #auc(grid_sizes, m2)
    return auc(grid_sizes, m3)
        
def save_pc(pc, epoch, number):
    global lidar_global
    global lidar_listend
    if lidar_listend[number] == 0:
        lidar_listend[number] = 1
        pc = np.frombuffer(pc.raw_data, dtype=np.dtype('f4'))
        if len(pc):
            pc = np.unique(np.round(pc.reshape(-1,4)[:,0:3], 3), axis = 0)
            #a = copy.deepcopy(pc[:,1])
            #pc[:,1] = copy.deepcopy(pc[:,0])
            #pc[:,0] = copy.deepcopy(a)
            lidar_global = copy.deepcopy(pc)
        else:
            lidar_global = np.array([]).reshape(-1,4)[:,0:3]
        
def save_epoch(best_sensors_epoch, metric_epoch, epoch, out_root, graph):
    with open(out_root + "metric.csv", "a+") as f:
        f.write("%d, %f\n" %(epoch, metric_epoch))
    np.save(out_root + "/best_config/%06d" % (epoch), best_sensors_epoch)
    if np.mod(epoch,50) == 0:
        global lidar_global
        global lidar_listend
        lidar = []
        lidar_number = []
        n_sensors = best_sensors_epoch.shape[1]
        lidar_listend = np.zeros(n_sensors)
        
        for i in range(n_sensors):
            lidar_global = 1
            sensor_transformation = get_transform(start_location, graph.vertex_properties["pos"][graph.vertex(best_sensors_epoch[0,i])], best_sensors_epoch[1:4,i])
            sensor.set_transform(sensor_transformation)
            
            time_start = time()
            while (not check_location(sensor.get_location(), sensor_transformation.location)) or \
                  (not check_rotation(sensor.get_transform().rotation, sensor_transformation.rotation)):
                # Wait till the sensor is at the correct spot
                  if  time() -time_start > 10:
                    print("Problem with the placing of the sensor.\n",
                           sensor.get_location(), sensor_transformation.location,
                           sensor.get_transform().rotation, sensor_transformation.rotation
                           )
                    break
            sensor.listen(lambda point_cloud: save_pc(point_cloud, epoch, i))
            
            time_start = time()
            while type(lidar_global) == int:
                # Wait till the sensor has measured the scene
                if  time() - time_start > 10:
                    print("Problem with the listening of the sensor.")
                    break
            
            sensor.stop()
            rot_mat = R.from_euler("yzx",[sensor.get_transform().rotation.pitch,
                                      sensor.get_transform().rotation.yaw,
                                      sensor.get_transform().rotation.roll],
                                      degrees=True).as_matrix()
        
            lidar_temp = copy.deepcopy(lidar_global)
            lidar_temp = np.dot(np.linalg.inv(rot_mat), lidar_temp.T).T
            lidar.append(lidar_temp + [sensor.get_location().x - start_location.x,
                                   sensor.get_location().y - start_location.y,
                                   sensor.get_location().z - start_location.z])
            lidar_number.append([i]*lidar_temp.shape[0])
        lidar = np.concatenate(lidar)
        lidar_number = np.concatenate(lidar_number)

        lidar = np.concatenate((lidar, lidar_number.reshape(-1,1)), 1)
        np.save(out_root + "lidar/%06d" % epoch, lidar)
    

if __name__=="__main__":
    np.random.seed(11)
    name = "Vehicle.Audi.Etron"
            
    graph = load_graph("../Oberflaeche/%s.gt" % name)
    
    world = loading_carla(True)
    vehicle = place_car(world, name)
    print("Placed Car")
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    try:
        sensor = create_sensor(world, channels = 16, upper_fov=15, lower_fov=-15, points_per_second = 300000)
        for n_sensors in range(1,5):
            
            out_root = "_out3/%s_%d_sensors/" % (name, n_sensors)
            
            if not os.path.exists(out_root + "/best_config"):
                    os.makedirs(out_root + "/best_config")
                    
            if not os.path.exists(out_root + "/lidar"):
                    os.makedirs(out_root + "/lidar")
                    
            params = {'constrains_pitch': lambda p: p > -89 and p < 89,
                      'constrains_roll': lambda r: r > -89 and r < 89,
                      'lower_bound_pitch': -89,
                      'lower_bound_roll': -89,
                      'upper_bound_pitch': 89,
                      'upper_bound_roll': 89,
                      'rate_loc': 20,
                      'rate_rot': lambda ep: max([10/(1+np.exp(ep/250)), 0.5]),
                      'n_sensors': n_sensors}
                      
            n_epochs = 2000
            best_sensors_epoch = []
            metric_epoch = []

            evo = Evolution(
                pool_size=20, fitness=fitness, individual_class=Optimization, n_offsprings=6,
                pair_params={'alpha': 0.5},
                mutate_params=params,
                init_params=params,
                sensor=sensor,
                graph=graph)
            i = 0
            s_time = time()
            while i < n_epochs:
                print("Epoch: ", i)
                metric_epoch.append(evo.step(sensor = sensor, epoch = i))
                best_sensors_epoch.append(evo.pool.individuals[-1].value)     
                print("Best conf.")
                for j in range(n_sensors):
                    print("For Sensor: ", j)
                    print(graph.vertex_properties["pos"][graph.vertex(best_sensors_epoch[-1][0,j])])
                    print(best_sensors_epoch[-1][:,j])
                print("Current metric: ", metric_epoch[-1])

                save_epoch(best_sensors_epoch[-1], metric_epoch[-1], i, out_root,graph)
                i += 1
                print(f"Epoch {i} time: {time() - s_time}")
    finally:
        sensor.destroy()
        print("Destroyed Sensors")
        vehicle.destroy()
        print("Destroyed Car")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
