import glob
import os
import sys


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

try:
    sys.path.append("../utils/")
except IndexError:
    pass

global one_box_size
one_box_size = 1

start_location = carla.Location(0, 0, 2)

def place_car(world,name):
    vehicle_bp = world.get_blueprint_library().filter(name.lower())[0]
    return world.spawn_actor(vehicle_bp, carla.Transform(start_location, carla.Rotation(0,0,0)))

def place_boxes(phi, start_location, world, h= 1,l=1,w=1,car=False):
    if car:
        x = 0
        y = 0
    else:
        x = 10*np.cos(phi/180*np.pi)
        y = 10*np.sin(phi/180*np.pi)
    length = l
    width = w
    height = h
    yaw = phi
    cat = 0
    length, width, height = int(length), int(width), int(height)
    x, y = loc_sens_x2world(x, start_location), loc_sens_y2world(y, start_location)    
    box_bp = world.get_blueprint_library().filter('box02*')[0]
    yaw_rad = yaw * np.pi/180
    rot = np.array([np.cos(yaw_rad),-np.sin(yaw_rad),np.sin(yaw_rad),np.cos(yaw_rad)]).reshape(2,2)
    for i in range(length):
        for j in range(width):
            b_xy_1 = np.array([x, y]) + np.dot(rot, np.array([one_box_size*i                         + (1/2 * 0.6486 - length/2 * one_box_size),
                                                              one_box_size*j                         + (1/2 * 0.6486 - width/2*one_box_size )]))
            b_xy_2 = np.array([x, y]) + np.dot(rot, np.array([one_box_size*i                         + (1/2 * 0.6486 - length/2 * one_box_size),
                                                              one_box_size*j + one_box_size - 0.6486 + (1/2 * 0.6486 - width/2*one_box_size )]))
            b_xy_3 = np.array([x, y]) + np.dot(rot, np.array([one_box_size*i + one_box_size - 0.6486 + (1/2 * 0.6486 - length/2 * one_box_size),
                                                              one_box_size*j                         + (1/2 * 0.6486 - width/2*one_box_size )]))
            b_xy_4 = np.array([x, y]) + np.dot(rot, np.array([one_box_size*i + one_box_size - 0.6486 + (1/2 * 0.6486 - length/2 * one_box_size),
                                                              one_box_size*j + one_box_size - 0.6486 + (1/2 * 0.6486 - width/2*one_box_size )]))
            for k in range(height):
                if i == 0 or j == 0 or k == 0 or i == length-1 or j == width - 1 or k == height -1:
                    transform1 = carla.Transform(carla.Location(b_xy_1[0], b_xy_1[1], one_box_size*k + 0.009252), carla.Rotation(0.,yaw,0.))
                    transform2 = carla.Transform(carla.Location(b_xy_2[0], b_xy_2[1], one_box_size*k + 0.009252), carla.Rotation(0.,yaw,0.))
                    transform3 = carla.Transform(carla.Location(b_xy_3[0], b_xy_3[1], one_box_size*k + 0.009252), carla.Rotation(0.,yaw,0.))
                    transform4 = carla.Transform(carla.Location(b_xy_4[0], b_xy_4[1], one_box_size*k + 0.009252), carla.Rotation(0.,yaw,0.))
                    
                    transform5 = carla.Transform(carla.Location(b_xy_1[0], b_xy_1[1], one_box_size*k + 0.009252 + (one_box_size - 0.6486)), carla.Rotation(0.,yaw,0.))
                    transform6 = carla.Transform(carla.Location(b_xy_2[0], b_xy_2[1], one_box_size*k + 0.009252 + (one_box_size - 0.6486)), carla.Rotation(0.,yaw,0.))
                    transform7 = carla.Transform(carla.Location(b_xy_3[0], b_xy_3[1], one_box_size*k + 0.009252 + (one_box_size - 0.6486)), carla.Rotation(0.,yaw,0.))
                    transform8 = carla.Transform(carla.Location(b_xy_4[0], b_xy_4[1], one_box_size*k + 0.009252 + (one_box_size - 0.6486)), carla.Rotation(0.,yaw,0.))
                    
                    world.spawn_actor(box_bp, transform1)
                    world.spawn_actor(box_bp, transform2)
                    world.spawn_actor(box_bp, transform3)
                    world.spawn_actor(box_bp, transform4)
                    world.spawn_actor(box_bp, transform5)
                    world.spawn_actor(box_bp, transform6)
                    world.spawn_actor(box_bp, transform7)
                    world.spawn_actor(box_bp, transform8)

def loc_world2sens_x(x, sensor_location):
    return x - sensor_location.x
    
def loc_world2sens_y(y, sensor_location):
    return y - sensor_location.y
    
def loc_world2sens_z(z, sensor_location):
    return z - sensor_location.z
    
def loc_sens_x2world(x, sensor_location):
    return x + sensor_location.x
    
def loc_sens_y2world(y, sensor_location):
    return y + sensor_location.y
    
def loc_sens_z2world(z, sensor_location):
    return z + sensor_location.z


def loading_carla(load_world = False):
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(4.0)
        
        
        if load_world:
            world = client.get_world()
            
            settings = world.get_settings()
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
            
            
            world.unload_map_layer(carla.MapLayer.All)
            world.load_map_layer(carla.MapLayer.Ground)
            
            spectator = world.get_spectator()
            transform = carla.Transform(carla.Location(x=-4, z=2.5), carla.Rotation())
            spectator.set_transform(transform)
            
            #if world.get_map().name != "Town03":
            #    world = client.load_world("Town03")

            world.set_weather(getattr(carla.WeatherParameters, "CloudyNoon"))

            print("Placing Objects")
            for phi in np.arange(0,360,4.0107, dtype = float):
                place_boxes(phi, start_location, world, h=4,l=1,w=1)
            return world
        else:
            world = client.get_world()
            
            settings = world.get_settings()
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
            
            return world
    finally:
        print("loading finished")
        
if __name__ == "__main__":
    world = loading_carla(True)
    vehicle = place_car(world, "Vehicle.Audi.Etron")
    #place_boxes(0, start_location, world, h= 2,l=4,w=2, car=True)
