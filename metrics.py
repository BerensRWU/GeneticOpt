import numpy as np

def car2cylin(pc):
    rho = np.sqrt(pc[:,0]**2 + pc[:,1]**2)
    phi = np.arctan2(pc[:,1], pc[:,0])
    z = pc[:,2]
    return np.array([rho,phi,z])

def cart2sphe(pc):
    rho = pc[:,0]**2 + pc[:,1]**2
    r = np.sqrt(rho + pc[:,2]**2)
    theta = np.arctan2(pc[:,2], np.sqrt(rho))
    phi = np.arctan2(pc[:,1],pc[:,0])
    return np.array([r,theta,phi])

def lob1(grid, n_lidar=3):
    return np.float32(grid != 0)
    
def lob2(grid, n_lidar=3):
    return np.float32(grid)/n_lidar
    
def lob3_2(grid, n_lidar=3):
    max_value = 2-1/2**(n_lidar-1)
    return np.float32(2-1/2**(grid-1))/max_value
    
def lob3_4(grid, n_lidar=3):
    max_value = 4-1/4**(n_lidar-1)
    return np.float32(4-1/4**(grid-1))/max_value
    

def metric_fun(pc, n_azimuth = 100, n_z = 32, n_lidar=3,
            weight_front = 1, weight_back = 1, weight_left = 1, weight_right = 1,
            metric = lob3_2):
    #z_min = 1.299
    grid = np.zeros((n_z, n_azimuth))

    for i in range(n_lidar):
        pc0 = pc[pc[:,3] == i]

        rho0, phi0, z0 = car2cylin(pc0)
        mask0 = np.abs(rho0-9.5) < 0.05
        
        z0 = np.min([(-z0[mask0] ) / 4 *  n_z, np.zeros(mask0.sum()) + n_z-1], axis = 0)
        phi0 = np.min([(phi0[mask0] + np.pi) / (2*np.pi) * n_azimuth, np.zeros(mask0.sum()) + n_azimuth-1], axis = 0)
        
        grid[np.int32(z0),np.int32(phi0)] += 1
        
    lob  = metric(grid, n_lidar)
    weight = np.concatenate([weight_back + np.zeros((n_z,int(np.ceil(n_azimuth/8)))),
                             weight_right + np.zeros((n_z,int(n_azimuth/4))),
                             weight_front + np.zeros((n_z,int(n_azimuth/4))), 
                             weight_left + np.zeros((n_z,int(n_azimuth/4))), 
                             weight_back + np.zeros((n_z,int(n_azimuth/8)))],axis = 1)
    return lob * weight
    
if __name__ =="__main__":
    from sklearn.metrics import auc
    n_lidar = 3
    n_azimuth = 300
    n_z = 20
    grid = np.ones((n_z,n_azimuth))*n_lidar
    lob = lob2(grid, n_lidar)
    
    weight = np.concatenate([2 + np.zeros((n_z,int(np.ceil(n_azimuth/8)))),
                             1 + np.zeros((n_z,int(n_azimuth/4))),
                             2 + np.zeros((n_z,int(n_azimuth/4))), 
                             1 + np.zeros((n_z,int(n_azimuth/4))), 
                             2 + np.zeros((n_z,int(n_azimuth/8)))],axis = 1)
    a = lob * weight
    print(a.sum()/(n_azimuth*int(n_z)))
    print(auc(np.arange(120,360,60),np.array([1.5,1.5,1.5,1.5])))
