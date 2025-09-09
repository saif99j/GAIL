import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Env():
    def __init__(self, position=[[0., 0.]], orientation=[0.], speed=[0.2], velocity=math.pi/18, sim_time=0.1, unit=1., threshold=0.75, tolerate=5):
        self.points = unit * np.array([[0.7, 0], [0.25, 0.433], [-0.433, 0.25], [-0.433, -0.25], [0.25, -0.433]])
        self.position, self.pos = position, copy.deepcopy(position)
        self.orientation, self.ori = orientation, copy.deepcopy(orientation)
        self.speed, self.spe = speed, copy.deepcopy(speed)
        self.velocity = velocity
        self.sim_time = sim_time
        self.unit = unit
        self.threshold = threshold * unit
        self.tolerate = tolerate

    def reset(self):
        self.position = copy.deepcopy(self.pos)
        self.orientation = copy.deepcopy(self.ori)
        self.speed = copy.deepcopy(self.spe)

        s, done = self.observe()
        r = self.reward()
        
        return s
    
    def observe(self):
        x, y, z = self.position[0][0], self.position[0][1], self.orientation[0]
        sx = math.cos(z) * self.speed[0]
        sy = math.sin(z) * self.speed[0]
        s = [x, y, sx, sy]
        done = False
        x_to_endpt = x - 10
        y_to_endpt = y - 0
        dis_to_endpt = math.sqrt(x_to_endpt**2+y_to_endpt**2)
        for i in range(1, len(self.position)):
            x_, y_, z_ = self.position[i][0], self.position[i][1], self.orientation[i]
            sx_ = math.cos(z_) * self.speed[i]
            sy_ = math.sin(z_) * self.speed[i]
            dx_ = x - x_
            dy_ = y - y_
            dis = math.sqrt(dx_**2+dy_**2)
            
            s += [dis, dx_, dy_, sx_, sy_]
            if dis < self.threshold:
                done = True
                
        if abs(y) > self.tolerate * self.unit:
            done = True
         
#         if abs(y) < 1 and x > 4:
#             done = True

        if dis_to_endpt < 1.5:
            done = True
            
        return s, done

    def reward(self):
        x, y, z = self.position[0][0], self.position[0][1], self.orientation[0]
        
        for i in range(1, len(self.position)):
            x_, y_, z_ = self.position[i][0], self.position[i][1], self.orientation[i]
            dx_ = x - x_
            dy_ = y - y_
            dis = math.sqrt(dx_**2+dy_**2)
            
#           dist from curr pt to end pt
            x_to_endpt = x - 10
            y_to_endpt = y - 0
            dis_to_endpt = math.sqrt(x_to_endpt**2+y_to_endpt**2)

            if dis < self.threshold:
                print("Collision!")
                return -100
           
        if abs(y) < self.unit and y < 0:
            return 0.1
        elif abs(y) > self.tolerate * self.unit:
            print("Overboard!")
            return -20
#         elif abs(y) < 1 and x > 4:
#             print("resume route1111111111111111111111")
#             return 10
        # y < 0 is to keep turning right
        elif dis_to_endpt < 1.5 and y < 0:
            print("------Arrive end pt------")
            return 120
        else:
            # return -2 ** (1/dis_to_endpt) + 2 ** (1/(dis))

            # return -2 ** (-1/dis_to_endpt) - 2 ** (1/(dis+1)) * 0.5

            # return (-dis_to_endpt/10 - 10/dis)/10

            #good to env1 and env3
            # return 2 ** (1/dis) - 2 ** (1/dis_to_endpt)
            
            #good to env4
            return (1/dis_to_endpt)/7
    
    def step(self, a):
        a -= 1

        self.position[0][0] += self.speed[0] * self.sim_time * math.cos(self.orientation[0])
        self.position[0][1] += self.speed[0] * self.sim_time * math.sin(self.orientation[0])
        self.orientation[0] += self.velocity * self.sim_time * a
        
        for i in range(1, len(self.position)):
            self.position[i][0] += self.speed[i] * self.sim_time * math.cos(self.orientation[i])
            self.position[i][1] += self.speed[i] * self.sim_time * math.sin(self.orientation[i])

        s, done = self.observe()
        r = self.reward()
        
        return s, r, done, ''

    def render(self):

        points = self.rotate(self.points, -self.orientation[0])
        points = self.translate(points, self.position[0][0], self.position[0][1])
        poly = mpatches.Polygon(points, color= "r")
        polys = [poly]
        
        for i in range(1, len(self.position)):
            points = self.rotate(self.points, -self.orientation[i])
            points = self.translate(points, self.position[i][0], self.position[i][1])
            poly = mpatches.Polygon(points, color= "black")
            polys.append(poly)
            
        return polys
    
    def translate(self, points, dx, dy):
        points = copy.deepcopy(points)
        x, y = points[:, 0], points[:, 1]
        new_x = x + dx
        new_y = y + dy
        points[:, 0], points[:, 1] = new_x, new_y
        return points
        
    def rotate(self, points, rad):
        points = copy.deepcopy(points)
        x, y = points[:, 0], points[:, 1]
        new_x = x * math.cos(rad) + y * math.sin(rad)
        new_y = -x * math.sin(rad) + y * math.cos(rad)
        points[:, 0], points[:, 1] = new_x, new_y
        return points



envs = {
    'env1':{
        'position': [[0.,0.], [10., 0.]],
        'orientation': [0., math.pi],
        'speed': [0.2, 0.15]
    },
    'env2':{
        'position': [[0.,0.], [2.5, 0.]],
        'orientation': [0., 0],
        'speed': [0.15, 0.05]
    },
    'env3':{
        'position': [[0.,0.], [9., -4.]],
        'orientation': [0.,  3 * math.pi / 4],
        'speed': [0.15, 0.2]
    },
    'env4':{
        'position': [[0.,0.], [9., -4.], [11., 0.5]],
        'orientation': [0., 3 * math.pi / 4, math.pi],
        'speed': [0.2, 0.2, 0.2]
    },
    'env5':{
        'position': [[0.,0.], [5., 0.], [4., -4.], [3, 3]],
        'orientation': [0., math.pi, math.pi / 2, - math.pi / 2],
        'speed': [0.2, 0.2, 0.2, 0.2]
    },
}
