# -*- coding: Latin-1 -*-
# TP optim : maximisation of the area
# par l'algorithme PSO
# Peio Loubiere & Rachid Chelouah pour l'EISTI
# septembre 2017
# use : python MaxRect_SA.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import pyclipper
from pyclipper import scale_to_clipper, scale_from_clipper
from functools import partial
from functools import reduce
import time
time.millis = lambda: int(round(time.time() * 1000))

# ************ Parameters of metaheuristics *****************
INF = 10
T0 = 5000
IterMax = 15000
# ***********************************************************

# ***************** Problem settings ************************
#  Different proposals of parcels:
#polygon = ((10,10),(10,400),(400,400),(400,10))
#polygon = ((10,10),(10,300),(250,300),(350,130),(200,10))
#polygon = ((50,150),(200,50),(350,150),(350,300),(250,300),(200,250),(150,350),(100,250),(100,200))
polygon = ((50,50),(50,400),(220,310),(220,170),(330,170),(330,480),(450,480),(450,50))
# ***********************************************************


# Collect bounding box bounds around the parcel
def getBounds(polygon):
    xmin = reduce(lambda acc, e: min(acc,e[0]),polygon[1:],polygon[0][0])
    xmax = reduce(lambda acc, e: max(acc,e[0]),polygon[1:],polygon[0][0])
    ymin = reduce(lambda acc, e: min(acc,e[1]),polygon[1:],polygon[0][1])
    ymax = reduce(lambda acc, e: max(acc,e[1]),polygon[1:],polygon[0][1])
    return xmin, xmax, ymin, ymax

# Transformation of a problem solution into a rectangle for clipping
# Return the rectangle (A(x1,y1), B(x2,y2), C(x3,y3), D(x4,y4))
# Cx,Cy : centrer point
def pos2rect(rect):
    Cx, Cy, angle, W, H = rect
    alpha = np.tan(W / H)
    Ox = W/2
    Oy = H/2
    a = (Cx + Ox*np.cos(angle+alpha)- (Oy*np.sin(angle+alpha)), Cy + Ox*np.sin(angle+alpha) + (Oy*np.cos(angle+alpha)))
    b = (Cx + Ox*np.cos(angle-alpha)- (Oy*np.sin(angle-alpha)), Cy + Ox*np.sin(angle-alpha) + (Oy*np.cos(angle-alpha)))
    c = (Cx + Ox*np.cos(np.pi + angle+alpha)- (Oy*np.sin(np.pi +angle+alpha)), Cy + Ox*np.sin(np.pi + angle+alpha) + (Oy*np.cos(np.pi + angle+alpha)))
    d = (Cx + Ox*np.cos(np.pi + angle-alpha)- (Oy*np.sin(np.pi + angle-alpha)), Cy + Ox*np.sin(np.pi + angle-alpha) + (Oy*np.cos(np.pi + angle-alpha))) 
    return a, b, c, d

# Distance between two points (x1,y1), (x2,y2)
def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

# Area of the rectangle (A(x1,y1), B(x2,y2), C(x3,y3), D(x4,y4))
# 	= distance AB * distance BC
def area(rect):
    pa, pb, pc, pd = pos2rect(rect)
    return distance(pa, pb) * distance(pb, pc)

# Clipping
# Predicate that verifies that the rectangle is in the polygon
# Test if
# 	- there is an intersection (!=[]) between the figures and
#	- both lists with the same length
# 	- all the points of the rectangle belong to the result of clipping
# If error (~flat angle), return false
def verifConstraint(rect, polygon):
    pc = pyclipper.Pyclipper()
    rect = pos2rect(rect)
    pc.AddPath(scale_to_clipper(polygon), pyclipper.PT_CLIP, True)
    pc.AddPath(scale_to_clipper(rect), pyclipper.PT_SUBJECT, True)
    solution = scale_from_clipper(pc.Execute(pyclipper.CT_INTERSECTION,
                    pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD))
    if (len(solution) < 1): return False
    if len(solution[0]) != 4:
        return False
    for p1 in rect:
        tmp = False
        for p2 in solution[0]:
            if distance (p1,p2)< 0.1:
                tmp = True
        if tmp == False:
            return False
    return True

def initOne(polygon):
    res = None
    xmin, xmax, ymin, ymax = getBounds(polygon)
    d = min(xmax-xmin, ymax-ymin)
    while res == None:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        alpha = np.random.uniform(0, 2*np.pi)
        W = np.random.uniform(0, np.sqrt(T0)*d)
        H = np.random.uniform(0, np.sqrt(T0)*d)
        res = x, y, alpha, W, H
        if not verifConstraint(res, polygon): res = None
    draw(polygon, res)
    return res

counter = 200
def fluctuation(polygon, rect):
    global counter 
    counter -= 1
    if counter < 0:
        draw(polygon, rect)
        counter = 1000
    res = None
    tries = 10
    while res == None and tries > 0:
        tries -= 1
        x, y, angle, W, H= rect
        x += np.random.uniform(-INF, +INF) 
        y += np.random.uniform(-INF, +INF)
        angle += np.random.uniform(-INF, +INF)
        W += np.random.uniform(-INF, +INF)
        H += np.random.uniform(-INF, +INF)
        res = x, y, angle, W, H
        if not verifConstraint(res, polygon): res = None
    return res if tries > 0 else rect


##############################################################################################################

def metropolis(x_new, x_old, sysEnergy, system):
    energy_new = sysEnergy(x_new)
    energy_old = sysEnergy(x_old)
    delta = energy_new - energy_old
    # print(delta)
    if delta <= 0: # if improving,
        if energy_new <= system['best_energy']: # comparison to the best, if better, save and refresh the figure
            system['best_energy'] = energy_new
            system['best_point'] = x_new
        return (x_new, energy_new) # the fluctuation is retained, returns the neighbor
    else:
        if np.random.uniform() > np.exp(-delta/system['T']): # the fluctuation is not retained according to the proba
            return (x_old, energy_old)              # initial path
        else:
            return (x_new, energy_new)              # the fluctuation is retained, returns the neighbor

def solve(init, fluctuation, sysEnergy, T0, IterMax):
    Henergy = []     # energy
    Htime = []       # time
    HT = []           # temperature
    Hbest = []        # distance
    IterMax = 15000
    # ######################################### INITIALIZING THE ALGORITHM ####### #####################
    x = init()
    energy = sysEnergy(x)
    system = dict()
    system['best_point'] = x
    system['best_energy'] = energy
    t = 0
    system['T'] = T0
    iterStep = 9
    # ############################################ PRINCIPAL LOOP OF THE ALGORITHM ###### ######################
    millis = time.millis()
    # Convergence loop on criteria of number of iteration (to test the parameters)
    for i in range(IterMax):
    # Convergence loop on temperature criterion
         # cooling law enforcement
        while (iterStep > 0): 
          # choice of two random cities
            new = fluctuation(x)
            # application of the Metropolis criterion to determine the persisted fulctuation
            (x, energy) = metropolis(new, x, sysEnergy, system)
            iterStep -= 1
        # cooling law enforcement
        t += 1
        # rules of temperature decreases
        system['T'] = system['T']*0.9995
        iterStep = 9
        #historization of data
        if t % 2 == 0:
            Henergy.append(energy)
            Htime.append(t)
            HT.append(system.get('T'))
            Hbest.append(system.get('best_energy'))

    new_millis = time.millis()
    print("Time of execution: {}ms.".format((new_millis-millis)))
    ############################################## END OF ALGORITHM - DISPLAY RESULTS ### #########################
    drawStats(Htime, Henergy, Hbest, HT)

    print("a,b,c,d are", system['best_point'])
    print("Total area is ", -system['best_energy'])

    return (system, (Htime, Henergy, Hbest, HT))
#####################################################################################################################
#Display
fig = plt.figure()
canv = fig.add_subplot(1,1,1)
def poly2list(polygon):
    polygonfig = list(polygon)
    polygonfig.append(polygonfig[0])
    return polygonfig

def make_patch(poly):
    poly = poly2list(poly)
    codes = [Path.MOVETO]
    for i in range(len(poly)-2):
      codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    path = Path(poly, codes)
    patch = patches.PathPatch(path, facecolor='orange', alpha=0.4, lw=2)
    return patch

def draw(polygone,rect):
    global canv, codes
    canv.clear()
    canv.set_xlim(0,500)
    canv.set_ylim(0,500)
    canv.add_patch(make_patch(polygone))
    canv.add_patch(make_patch(pos2rect(rect)))
    # Title display (rectangle area)
    plt.title("area : {}".format(round(area(rect),2)))
    plt.draw()
    plt.pause(0.1)

def partialDraw(rect):
    global canv, codes
    canv.add_patch(make_patch(pos2rect(rect)))
    plt.title("area : {}".format(round(area(rect),2)))

def drawNew(polygone):
    global canv, codes
    canv.clear()
    canv.set_xlim(0,500)
    canv.set_ylim(0,500)
    canv.add_patch(make_patch(polygone))
    plt.draw()
    plt.pause(0.1)

def flush():
    plt.draw()
    plt.pause(0.1)

def drawStats(Htime, Henergy, Hbest, HT):
    # display des courbes d'evolution
    fig2 = plt.figure(2, figsize=(4, 6))
    plt.subplot(3,1,1)
    plt.semilogy(Htime, [-el for el in Henergy])
    plt.title("Evolution of the total energy of the system")
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.subplot(3,1,2)
    plt.semilogy(Htime, [-el for el in Hbest])
    plt.title('Evolution of the best energy')
    plt.xlabel('time')
    plt.ylabel('Best energy')
    plt.subplot(3,1,3)
    plt.semilogy(Htime, HT)
    plt.title('Evolution of the temperature of the system')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.show()

# SIMULATED ANNEALING
task = partial(initOne, polygon), partial(fluctuation, polygon), lambda x:-area(x)
solution = solve(*task, T0, IterMax)

