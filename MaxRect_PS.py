# -*- coding: Latin-1 -*-
# TP optim : maximisation of the area
# par l'algorithme PSO
# Peio Loubiere & Rachid Chelouah pour l'EISTI
# septembre 2017
# usa : python MaxRect_PS.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import pyclipper
from pyclipper import scale_to_clipper, scale_from_clipper
from functools import partial
from functools import reduce


# ***************** Problem settings ************************
#  Different proposals of parcels:
#polygon = ((10,10),(10,400),(400,400),(400,10))
#polygon = ((10,10),(10,300),(250,300),(350,130),(200,10))
#polygon = ((50,150),(200,50),(350,150),(350,300),(250,300),(200,250),(150,350),(100,250),(100,200))
polygon = ((50,50),(50,400),(220,310),(220,170),(330,170),(330,480),(450,480),(450,50))
psi = 0.7
cmax = 1.7
Nb_cycles =  500
Nb_particle = 20
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
def pos2rect(sol):
    Cx, Cy, angle, W, H = sol
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
def area(sol):
    pa, pb, pc, pd = pos2rect(sol)
    return distance(pa, pb) * distance(pb, pc)

# Clipping
# Predicate that verifies that the rectangle is in the polygon
# Test if
# 	- there is an intersection (!=[]) between the figures and
#	- both lists with the same length
# 	- all the points of the rectangle belong to the result of clipping
# If error (~flat angle), return false
def verifConstraint(sol, polygon):
    pc = pyclipper.Pyclipper()
    rect = pos2rect(sol)
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

############################################################################################
def initOne(polygon):
    res = None
    xmin, xmax, ymin, ymax = getBounds(polygon)
    d = min(xmax-xmin, ymax-ymin)
    while res == None:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        alpha = np.random.uniform(0, 2*np.pi)
        width = np.random.uniform(0, d/2)
        length = np.random.uniform(0, d/2)
        res = x, y, alpha, width, length
        if not verifConstraint(res, polygon): res = None
    draw(polygon, res)
    return res

# Init of the population
def initPop(nb, init, sysEnergy):
    pop = [init() for i in range(nb)]
    return [{"pos":el, "bestpos":el, "energy":sysEnergy(el),
        "vit":[0]*len(el), "bestenergy":sysEnergy(el)} for el in pop]

# Returns the best particle depends on the metaheuristic
def best(p1, p2, sysEnergy):
    return p1 if sysEnergy(p1["pos"]) < sysEnergy(p2["pos"]) else p2

# Return a copy of the best particle of the population
def getBest(population, sysEnergy):
    return dict(reduce(lambda acc, e: best(acc, e, sysEnergy),population[1:],population[0]))


# Update information for the particles of the population (swarm)
def update(particle,bestParticle):
    nv = dict(particle)
    if(particle["energy"] < particle["bestenergy"]):
        nv['bestpos'] = particle["pos"][:]
        nv['bestenergy'] = particle["energy"]
    nv['bestvois'] = bestParticle["bestpos"][:]
    return nv

# Calculate the limit
def limiting(position, velocity, validate):
    newpos = [p+v for p, v in zip(position, velocity)]
    if not validate(newpos):
        velocity = [-vel for vel in velocity]
        return position, velocity
    return newpos, velocity


# Calculate the velocity and move a paticule
def move(particle, cmax, psi, validate, sysEnergy):
    nv = dict(particle)
    dim = len(particle["pos"])
    velocity = [0]*dim
    for i in range(dim):
        velocity[i] = (particle["vit"][i]*psi + \
        cmax*np.random.uniform()*(particle["bestpos"][i] - particle["pos"][i]) + \
        cmax*np.random.uniform()*(particle["bestvois"][i] - particle["pos"][i]))
    position = list(particle["pos"])
    position, velocity = limiting(position, velocity, validate)
    nv["vit"] = velocity
    nv["pos"] = position
    nv["energy"] = sysEnergy(position)
    return nv

def solve(init, validate, sysEnergy, drawThing, drawNew, flush,psi, cmax,Nb_cycles,  Nb_particle):
    Htemps = []       # temps
    Hbest = []        # distance
    # initialization of the population
    swarm = initPop(Nb_particle, init, sysEnergy)
    # initialization of the best solution
    best = getBest(swarm, sysEnergy)
    best_cycle = best
    for i in range(Nb_cycles):
        #Update informations
        swarm = [update(e,best_cycle) for e in swarm]
        # velocity calculations and displacement
        swarm = [move(e, cmax, psi, validate, sysEnergy) for e in swarm]
        # Update of the best solution
        best_cycle = getBest(swarm, sysEnergy)
        if (best_cycle["bestenergy"] < best["bestenergy"]):
            best = best_cycle
            # draw(best['pos'], best['fit'])

        # historization of data
        if i % 10 == 0:
            Htemps.append(i)
            Hbest.append(best['bestenergy'])

        # swarm display
        if i % 20 == 0:
            drawNew()
            for el in swarm:
                drawThing(el["pos"])
            flush()

    # END, displaying results
    Htemps.append(i)
    Hbest.append(best['bestenergy'])
    print("Best area is ", -best['bestenergy'])
    return best

#################################################################################
# DISPLAY 
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

def draw(polygone,sol):
    global canv, codes
    canv.clear()
    canv.set_xlim(0,500)
    canv.set_ylim(0,500)
    canv.add_patch(make_patch(polygone))
    canv.add_patch(make_patch(pos2rect(sol)))
    plt.title("area : {}".format(round(area(sol),2)))
    plt.draw()
    plt.pause(0.1)

def partialDraw(sol):
    global canv, codes
    canv.add_patch(make_patch(pos2rect(sol)))
    plt.title("area : {}".format(round(area(sol),2)))

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

#PARTICLE SWARM
particle_task = partial(initOne, polygon), partial(lambda x, y :verifConstraint(y,x), polygon), lambda x:-area(x), partialDraw, partial(drawNew, polygon), flush
solve(*particle_task,psi, cmax, Nb_cycles, Nb_particle)
