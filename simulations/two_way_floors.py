# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:32:29 2024

@author: rbruins
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
from itertools import count
from collections import defaultdict
import opensees.openseespy as ops

def single_span(plate,L,B):
    return np.pi / 2 * np.sqrt(plate.EIx/(plate.rho*L**4))


def f_sbr(plate,L,B):
    f_1 = np.pi / 2 * np.sqrt(plate.EIx/(plate.rho*L**4))
    f_2 = np.sqrt(1+(2*(L/B)**2+(L/B)**4)*plate.EIy/plate.EIx)
    
    return f_1 * f_2

def f_en_single(plate,L,B):
    EIx = plate.EIx
    EIy = plate.EIy
    p = plate.rho
    
    
    f_1 = np.pi / (2 * L**2) * np.sqrt(EIx/p)
    return f_1

def f_en(plate,L,B):
    
    EIx = plate.EIx
    EIy = plate.EIy
    p = plate.rho
    
    f_1 = np.pi / (2 * L**2) * np.sqrt(EIx/p)
    
    k_e2 = np.sqrt(1+((L/B)**4 * EIy) /EIx)
    
    
    B_L2 = np.min([B,L],axis=0) /np.max([B,L],axis=0) 
    
    k_e1 = 1.0
    
    
    return k_e1*k_e2 * f_1

def f_lit(plate,L,B,m=1,n=1):
    a= L
    b = B
    
    EIx = plate.EIx
    EIy = plate.EIy
    vx = plate.vx
    vy = plate.vy
    p = plate.rho
    
    Dx = EIx / (1-vx*vy) 
    Dy = EIy / (1-vx*vy)
    Dk =  EIy / (2*(1+vx))
    DXY = EIx * vy + 2*Dk
    
    wmm = np.pi**2 / (a**2 * np.sqrt(p)) * np.sqrt(Dx*m**4 + 2 * DXY * m**2*n**2 * (a/b)**2 + Dy*n**4*(a/b)**4)

    return wmm / (2 * np.pi)

def f_lit2(plate,L,B,m=1,n=1):
    a= L
    b = B
    
    EIx = plate.EIx
    EIy = plate.EIy
    vx = plate.vx
    vy = plate.vy
    p = plate.rho
    
    Dx = EIx / (1-vx*vy) 
    Dy = EIy / (1-vx*vy) 
    # Dk =  EIy / (2*(1+vx))
    DXY = 0  #EIx * vy + 2*Dk
    
    wmm = np.pi**2 / (a**2 * np.sqrt(p)) * np.sqrt(Dx*m**4 + 2 * DXY * m**2*n**2 * (a/b)**2 + Dy*n**4*(a/b)**4)

    return wmm / (2 * np.pi)

def w(Amm,m,n,x,y,L,B):
    return np.sin(m*np.pi * x/ L) * np.sin(n*np.pi * y/ B)

def opensees(plate,L,B,mesh_size = 1.0,plate_width = -1,shell=True,output = True,torsion_reduction_factor=1):
        
    div = plate_width
    
    if div <=0:
        div = -1

    ms = mesh_size
    
    xdicr = int(L / ms)+1
    ydicr = int(B / ms)+1
    
    x = np.linspace(0,L,xdicr)
    y = np.linspace(0,B,ydicr)
    
    
    
    
    
    xx,yy = np.meshgrid(x,y)

    # zz = np.arange(len(yy.flatten())).reshape(xx.shape)
    
    ops.wipe()
    ops.model('basic','-ndm',3,'-ndf',6)
    
    nums = (np.arange(0,len(xx.flatten())) + 1).reshape(xx.shape)
    node_nums = [int(a) for a in nums.flatten()]
    all_ = nums.flatten()
    
    # middle of the plate
    mid = int( all_[int(len(all_) / 2)])

     
    if output:
        print("mid plate point")
        print(xx.flatten()[mid-1],yy.flatten()[mid-1])

    l_b = nums[:-1,:-1]
    r_b = nums[:-1,1:]
    r_t = nums[1:,1:]
    l_t = nums[1:,:-1]
    
    locs = [l_b,r_b,r_t,l_t]
    
    element_nodes = np.array([a.flatten() for a in locs]).T
    
    # print(element_nodes)
    
    
    
    fixed = np.unique([b for a in [nums[0,:], nums[-1,:], nums[:,0],nums[:,-1] ] for b in a])
    
    nodes = {}
    
    
    for i,(x,y) in enumerate(zip(xx.flatten(),yy.flatten()),1):
        nodes[i] = (x,y)
        if i in fixed:
            c = "r"
        else:
            c = "b"
        
        ops.node(i,x,y,0)
        
        if i == mid:
            c = "r"
        else:
            c = "k"
        
        # plt.plot(x,y,"o",c=c)
        if i in fixed:
            ops.fix(i,1,1,1,0,0,0)
            
    Ex = plate.Ex
    Ey = plate.Ey
    Ez = plate.Ex
    nu_xy = plate.vx
    nu_yz = plate.vy
    nu_zx = plate.vx
    

    
    
    masses = defaultdict(float)
    
    if shell:
            
        # shear in 
        
        Gxy = Ey / (2*(1+nu_xy)) * torsion_reduction_factor
        Gyz = Ex / (2*(1+nu_yz))
        Gzx = Ez / (2*(1+nu_zx))
           
        
        
        e_type = "ShellMITC4"
        
        mat_num = 1
        matTag = 1
        
        ops.nDMaterial('ElasticOrthotropic', matTag, Ex, Ey, Ez, nu_xy, nu_yz, nu_zx, Gxy, Gyz, Gzx, 0)
        ops.section('PlateFiber', 100 , matTag, h)
        mat_num = 100
        
        for e,_nodes in enumerate(element_nodes,1):
            # print(e,*_nodes,mat_num)
            
            e_nodes = _nodes
            
            # plate division
            if div > 0 and nodes[_nodes[0]][1] % div < 1e-5 and nodes[_nodes[0]][1] > 0:
                
                # add nodes

                n1 = int(_nodes[0])
                n2 = int(_nodes[1])
                
                offset = 100000                
                
                
                # create new node and fix the previous and current
                if n1 + offset not in nodes:
                    nodes[n1+offset] = nodes[_nodes[0]]
                    ops.node(n1+offset,*nodes[_nodes[0]],0)
                        
                    
                    ops.equalDOF(n1,n1+offset,1,2,3)

                if n2 + offset not in nodes:
                    nodes[n2+offset] = nodes[_nodes[1]]
                    ops.node(n2+offset,*nodes[_nodes[1]],0)
                    ops.equalDOF(n1,n1+offset,1,2,3)
                    
                n1 += offset
                n2 += offset
                
                
                # print(nodes[],nodes[_nodes[1]])
                # print([nodes[int(a)] for a in _nodes])
                
                e_nodes = [n1,n2,_nodes[2],_nodes[3]]
                
                
            
            ops.element(e_type,e,*[int(a) for a in e_nodes],mat_num)
            
            area = ms **2
            
            for a in e_nodes:
                masses[int(a)] += area / 4 * plate.rho
            
            
    else:
        
        eleTag = count()
        transfTag = 10
        ops.geomTransf('Linear', transfTag,0.0,0.0,1.0) 
        
        # X_direction
        for a in nums:
            # print(a)
            
            A = h * ms
            E = Ex
            G = Gxy
            
            _b = min(ms,h) / 2
            _a = max(ms,h) / 2
            
            # keep J = 0 as you need to fit J 
            J = _a*_b**3 * (16/3 -3.36 *_b/_a *(1-_b**4 /(12*_a**4))) 
            J /= 10
            # J = 1/3 * h**3 * ms  + 1e-5
            Iy = h**3 / 12. * ms
            Iz = ms**3 / 12. * h / 10
            
            for n1,n2 in zip(a,a[1:]):
                ops.element('elasticBeamColumn', next(eleTag), int(n1), int(n2), A ,E, G, J, Iy, Iz, transfTag)#, '-mass', *([mass]*3 + [0.0]*3))
                
                # length /2 * 
                masses[int(n1)] += ms / 2 * plate.rho * ms
                masses[int(n2)] += ms / 2 * plate.rho * ms

        for b in nums.T:
            # print(a)
            
            A =h * ms
            E = Ey
            G = Gxy
            
            _b = min(ms,h) / 2
            _a = max(ms,h) / 2
            # keep J = 0 as you need to fit J 
            
            J = _a*_b**3 * (16/3 -3.36 *_b/_a *(1-_b**4 /(12*_a**4))) 
            J /= 10
            #J = 1/3 * h**3 * ms + 1e-5
            Iy = h**3 / 12. * ms
            Iz = ms**3 / 12. * h
            
            for n1,n2 in zip(b,b[1:]):
                
                
                extra = []
                if div > 0:
                    
                    # release for CLT plates
                    
                    if nodes[n2][1] % div < 1e-5:
                        if nodes[n2][1] < B:
                            extra = ["-releasey",2]
                            
                ops.element('elasticBeamColumn', next(eleTag), int(n1), int(n2), A ,E, G, J, Iy, Iz, transfTag,*extra)#, '-mass', *([mass]*3 + [0.0]*3))
            
            
        #print(nums)
        #ddd
        
        #ops.element('elasticBeamColumn', eleTag, int(n1), int(n2), beam.A ,beam.E, beam.G, beam.J, beam.Iy*Iy_correction_factor, beam.Iz, transfTag, '-mass', *([beam.mass]*3 + [0.0]*3),*extra)
        
    for _node, mass in masses.items():
        ops.mass(_node,*[mass]*3,0,0,0)
        
    numEigen = 1 
    
    eigenValues = np.array(ops.eigen(numEigen))
    
    freqs = (eigenValues)**0.5 / (2*np.pi)

    
    
    
    ops.record() 
    
    
    
    mass_dist = np.array([masses[a] for a in node_nums])
    
    # ndf = GetNodalDisp("ms.txt")
    
    modal_masses_precentage = dict()
    
    
   
    
    if output:
        print("\tMode\tfreq\tmass")
        
    
    
    for i in range(numEigen):
        ev_data = np.array([ops.nodeEigenvector(a,i+1,3) for a in node_nums])
        
        # normalize
        ev_data /= np.max(np.abs(ev_data))
        
        
        
        zz= ev_data[nums-1]
        
        # size = np.product([a-1 for a in nums.shape])
        
    
        modal_masses_precentage[i] = np.sum(ev_data ** 2 * mass_dist) / np.sum(mass_dist)
        if output:
            print(f"\t{i:5}\t{freqs[i]:5.2f}\t{modal_masses_precentage[i]*100:5.1f}%")
        
        # plot mode shape
        # plt.figure()
        # plt.contourf(xx,yy,zz)
        # plt.axis("equal")
        # plt.show()
        # dd
    
        

    return freqs[0], modal_masses_precentage[0]



class PlateProperties:
    
    def __init__(self,Ex,ratio,h,p,vx=0,vy=0):
        self.Ex=  Ex
        self.Ey = self.Ex*ratio
        self.thickness = h
        self.rho = p
        
        self.density = p /h
        
        self.vx = vx
        self.vy = vy
        
    @property
    def I(self):
        return self.thickness ** 3 / 12

    @property
    def EIx(self):
        return self.Ex * self.I
    
    @property
    def EIy(self):
        return self.Ey * self.I

Ex = 10000E6

h = 0.25

I = h**3 / 12

ratio = 0.5


plate = PlateProperties(Ex,ratio,h,500)

L = 6
B = 30

num = 40

L_r = np.full(num, 6)
B_r = np.linspace(6,30,num)

def modal_mass(B,L,m,n):
    x = np.linspace(0,L,101)
    y = np.linspace(0,B,101)
    xx,yy = np.meshgrid(x, y)
    a = 1.0
    zz = w(a,m,n,xx,yy,L,B)

    ratio = np.sum(zz**2) / ((len(x) - 1)* (len(y)-1))
    # a = L
    # b = B
    
    # output function:
    # output = -(a*b* np.sin(2*np.pi*m)-2*np.pi *m) * (2*np.pi*n-np.sin(2*np.pi)) / (16*np.pi**2*m*n)
    # https://www.wolframalpha.com/input?i=integral+of+sin%5E2%28m*pi*x%2Fa%29*sin%5E2%28n*pi*y%2Fb%29+dxdy+from+0+to+a+and+0+to+b
    
    # for m is integer > 0 and n integer > 0
    # np.sin(2*np.pi(m))) == 0
    # hence
    # output2 = -((-2*np.pi *m) * (2*np.pi*n)) / (16*np.pi**2*m*n)
    # expanding function (step 1)
    #output2 = 4 * np.pi**2 * m*n / (16*np.pi**2*m*n)
    
    # step 2
    #output2 = 4  / 16
    
    # output = 0.25

    
    return ratio

###### analytical mode shape calcs    
# _max = 6


# print(f"fe analytical {B}x{L}")

# for m in range(1,_max):
#     for n in range(1,_max):
#         print(f"\t{f_lit(plate,L,B,m=m,n=n):5.2f}\t{modal_mass(B,L,m,n)*100:0.1f}%")
        
        # print()
    
fig,ax = plt.subplots(1,1,figsize = [10,5])

##### literature formulations

ax.axhline(f_en_single(plate,L,B),lw=1,ls="--",c="g",label="Single span")

#ax.plot(B_r, f_lit(plate,L_r,B_r),c="r",lw=10, label= "literature")
#ax.plot(B_r, f_lit2(plate,L_r,B_r),c="blue",lw=10, label= "literature2")

ax.plot(B_r, f_sbr(plate,L_r,B_r),c="b",lw=3,ls="--",label="SBR 2 way")

ax.plot(B_r, f_en(plate,L_r,B_r),c="orange", label= "EN 2 way")

f_r = list()

B_r = [6,9,12,15,18,21,24,27,30]
# B_r = [15,30]
L_r = np.full_like(B_r,6)


print(f"length\twidth\tfreq\tmodal")
for l,b in zip(L_r,B_r):
    
    fe,modal_mass = opensees(plate,l,b,mesh_size=0.25,plate_width=3,shell=True,torsion_reduction_factor = 0.1)
    f_r.append(fe)
    
    print(f"{l:5}\t{b:5}\t{fe:5.2f}Hz\t{modal_mass*100:4.1f}%")
    # break
ax.plot(B_r, f_r,"-o",ms=5,c="cyan", label= "opensees plate width 3m, 10% torsion")

f_r = list()

print(f"length\twidth\tfreq\tmodal")
for l,b in zip(L_r,B_r):
    
    fe,modal_mass = opensees(plate,l,b,mesh_size=0.25,plate_width=3,shell=True)
    f_r.append(fe)
    
    print(f"{l:5}\t{b:5}\t{fe:5.2f}Hz\t{modal_mass*100:4.1f}%")
    # break


ax.plot(B_r, f_r,"-o",ms=5,c="olive", label= "opensees plate width 3m 100% torsion")


ax.set_title(f"L= {L}m, EIy/EIx {plate.EIy/plate.EIx:0.3f}")
ax.set_xlabel("Plate width \"B\"")
ax.legend()
