import numpy as np
import graphics as gp
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy
import cProfile

class RingSim(object):    
    def __init__(self, RingNum, E0, H0, Tau0=1e-10, alpha=1.5, T=293, RingSize=64, PCE=0.81, PCH=0.81, Frequency=37, Edist=1e-23, Hdist=1, ER=20, ER_dist=0.5):
        # Define how many regions each ring is split up into
        self.RingSize = RingSize
        self.RingNum = RingNum
        self.RowSize = int(np.sqrt(RingNum))
        # Calculate duration of each field step
        self.TickTime = 1 / (Frequency * RingSize)
        self.time = 0
        #Set model parameters
        self.Tau0 = Tau0
        self.PCE = PCE
        self.PCH = PCH
        self.alpha = alpha
        self.kT = T * 1.38e-23
        self.ER = np.random.normal(ER, ER_dist, (RingSize, RingNum))
        # Set junction location indices
        self.j1 = 0
        self.j2 = int(RingSize/4)
        self.j3 = int(RingSize/2)
        self.j4 = int(3*RingSize/4)
        # Initialise Arrays
        Array = np.zeros((RingSize, RingNum))
        Array[self.j2, :] = -1
        Array[self.j4, :] = 1
        self.Array = Array
        # Initialise Junction Parameters
        self.junclen = int(2 * self.RowSize * (self.RowSize-1))
        self.K = int(self.junclen/2)
        self.Jdw = np.zeros(self.junclen)
        self.Jdw[:int(self.junclen/2)] = 2
        self.Je = np.random.normal(E0, Edist, self.junclen)
        self.Jh = np.random.normal(H0, Hdist, self.junclen)
        # Create map of junction locations and their corresponding ring locations
        juncmap = np.zeros((self.junclen, 2, 2))
        for i in range(self.K):
            juncmap[i, 0] = self.j2, i
            juncmap[i, 1] = self.j4, i+self.RowSize
            juncmap[i+self.K, 0] = self.j1, i%(self.RowSize-1)+i//(self.RowSize-1)*self.RowSize
            juncmap[i+self.K, 1] = self.j3, 1+i%(self.RowSize-1)+i//(self.RowSize-1)*self.RowSize
        self.juncmap=juncmap.astype('int')
        # Set initial magnetisation directions
        Magnetisation = np.copy(Array)
        Magnetisation[:self.j2, :] = -1
        Magnetisation[self.j2:self.j4, :] = 1
        Magnetisation[self.j4:, :] = -1
        Magnetisation[self.j2, :] = 0
        Magnetisation[self.j4, :] = 0
        self.Magnetisation = Magnetisation
        # Calculate net magnetisation components
        self.Mxmods = np.zeros((self.RingSize))
        self.Mymods = np.zeros((self.RingSize))
        for i in range(self.RingSize):
            self.Mxmods[i] = -np.sin(2*np.pi*i/self.RingSize)
            self.Mymods[i] = -np.cos(2*np.pi*i/self.RingSize)
        self.Mxmods = np.tile(self.Mxmods[:, None], self.RingNum)
        self.Mymods = np.tile(self.Mymods[:, None], self.RingNum)
        self.Mx = self.Magnetisation*self.Mxmods
        self.My = self.Magnetisation*self.Mymods
        self.MaxM = np.sum(self.My.flatten())
        self.MaxM_ring = self.MaxM/self.RingNum
        self.NetMx = np.sum(self.Mx.flatten())/self.MaxM
        self.NetMy = np.sum(self.My.flatten())/self.MaxM
        self.DWP = 1
        # Dictionary for UTF printout
        self.UTF_dict = ['\u2191', '\u2193', '\u2192', '\u2190', '\u2B00', '\u2B01', '\u2B02', '\u2B03', '\u2940', '\u2B08', '\u2B09', '\u2B0A', '\u2B0B', '\u2941']

    def calc_mag(self):
       # Copy array state
       array_state = np.copy(self.Array)
       # Create blank matrices for Mx and My
       direction = np.ones((self.RingSize, self.RingNum))
       # Find locations of DWs and order by ring number
       h2hs = np.argwhere(array_state==1)
       t2ts = np.argwhere(array_state==-1)
       h2hs = h2hs[h2hs[:,1].argsort()]
       t2ts = t2ts[t2ts[:,1].argsort()]
       uniques = np.unique(np.concatenate((h2hs[:,0,None], t2ts[:,0,None]), axis=1), axis=0)
       for pair in uniques:
           h2h = pair[0]
           t2t = pair[1]
           h1 = np.argwhere(h2hs[:, 0]==h2h).flatten()
           t1 = np.argwhere(t2ts[:, 0]==t2t).flatten()
           locs = np.intersect1d(h1, t1)
           rings = h2hs[locs, 1]
           if h2h > t2t:
               if t2t > 0:
                   direction[0:t2t, rings]=-1
               direction[h2h:, rings]=-1
           if t2t > h2h:
               direction[h2h:t2t, rings]=-1
       # Set vortex locations by the larger domain in the previous step
       vortices = np.argwhere(np.sum(np.abs(array_state), axis=0)==0)
       if len(vortices) > 0:
           direction[:, vortices] = np.sign(np.sum(self.PastMag[:, vortices], axis=0))
       # Set magnetisation direction of DW locations to zero
       direction[h2hs[:, 0], h2hs[:, 1]]=0 
       direction[t2ts[:, 0], t2ts[:, 1]]=0 
       # Set new magnetisation
       self.Magnetisation = np.copy(direction)
       self.Mx = self.Magnetisation*self.Mxmods
       self.My = self.Magnetisation*self.Mymods
       self.NetMx = np.sum(self.Mx.flatten())/self.MaxM
       self.NetMy = np.sum(self.My.flatten())/self.MaxM
       self.DWP = np.sum(np.abs(array_state))/(2*self.RingNum)
       self.Array = np.copy(array_state)
     
    def depin(self, Hin, Hloc):
        # Calculate theta from location
        Htheta = Hloc*2*np.pi/self.RingSize
        # Copy previous domain state
        array_state = np.copy(self.Array)
        # Create blank matrix for lag angles
        lag_angles = np.zeros((self.RingSize, self.RingNum))
        # Calculate angle normal to the applied field
        normang = (Htheta-np.pi/2)%(2*np.pi)
        # Loop over indices
        for index in range(self.RingSize):
            # Calculate the angle from x for any DWs present
            theta_loc = index * 2*np.pi/self.RingSize * np.abs(array_state[index, :])
            # Calculate the location for the energy minimum for each dw
            theta_em = (normang * np.abs(array_state[index, :]) + np.pi/2 * array_state[index, :])%(2*np.pi)
            # Set lag angle as difference between the two
            lag_angles[index, :] = np.abs(theta_loc - theta_em)
        # Ensure smallest path for lag angle > pi
        lag_angles[lag_angles>np.pi] = 2*np.pi - lag_angles[lag_angles>np.pi]

        sing_dws = np.argwhere(self.Jdw==1)
        doub_dws = np.argwhere(self.Jdw==2)

        # Create map of effective E0/H0s at each junction
        Eeff = np.zeros(self.junclen)
        Heff = np.zeros(self.junclen)
        Eeff[sing_dws] = self.Je[sing_dws]
        Eeff[doub_dws] = self.Je[doub_dws] * self.PCE
        Heff[sing_dws] = self.Jh[sing_dws]
        Heff[doub_dws] = self.Jh[doub_dws] * self.PCH

        Junclags = np.zeros(self.junclen)
        inds = np.argwhere(Eeff != 0).flatten()
        for index in inds:
            location = self.juncmap[index]
            l1 = location[0, 0]
            r1 = location[0, 1]
            l2 = location[1, 0]
            r2 = location[1, 1]
            if lag_angles[l1, r1] != 0:
                Junclags[index] = lag_angles[l1, r1]
            else:
                Junclags[index] = lag_angles[l2, r2]

        FieldComponents = np.sin(Junclags) * Hin
        with np.errstate(divide='ignore', invalid='ignore'):
            DeltaE = Eeff * (1 - FieldComponents/Heff)**self.alpha
        Tau = self.Tau0 * np.exp(DeltaE/self.kT)
        DepinChance = 1-np.exp(-self.TickTime/Tau)
        DepinChance[np.isnan(DepinChance)] = 0
        DiceRoll = np.random.ranf(len(DepinChance))
        depinned = np.argwhere(DepinChance>DiceRoll).flatten()
        # Mark depinned domain walls and change Jdw to reflect depinned locations
        self.Jdw[depinned] = 0
        depinned_indices = np.asarray(np.reshape(self.juncmap[depinned], (2*len(depinned), 2)), dtype='int')
        depinned_dws = np.zeros((self.RingSize, self.RingNum))
        depinned_dws[depinned_indices[:, 0], depinned_indices[:, 1]] = 1
        # Catch single dw over junction cases
        depinned_dws *= array_state
        # catch edge roughness cases
        allFieldComponents = np.abs(np.sin(lag_angles))*Hin
        allFieldComponents[allFieldComponents<self.ER] = 0
        allFieldComponents[allFieldComponents>self.ER] = 1
        depinned_dws *= allFieldComponents
        # Add DWs from non-junction locations
        nonjunc = np.copy(array_state)
        nonjunc[[self.j1, self.j2, self.j3, self.j4], :] = 0
        depinned_dws += nonjunc
        # Add DWs from the perimeter
        depinned_dws[0, self.RowSize*np.arange(1, self.RowSize+1)-1] += array_state[0, self.RowSize*np.arange(1, self.RowSize+1)-1]
        depinned_dws[int(self.RingSize/4), int(self.RingNum-self.RowSize):] += array_state[int(self.RingSize/4), int(self.RingNum-self.RowSize):]
        depinned_dws[int(self.RingSize/2), self.RowSize*np.arange(0, self.RowSize, 1)] += array_state[int(self.RingSize/2), self.RowSize*np.arange(0, self.RowSize, 1)]
        depinned_dws[int(3*self.RingSize/4), 0:self.RowSize] += array_state[int(3*self.RingSize/4), 0:self.RowSize]
        return depinned_dws
       
    def propagate(self, depinned_dws, Hloc):
        # Create log of dws propagating beyond junctions
        propagations = np.zeros((4, self.RingNum))
        for i in range(4):
            propagations[i] = depinned_dws[int(i*self.RingSize/4)]
        # Copy previous array state
        array_state = np.copy(self.Array)
        # Set energy minimum for each DW type
        h2hmin = Hloc
        t2tmin = int((Hloc+self.RingSize/2)%self.RingSize)
        # Loop over locations and propagate dws
        for loc in range(self.RingSize):
            # Find dws at location
            h2hs = np.argwhere(depinned_dws[loc]==1)
            t2ts = np.argwhere(depinned_dws[loc]==-1)
            # Set copy of array state for these locations to zero (propagated dws)
            array_state[loc, h2hs] = 0
            array_state[loc, t2ts] = 0
            # If location is at junction:
            if loc % self.RingSize/4 == 0:
                # If minima is <= 1/4 rotation away, move to minimum
                if np.abs(h2hmin - loc) <= self.RingSize/4 or np.abs(h2hmin - loc) >= 3*self.RingSize/4:
                    array_state[h2hmin, h2hs] += 1
                if np.abs(t2tmin - loc) <= self.RingSize/4 or np.abs(t2tmin - loc) >= 3*self.RingSize/4:
                    array_state[t2tmin, t2ts] += -1
                # If minima is further away, move to first junction it passes
                if self.RingSize/4 < np.abs(h2hmin - loc) < 3*self.RingSize/4:
                    # Ensure closest junction then move
                    dist = np.abs(h2hmin-loc)
                    sign = np.sign(h2hmin-loc)
                    if dist > self.RingSize/2 and sign == -1 or dist < self.RingSize/2 and sign == 1:
                        array_state[int(loc+self.RingSize/4)%self.RingSize, h2hs] += 1
                    else:
                        array_state[int(loc-self.RingSize/4)%self.RingSize, h2hs] += 1
                if self.RingSize/4 < np.abs(t2tmin - loc) < 3*self.RingSize/4:
                    # Ensure closest junction then move
                    dist = np.abs(t2tmin-loc)
                    sign = np.sign(t2tmin-loc)
                    if dist > self.RingSize/2 and sign == -1 or dist < self.RingSize/2 and sign == 1:
                        array_state[int(loc+self.RingSize/4)%self.RingSize, t2ts] += -1
                    else:
                        array_state[int(loc-self.RingSize/4)%self.RingSize, t2ts] += -1
            # For DWs away from a junction, follow the field
            else:
                array_state[h2hmin, h2hs] += 1
                array_state[t2tmin, t2ts] += -1 
        # Update array state from propagated state        
        self.Array = np.copy(array_state)
        return propagations
      
    def find_frustrated(self, propagations):
        # Find propagation locations
        proplocs = np.argwhere(propagations!=0)  
        # Restore to locations from junction number
        proplocs[:, 0] *= int(self.RingSize/4)   
        # Create output for frustrated junctions
        frustrated = []
        # loop over propagated DWs and check for frustration
        for l1 in proplocs:
            try:
                # find ring pair of junction
                juncloc = np.argwhere((self.juncmap==l1).all(axis=2))
                j1, j2 = self.juncmap[juncloc[0, 0]]
                # if magnetisation is frustrated, add to the list
                if self.Magnetisation[int(j1[0]), int(j1[1])] == self.Magnetisation[int(j2[0]), int(j2[1])]:
                    frustrated.append([j1, j2])
            # Except clause for propagations from perimeter rings
            except:
                continue
        # Remove duplicates
        frustrated = np.unique(np.asarray(frustrated), axis=0)
        return frustrated
    
    def nucleate(self, frustrated, Hloc):
        # Copy previous state
        array_state = np.copy(self.Array)
        # Loop over frustrated locations
        h2hmin = int(Hloc)
        t2tmin = int(Hloc+self.RingSize/2)%self.RingSize
        for pair in frustrated:
            # Find the location of the rings
            loc1, ring1 = pair[0].astype('int')
            loc2, ring2 = pair[1].astype('int')
            # If the ring is empty, nucleate
            if np.sum(np.abs(array_state[:, ring1]))==0:
                # If minima is <= 1/4 rotation away, move to minimum
                if np.abs(h2hmin - loc1) < self.RingSize/4 or np.abs(h2hmin - loc1) > 3*self.RingSize/4:
                    array_state[h2hmin, ring1] += 1
                # If minima is further away, move to first junction it passes
                else: 
                    # Ensure closest junction then move
                    dist = np.abs(h2hmin-loc1)
                    sign = np.sign(h2hmin-loc1)
                    if dist > self.RingSize/2 and sign == -1 or dist < self.RingSize/2 and sign == 1:
                        array_state[int(loc1+self.RingSize/4)%self.RingSize, ring1] += 1
                    else:
                        array_state[int(loc1-self.RingSize/4)%self.RingSize, ring1] += 1
                if np.abs(t2tmin - loc1) < self.RingSize/4 or np.abs(t2tmin - loc1) > 3*self.RingSize/4:
                    array_state[t2tmin, ring1] += -1
                else:
                    # Ensure closest junction then move
                    dist = np.abs(t2tmin-loc1)
                    sign = np.sign(t2tmin-loc1)
                    if dist > self.RingSize/2 and sign == -1 or dist < self.RingSize/2 and sign == 1:
                        array_state[int(loc1+self.RingSize/4)%self.RingSize, ring1] += -1
                    else:
                        array_state[int(loc1-self.RingSize/4)%self.RingSize, ring1] += -1
            if np.sum(np.abs(array_state[:, ring2]))==0:
                # If minima is <= 1/4 rotation away, move to minimum
                if np.abs(h2hmin - loc2) < self.RingSize/4 or np.abs(h2hmin - loc2) > 3*self.RingSize/4:
                    array_state[h2hmin, ring2] += 1
                else:
                    # Ensure closest junction then move
                    dist = np.abs(h2hmin-loc2)
                    sign = np.sign(h2hmin-loc2)
                    if dist > self.RingSize/2 and sign == -1 or dist < self.RingSize/2 and sign == 1:
                        array_state[int(loc2+self.RingSize/4)%self.RingSize, ring2] += 1
                    else:
                        array_state[int(loc2-self.RingSize/4)%self.RingSize, ring2] += 1
                if np.abs(t2tmin - loc2) < self.RingSize/4 or np.abs(t2tmin - loc2) > 3*self.RingSize/4:
                    array_state[t2tmin, ring2] += -1
                # If minima is further away, move to first junction it passes
                else:
                    # Ensure closest junction then move
                    dist = np.abs(t2tmin-loc2)
                    sign = np.sign(t2tmin-loc2)
                    if dist > self.RingSize/2 and sign == -1 or dist < self.RingSize/2 and sign == 1:
                        array_state[int(loc2+self.RingSize/4)%self.RingSize, ring2] += -1
                    else:
                        array_state[int(loc2-self.RingSize/4)%self.RingSize, ring2] += -1
        # Set array state from new state
        self.Array = np.copy(array_state)
        
    def reset_juncs(self):
        # New blank vector for Jdw
        junc_count = np.zeros((self.junclen))
        # Copy array state
        array_state = np.copy(self.Array)
        # Count DWs at junctions
        for i in range(self.junclen):
            junc = self.juncmap[i]
            l1 = junc[0, 0]
            r1 = junc[0, 1]
            l2 = junc[1, 0]
            r2 = junc[1, 1]
            junc_count[i] = np.abs(array_state[l1, r1]) + np.abs(array_state[l2, r2])
        self.Jdw = np.copy(junc_count)
        
    def visualise(self, t):
        path = os.getcwd()
        t += int(3*self.RingSize/4)
        #Sets the blank window for the graphical output to be drawn onto
        win = gp.GraphWin("My Window", 1000, 1000)
        win.setBackground('white')    
        K = self.RingSize/2
        for j in range(self.RingNum):
            #Erases data from previous entry
            # Defines the X position of the centre each ring
            X=(j%self.RowSize)*(1000/self.RowSize)+(500/self.RowSize)
            # Defines the Y position of the centre each ring
            Y=(j//self.RowSize)*(1000/self.RowSize)+(500/self.RowSize)       
            #Draw the geometry for each ring
            OD = gp.Circle(gp.Point(X,Y),500/self.RowSize)
            OD.draw(win)
            ID = gp.Circle(gp.Point(X,Y),500/self.RowSize-40)
            ID.draw(win)
            # Draws the minimum energy location of '+' and '-' DWs
            minplus=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(((t)%self.RingSize)*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(((t)%self.RingSize)*np.pi/K)), path+'/Sprites/MinPlus.gif')
            minplus.draw(win)
            minmin=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(((t-K)%self.RingSize)*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(((t-K)%self.RingSize)*np.pi/K)), path+'/Sprites/MinMinus.gif')
            minmin.draw(win)
            #Provide real time information as to how many ticks have passed and how many
            #DWs are present in the array at each instance.
            # Plots each ring from within the array
            Ring=self.Array[:, j]
            Dir=self.Magnetisation[:, j]
            for k in range(self.RingSize):
                if Dir[k] == 0 and Ring[k] == 1:
                    plus = gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/Plus.gif')
                    plus.draw(win)
                elif Dir[k] == 0 and Ring[k]==-1:
                    minus = gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/Minus.gif')
                    minus.draw(win)
                if Dir[k] == 1:
                    if k==0:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/DownR.gif')
                        mg.draw(win)
                    elif k==self.RingSize/8:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/DLR.gif')
                        mg.draw(win)
                    elif k==self.RingSize/4:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/LeftR.gif')
                        mg.draw(win)
                    elif k==3*self.RingSize/8:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/ULR.gif')
                        mg.draw(win)
                    elif k==self.RingSize/2:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/UpR.gif')
                        mg.draw(win)
                    elif k==5*self.RingSize/8:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/URR.gif')
                        mg.draw(win)
                    elif k==3*self.RingSize/4:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/RightR.gif')
                        mg.draw(win)
                    elif k==7*self.RingSize/8:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/DRR.gif')
                        mg.draw(win)
                elif Dir[k] == -1:
                    if k==0:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/UpB.gif')
                        mg.draw(win)
                    elif k==self.RingSize/8:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/URB.gif')
                        mg.draw(win)
                    elif k==self.RingSize/4:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/RightB.gif')
                        mg.draw(win)
                    elif k==3*self.RingSize/8:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/DRB.gif')
                        mg.draw(win)
                    elif k==self.RingSize/2:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/DownB.gif')
                        mg.draw(win)
                    elif k==5*self.RingSize/8:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/DLB.gif')
                        mg.draw(win)
                    elif k==3*self.RingSize/4:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/LeftB.gif')
                        mg.draw(win)
                    elif k==7*self.RingSize/8:
                        mg=gp.Image(gp.Point(X+((500/self.RowSize)-20)*np.cos(k*np.pi/K),Y+((500/self.RowSize)-20)*+np.sin(k*np.pi/K)), path+'/Sprites/ULB.gif')
                        mg.draw(win)
        #Saves graphical output to a temporary output file
        win.postscript(file="output.eps")
        img = Image.open('output.eps')
        rgb_img = img.convert('RGB')
        rgb_img.save('output%i.jpg' %(t-12))
        win.close()
        
    def bigimgoutput_magnetisation(self, t, title='Output'):
        path = os.getcwd()+'\\Sprites\\'
        width = 32 * self.RowSize
        win = gp.GraphWin("My Window", width, width)
        win.setBackground('white')                     
        for j in range(self.RingNum):
            # Defines the X position of the centre each ring
            X=(j%self.RowSize)*(width/self.RowSize)+(width/(2*self.RowSize))
            # Defines the Y position of the centre each ring
            Y=(j//self.RowSize)*(width/self.RowSize)+(width/(2*self.RowSize))       
            # Plots each ring from within the array
            Ring = self.Array[:, j]
            Dir=self.Magnetisation[:, j]
            if np.sum(np.abs(Ring)) < 2:
                if np.sum(Dir) > 0:
                    RV = gp.Image(gp.Point(X, Y), path+"V_CW.png")
                    RV.draw(win)
                elif np.sum(Dir) < 0:
                    BV = gp.Image(gp.Point(X, Y), path+"V_ACW.png")
                    BV.draw(win)
            elif Ring[0] == 1 and Ring[int(self.RingSize/2)] == -1:
                ROS = gp.Image(gp.Point(X, Y), path+"Mag_8_0.png")
                ROS.draw(win)
            elif Ring[0] == -1 and Ring[int(self.RingSize/2)] == 1:
                RON = gp.Image(gp.Point(X, Y), path+"Mag_0_8.png")
                RON.draw(win)
            elif Ring[int(self.RingSize/4)] == 1 and Ring[int(3*self.RingSize/4)] == -1:
                ROW = gp.Image(gp.Point(X, Y), path+"Mag_12_4.png")
                ROW.draw(win)
            elif Ring[int(self.RingSize/4)] == -1 and Ring[int(3*self.RingSize/4)] == 1:
                ROE = gp.Image(gp.Point(X, Y), path+"Mag_4_12.png")
                ROE.draw(win)
            elif Ring[0] == 1 and Ring[int(self.RingSize/4)] == -1:
                BTSE = gp.Image(gp.Point(X, Y), path+"Mag_4_0.png")
                BTSE.draw(win)
            elif Ring[0] == -1 and Ring[int(self.RingSize/4)] == 1:
                RTSE = gp.Image(gp.Point(X, Y), path+"Mag_0_4.png")
                RTSE.draw(win)
            elif Ring[int(self.RingSize/4)] == 1 and Ring[int(self.RingSize/2)] == -1:
                BTSW = gp.Image(gp.Point(X, Y), path+"Mag_8_4.png")
                BTSW.draw(win)
            elif Ring[int(self.RingSize/4)] == -1 and Ring[int(self.RingSize/2)] == 1:
                RTSW = gp.Image(gp.Point(X, Y), path+"Mag_4_8.png")
                RTSW.draw(win)
            elif Ring[int(self.RingSize/2)] == 1 and Ring[int(3*self.RingSize/4)] == -1:
                BTNW = gp.Image(gp.Point(X, Y), path+"Mag_12_8.png")
                BTNW.draw(win)
            elif Ring[int(self.RingSize/2)] == -1 and Ring[int(3*self.RingSize/4)] == 1:
                RTNW = gp.Image(gp.Point(X, Y), path+"Mag_8_12.png")
                RTNW.draw(win)
            elif Ring[int(3*self.RingSize/4)] == 1 and Ring[0] == -1:
                BTNE = gp.Image(gp.Point(X, Y), path+"Mag_0_12.png")
                BTNE.draw(win)
            elif Ring[int(3*self.RingSize/4)] == -1 and Ring[0] == 1:
                RTNE = gp.Image(gp.Point(X, Y), path+"Mag_12_0.png")
                RTNE.draw(win)
        win.postscript(file="output.eps")
        img = Image.open('output.eps')
        rgb_img = img.convert('RGB')
        rgb_img.save(title+', t=%i.jpg' %t)
        win.close()
        
    def bigimgoutput_chiral(self, t, title='Output'):
        path = os.getcwd()+'\\Sprites\\'
        width = 32 * self.RowSize
        win = gp.GraphWin("My Window", width, width)
        win.setBackground('white')                     
        for j in range(self.RingNum):
            # Defines the X position of the centre each ring
            X=(j%self.RowSize)*(width/self.RowSize)+(width/(2*self.RowSize))
            # Defines the Y position of the centre each ring
            Y=(j//self.RowSize)*(width/self.RowSize)+(width/(2*self.RowSize))       
            # Plots each ring from within the array
            Ring = self.Array[:, j]
            Dir=self.Magnetisation[:, j]
            if np.sum(np.abs(Ring)) < 2:
                if np.sum(Dir) > 0:
                    RV = gp.Image(gp.Point(X, Y), path+"Vortex_CW.png")
                    RV.draw(win)
                elif np.sum(Dir) < 0:
                    BV = gp.Image(gp.Point(X, Y), path+"Vortex_ACW.png")
                    BV.draw(win)
            elif Ring[0] == 1 and Ring[int(self.RingSize/2)] == -1:
                ROS = gp.Image(gp.Point(X, Y), path+"Onion_right.png")
                ROS.draw(win)
            elif Ring[0] == -1 and Ring[int(self.RingSize/2)] == 1:
                RON = gp.Image(gp.Point(X, Y), path+"Onion_left.png")
                RON.draw(win)
            elif Ring[int(self.RingSize/4)] == 1 and Ring[int(3*self.RingSize/4)] == -1:
                ROW = gp.Image(gp.Point(X, Y), path+"Onion_down.png")
                ROW.draw(win)
            elif Ring[int(self.RingSize/4)] == -1 and Ring[int(3*self.RingSize/4)] == 1:
                ROE = gp.Image(gp.Point(X, Y), path+"Onion_up.png")
                ROE.draw(win)
            elif Ring[0] == 1 and Ring[int(self.RingSize/4)] == -1:
                BTSE = gp.Image(gp.Point(X, Y), path+"Tq_4_0.png")
                BTSE.draw(win)
            elif Ring[0] == -1 and Ring[int(self.RingSize/4)] == 1:
                RTSE = gp.Image(gp.Point(X, Y), path+"Tq_0_4.png")
                RTSE.draw(win)
            elif Ring[int(self.RingSize/4)] == 1 and Ring[int(self.RingSize/2)] == -1:
                BTSW = gp.Image(gp.Point(X, Y), path+"Tq_8_4.png")
                BTSW.draw(win)
            elif Ring[int(self.RingSize/4)] == -1 and Ring[int(self.RingSize/2)] == 1:
                RTSW = gp.Image(gp.Point(X, Y), path+"Tq_4_8.png")
                RTSW.draw(win)
            elif Ring[int(self.RingSize/2)] == 1 and Ring[int(3*self.RingSize/4)] == -1:
                BTNW = gp.Image(gp.Point(X, Y), path+"Tq_12_8.png")
                BTNW.draw(win)
            elif Ring[int(self.RingSize/2)] == -1 and Ring[int(3*self.RingSize/4)] == 1:
                RTNW = gp.Image(gp.Point(X, Y), path+"Tq_8_12.png")
                RTNW.draw(win)
            elif Ring[int(3*self.RingSize/4)] == 1 and Ring[0] == -1:
                BTNE = gp.Image(gp.Point(X, Y), path+"Tq_0_12.png")
                BTNE.draw(win)
            elif Ring[int(3*self.RingSize/4)] == -1 and Ring[0] == 1:
                RTNE = gp.Image(gp.Point(X, Y), path+"Tq_12_0.png")
                RTNE.draw(win)
        win.postscript(file="output.eps")
        img = Image.open('output.eps')
        rgb_img = img.convert('RGB')
        rgb_img.save(title+', t=%i.jpg' %t)
        win.close()
        
    def text_img_output(self, t, title='Output', save=True):
        # String to delimit each printout
        header = 'Model State, t = '+str(t)
        print(header)
        text = []
        # Find direction of Magnetisation in X and Y
        Mx = np.sum(self.Mx, axis=0)/self.MaxM_ring
        Mx[np.abs(Mx)<1e-3]=0
        Mx = np.round(Mx, decimals=1)
        My = np.sum(self.My, axis=0)/self.MaxM_ring
        My[np.abs(My)<1e-3]=0
        My = np.round(My, decimals=1)
        # Find onions
        up_onions = np.argwhere(My==1)
        down_onions = np.argwhere(My==-1)
        right_onions = np.argwhere(Mx==1)
        left_onions = np.argwhere(Mx==-1)
        # Find 3/4s
        ne_tq = np.intersect1d(np.argwhere(Mx==0.5), np.argwhere(My==0.5))
        nw_tq = np.intersect1d(np.argwhere(Mx==-0.5), np.argwhere(My==0.5))
        se_tq = np.intersect1d(np.argwhere(Mx==0.5), np.argwhere(My==-0.5))
        sw_tq = np.intersect1d(np.argwhere(Mx==-0.5), np.argwhere(My==-0.5))
        # Find Vortices
        vortices = np.intersect1d(np.argwhere(Mx==0), np.argwhere(My==0))
        # Assign each to a string identifier
        outputs = np.zeros(self.RingNum)
        outputs[up_onions.flatten()] = 0
        outputs[down_onions.flatten()] = 1
        outputs[right_onions.flatten()] = 2
        outputs[left_onions.flatten()] = 3
        outputs[ne_tq.flatten()] = 4
        outputs[nw_tq.flatten()] = 5
        outputs[se_tq.flatten()] = 6
        outputs[sw_tq.flatten()] = 7
        outputs[vortices.flatten()] = 8
        # Find signs for 3/4s and vortices
        signs = np.sign(np.sum(self.Magnetisation, axis=0))
        acw = np.argwhere(signs==-1)
        # Account for opposite direction
        outputs[acw.flatten()] += 5
        # Reshape into grid
        outputs = outputs.reshape(self.RowSize, self.RowSize)
        # Loop over and print
        for row in range(self.RowSize):
            string = []
            for ring in range(self.RowSize):
                string.append(self.UTF_dict[int(outputs[row, ring])])
            print(string)
            text.append(string)
        print(' ')
        if save==True:
            np.savetxt(title+' '+str(t)+'.txt', np.asarray(text), delimiter=' ', fmt='%s')

        
    def step_sim(self, Hin, Hloc):
        Hloc += int(3*self.RingSize/4)
        # Depin check
        depinned_dws = self.depin(Hin, Hloc%self.RingSize)
        # Initial propagation
        propagations = self.propagate(depinned_dws, Hloc%self.RingSize)
        # Recalculate magnetisation state
        self.calc_mag()
        # Find frustrated junctions
        frustrated = self.find_frustrated(propagations)
        # Nucleate
        self.nucleate(frustrated, Hloc%self.RingSize)
        # Recalculate Magnetisation
        self.calc_mag()
        self.reset_juncs()
        self.PastState=np.copy(self.Array)
        self.PastMag=np.copy(self.Magnetisation)
        
        

# Initialisation parameters


Nrings = 25**2
E0 = 2.625e-19
sim_duration = 10*64+1
for Happ in [18, 24, 26, 28, 32]:
    mag = np.zeros(sim_duration)
    H0 = 55
    # Initialise simulation
    sim = RingSim(Nrings, E0, H0)
    # For every timestep in the duration of the simulation
    for t in range(sim_duration):
        # Step the simulation at Happ, for a location given by time t
        sim.step_sim(Happ, t%sim.RingSize)
        mag[t] = sim.NetMy
            # if t % 16 == 0:
            #     #sim.text_img_output(t, save=True)
            #     sim.bigimgoutput_chiral(t)
    sim.bigimgoutput_magnetisation(t, title=str(Happ)+' ')
    plt.figure()
    plt.title(Happ)
    plt.plot(np.linspace(0, 10, 641), mag)
    plt.show()

            



