
# coding: utf-8

# # Statistics of Polymer Models
# ## Nikolas Schnellbächer (created 2018-07-04)
# 
# Here explore some statistical properties of two simples polymer models. 
# We use the freely jointed and the freely rotating chain and simulate both using random walk models.
# In all cases $N$ denotes the number of chain segments and $a$ is the segment length. For the freely rotating chain model $\theta$ is the rotation angle. In this exercise we work in two dimensions exclusively.

# In[1]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats


# In[2]:


# define measurement functions here

def get_msd(X, Y, sampleLengths):
    '''
    Calculates the msd of two-dimensional trajectories.
    '''
    msd = np.zeros((len(sampleLengths),))
    
    for i in range(len(sampleLengths)):
    
        msd[i] = np.mean(np.square(X[i, :]) + np.square(Y[i, :]))
    
    return msd

def FloryC(theta):
    '''
    Flory's characterictic ratio $C_{\infty}$
    '''
    return (1.0 + np.cos(theta)) / (1.0 - np.cos(theta))


# In[3]:


def FJC(sampleLengths, m, a = 1.0e-3, x0 = 0.0, y0 = 0.0):
    '''
    Random walk model for the freely jointed chain (FJC) in two dimensions (d = 2).
    '''
    nSamples = len(sampleLengths)
    iterations = (sampleLengths).astype(int)
    iterations[1:] = iterations[1:] - iterations[0:-1]
    totalIterations = np.cumsum(iterations)[-1]
    print("totalIterations = ",totalIterations)
    
    LOW, HIGH = 0.0, 2.0 * np.pi
    
    outx = np.zeros((nSamples, m))
    outy = np.zeros((nSamples, m))
    
    for k in range(m):
        
        x, y = x0, y0 # set initial position
        
        for j in range(nSamples):
            
            for i in range(iterations[j]):
                
                phi = np.random.uniform(low = LOW, high = HIGH)
                
                x += a * np.cos(phi)
                y += a * np.sin(phi)
                
            outx[j, k] = x
            outy[j, k] = y
    
    return outx, outy


# In[4]:


def FJC_vec(sampleLengths, m, a = 1.0e-3, x0 = 0.0, y0 = 0.0):
    '''
    Random walk model for the freely jointed chain (FJC) in two dimensions (d = 2).
    Vectorized version of the FJC algorithm.
    '''
    nSamples = len(sampleLengths)
    iterations = (sampleLengths).astype(int)
    iterations[1:] = iterations[1:] - iterations[0:-1]
    totalIterations = np.cumsum(iterations)[-1]
    print("totalIterations = ",totalIterations)
    
    LOW, HIGH = 0.0, 2.0 * np.pi
    
    outx = np.zeros((nSamples, m))
    outy = np.zeros((nSamples, m))
    
    x = np.ones((1, m)) * x0
    y = np.ones((1, m)) * y0

    outx[0, :] = x
    outy[0, :] = y

    for j in range(nSamples):
            
        for i in range(iterations[j]):
                
            phis = np.random.uniform(low = LOW, high = HIGH, size = m)
                
            x += a * np.cos(phis)
            y += a * np.sin(phis)
                
            outx[j, :] = x
            outy[j, :] = y
    
    return outx, outy


# To understand the implementation for the FRC algorithm below recall that in two dimensions the standard rotation matrix $R(\theta)$ reads
# \begin{align}
# R(\theta) = \begin{pmatrix}
# \cos\theta & -\sin\theta \\
# \sin\theta & \cos\theta
# \end{pmatrix} \, .
# \end{align}
# The binary coin flip choice at each iteration then simply decides whether the current orientation is either rotated by $R_{+}:=R(+\theta)$ or by $R_{-}:=R(-\theta)$.
# Given a starting orientation vector $\boldsymbol{f}$ we update the orientation by invoking
# \begin{align}
# \boldsymbol{f}_{i+1} = \begin{cases}
# R_+ \cdot \boldsymbol{f}_{i} & \text{for}\quad p_{\text{coin}} > 0.5\\
# R_- \cdot \boldsymbol{f}_{i} & \text{else}
# \end{cases}
# \end{align}
# Then the position update follows immediately as
# \begin{align}
# \boldsymbol{x}_{i+1} = \boldsymbol{x}_i + a \cdot \boldsymbol{f}_{i+1} \, .
# \end{align}
# Here $\boldsymbol{x}$ and $\boldsymbol{f}$ are both two-dimensional position and orientation vectors, respectively.

# In[5]:


def FRC(sampleLengths, m, a = 1.0e-3, theta = 0.2 * np.pi, x0 = 0.0, y0 = 0.0):
    '''
    Random walk model for the freely rotating chain (FRC) in two dimensions (d = 2).
    '''
    nSamples = len(sampleLengths)
    iterations = (sampleLengths).astype(int)
    iterations[1:] = iterations[1:] - iterations[0:-1]
    totalIterations = np.cumsum(iterations)[-1]
    print("totalIterations = ",totalIterations)
    
    LOW, HIGH = 0.0, 2.0 * np.pi
    
    outx = np.zeros((nSamples, m))
    outy = np.zeros((nSamples, m))
    
    # make sure to precalculate R0 and R1 outside of all the loops below!
    R0 = np.matrix([[np.cos(theta), -np.sin(theta)],                    [np.sin(theta), np.cos(theta)]])
    R1 = np.matrix([[np.cos(-theta), -np.sin(-theta)],                    [np.sin(-theta), np.cos(-theta)]])
    
    for k in range(m):
        
        x = np.array([[x0], [y0]]) # set initial position
        phi = np.random.uniform(LOW, HIGH)
        ori = np.array([[np.cos(phi)],[np.sin(phi)]])
        
        for j in range(nSamples):
            
            for i in range(iterations[j]):
                
                flip = np.random.choice([0, 1])
                
                if (flip == 1):
                    ori = np.dot(R1, ori)
                else:
                    ori = np.dot(R0, ori)

                x += a * ori
            
            outx[j, k] = x[0]
            outy[j, k] = x[1]
    
    return outx, outy


# The implementation of the FRC function above is a very literal and explicit implementation. It is a straight forward implementation of the described formulaes and very literally implements the algorithm.
# This makes it clear to understand what is going on, at the expense of computational speed. Below I show you a vectorized version for the FRC model, which is much faster and hence recommended for more extensive statistical analysis of the model.

# In[6]:


def FRC_vecSingle(N, a = 1.0e-3, theta = 0.2 * np.pi, x0 = 0.0, y0 = 0.0):
    '''
    Random walk model for the freely rotating chain (FRC) in two dimensions (d = 2).
    Vectorized version for a single trajectory (polymer).
    '''
    outx = np.zeros((N + 1,))
    outy = np.zeros((N + 1,))
    
    # initialize the random walker at the origin
    outx[0], outy[0] = x0, y0
    
    angles = np.zeros((N))
    # initial orientation uniformly sampled in [0, 2 * Pi)
    angles[0] = np.random.uniform(0.0, 2.0 * np.pi) 
    angles[1:] = theta * np.random.choice([-1.0, 1.0], N - 1)
    
    angles = np.cumsum(angles)
    outx[1:] = np.cumsum(a * np.cos(angles))
    outy[1:] = np.cumsum(a * np.sin(angles))

    return outx, outy

def FRC_wrapper(sampleIndices, m, a = 1.0e-3,                theta = 0.2 * np.pi, x0 = 0.0, y0 = 0.0):
    '''
    Wrapper function which creates m statistically 
    independent realizations of the FRC model.
    '''
    xVals = np.zeros((len(sampleIndices), m))
    yVals = np.zeros((len(sampleIndices), m))
    
    N = sampleIndices[-1]
    
    for i in range(m):
        tmpX, tmpY = FRC_vecSingle(N, a, theta, x0, y0)
        xVals[:, i] = tmpX[sampleIndices]
        yVals[:, i] = tmpY[sampleIndices]
    
    return xVals, yVals


# In[7]:


def plot_trajectories(X, Y, m = 10):
    
    f, ax = plt.subplots(1)
    f.set_size_inches(5.5, 5.5)
    
    ax.set_xlabel('$x$ coordinate', fontsize = 18)
    ax.set_ylabel('$y$ coordinate', fontsize = 18)
    labelfontsize = 15.0
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    
    for i in range(m):
        ax.plot(X[:, i], Y[:, i], lw = 1.0, alpha = 0.80)

    rx = np.max(X)
    ry = np.max(Y)
    r = np.max([rx, ry])
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    
    return None


# ## Assay 1 -  Random walk model for the FJC

# In[8]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(123456789)\n# create m = 25 FJC polymer configurations\nm = 25\nsampleLengths = np.arange(0.0, 501.0, 1)\nX1, Y1 = FJC(sampleLengths, m)')


# In[9]:


m = 25
plot_trajectories(X1, Y1, m)


# In[10]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(923456789)\n# create m = 25 FJC polymer configurations\nm = 25\nsampleLengths = np.arange(0.0, 501.0, 1)\nX1vec, Y1vec = FJC_vec(sampleLengths, m)')


# In[11]:


m = 25
plot_trajectories(X1vec, Y1vec, m)


# ## Random walk model for the FRC

# In[12]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(123456789)\n# create m = 25 FRC polymer configurations\nm = 25\nsampleLengths = np.arange(0.0, 501.0, 1)\nX2, Y2 = FRC(sampleLengths, m, theta = 0.1 * np.pi)')


# In[13]:


get_ipython().run_cell_magic('time', '', '# We repeat the same, using the vectorized version of the FRC algorithm.\nnp.random.seed(223456789)\n# create m = 25 FJC polymer configurations\nm = 25\nsampleIndices = np.arange(0.0, 501.0, 1).astype(int)\nX2_vec, Y2_vec = FRC_wrapper(sampleIndices, m, a = 1.0e-3, theta = 0.1 * np.pi)')


# In[14]:


m = 25
plot_trajectories(X2, Y2, m)


# In[15]:


m = 25
plot_trajectories(X2_vec, Y2_vec, m)


# In[16]:


# separate plotting for m = 25 polymer chains
print(X2_vec.shape)
print(Y2_vec.shape)
f, ax = plt.subplots(5, 5, figsize = (18.0, 18.0))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.plot(X2_vec[:, i], Y2_vec[:, i])


# In[17]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(923456789)\n# create m = 50 FRC polymer configurations\nm = 50\nsampleIndices = np.arange(0.0, 501.0, 1).astype(int)\nX3_vec, Y3_vec = FRC_wrapper(sampleIndices, m, theta = 0.01 * np.pi)')


# In[18]:


m = 50
plot_trajectories(X3_vec, Y3_vec, m)


# In[19]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(123456789)\n# create m = 100 FRC polymer configurations\nm = 100\nsampleIndices = np.arange(0.0, 501.0, 1).astype(int)\nX3_vec, Y3_vec = FRC_wrapper(sampleIndices, m, theta = 0.001\n             * np.pi)')


# In[23]:


m = 100
plot_trajectories(X3_vec, Y3_vec, m)


# Changing the rotation angle $\theta$ we can change the polymers persitence lenght $l_P$.
# The well established relation for the freely rotating chain model reads
# \begin{align}
# l_P = -\dfrac{a}{\ln\left(\cos\theta\right)} \, .
# \end{align}

# In[20]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(123456789)\n# create m = 25 FRC polymer configurations\nm = 10\nsampleIndices = np.arange(0.0, 501.0, 1).astype(int)\nX3_vec, Y3_vec = FRC_wrapper(sampleIndices, m, theta = 0.5 * np.pi)')


# In[21]:


m = 10
plot_trajectories(X3_vec, Y3_vec, m)


# In[22]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(123456789)\n# create m = 25 FRC polymer configurations\nm = 10\nsampleIndices = np.arange(0.0, 501.0, 1).astype(int)\nX3_vec, Y3_vec = FRC_wrapper(sampleIndices, m, theta = 0.75 * np.pi)')


# In[23]:


m = 10
plot_trajectories(X3_vec, Y3_vec, m)


# ## Assay 2 - FJC mean-squared-end-to-end vector as a function of chain length $N$

# Next, we analyze the mean-squared end-to-end vector as a function of segment length. The goal is of course to verify, that the well known relation
# \begin{align}
# \bigl\langle
# \boldsymbol{R}^2\,\bigl\rangle = Na^2
# \end{align}
# holds for the freely jointed chain (FJC). This is of course equivalent to 
# \begin{align}
# R = \sqrt{\bigl\langle
# \boldsymbol{R}^2\,\bigl\rangle}
# = \sqrt{N} a \, .
# \end{align}
# In the first part of this section on the FJC model we will show the $\sim N$ scaling and in the second part we verify the $\sim a^2$ scaling of the MSD.

# In[24]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(123456789)\nm = 1000 # number of statistically independent configurations\nsampleLengths = np.array([10, 500, 1000, 2500, 5000])\n\nX_ex2, Y_ex2 = FJC(sampleLengths, m)\n\nprint(X_ex2.shape)\nprint(Y_ex2.shape)')


# In[25]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(123456789)\nm = 1000 # number of statistically independent configurations\nsampleLengths = np.array([10, 500, 1000, 2500, 5000])\n\nX_ex2_vec, Y_ex2_vec = FJC_vec(sampleLengths, m)\n\nprint(X_ex2_vec.shape)\nprint(Y_ex2_vec.shape)')


# Here you can compare the execution time of the vectorized version against the execution time of the naive implementation of the FJC model. One gets a speed up of roughly 50 times faster code.

# In[26]:


def plot_msd_FJC(X, Y, a = 1.0e-3):
    
    f, ax = plt.subplots(1)
    f.set_size_inches(5.5, 4.0)
    
    ax.set_xlabel('chain length $N$', fontsize = 18)
    ax.set_ylabel(r'$\langle R^2\rangle$', fontsize = 18)
    labelfontsize = 15.0
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    
    ax.plot([-2.0, 6000.0], [a ** 2 * (-2.0), a ** 2 * (6000.0)], 
            lw = 2.0, 
            color = 'C0', 
            label = r'$\langle R^2\rangle = Na^2$', 
            zorder = 1)
    ax.scatter(X, Y, color = 'k', 
               s = 80,
               marker = 'o',
               facecolors = 'None',
               edgecolors = 'k',
               linewidth = 1.5,
               label = r'FJC (numeric)', zorder = 2)

    ax.set_xlim(-200.0, 5200.0)
    ax.set_ylim(-0.0005, 0.0062)
    
    leg = ax.legend(scatterpoints = 1,
                     markerscale = 1.0,
                     ncol = 1,
                     fontsize = 18)
    for i, legobj in enumerate(leg.legendHandles):
        legobj.set_linewidth(1.5)
    leg.draw_frame(False)
    return None


# In[27]:


sampleLengths_ex2 = np.array([10, 500, 1000, 2500, 5000])
msd_ex2 = get_msd(X_ex2, Y_ex2, sampleLengths_ex2)
plot_msd_FJC(sampleLengths_ex2, msd_ex2, a = 1.0e-3)


# In[28]:


sampleLengths_ex2 = np.array([10, 500, 1000, 2500, 5000])
msd_ex2_vec = get_msd(X_ex2_vec, Y_ex2_vec, sampleLengths_ex2)
plot_msd_FJC(sampleLengths_ex2, msd_ex2_vec, a = 1.0e-3)


# ## Assay 3 FJC mean-squared-end-to-end vector as a function of the segment length $a$

# In[29]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(123456789)\nm = 1000\nN = 2000\nsampleLengths_ex3 = np.array([N])\nstepSizes_ex3 = np.array([1.0e-3, 2.0e-3, 3.0e-3, 5.0e-3, 7.5e-3, 1.0e-2])\nres_ex3 = np.zeros((len(stepSizes_ex3)))\nfor i, a in enumerate(stepSizes_ex3):\n    print("segment length", a)\n    tmpX, tmpY = FJC(sampleLengths_ex3, m, a)\n    msd = get_msd(tmpX, tmpY, sampleLengths_ex3)\n    res_ex3[i] = msd[0]')


# In[30]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(123456789)\nm = 10000 # we can afford 10000 here\nN = 2000\nsampleLengths_ex3 = np.array([N])\nstepSizes_ex3 = np.array([1.0e-3, 2.0e-3, 3.0e-3, 5.0e-3, 7.5e-3, 1.0e-2])\nres_ex3_vec = np.zeros((len(stepSizes_ex3)))\nfor i, a in enumerate(stepSizes_ex3):\n    print("segment length", a)\n    tmpX, tmpY = FJC_vec(sampleLengths_ex3, m, a)\n    msd = get_msd(tmpX, tmpY, sampleLengths_ex3)\n    res_ex3_vec[i] = msd[0]')


# In[31]:


def plot_msd_FJC_a(X, Y, N = 2000.0):
    
    f, ax = plt.subplots(1)
    f.set_size_inches(5.5, 4.0)

    ax.set_xlabel('segment length $a$', fontsize = 18)
    ax.set_ylabel(r'$\langle R^2\rangle$', fontsize = 18)
    labelfontsize = 15.0
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    
    xVals = np.linspace(0.0, 1.05e-2, 500)
    yVals = np.array([N * a ** 2 for a in xVals])
    ax.plot(xVals, yVals, 
            lw = 2.0, color = 'C0', label = r'$\langle R^2\rangle = Na^2$')
    
    ax.scatter(X, Y, color = 'k', 
               s = 80,
               marker = 'o',
               facecolors = 'None',
               edgecolors = 'k',
               linewidth = 1.5,
               label = r'FJC (numeric)', zorder = 2)

    ax.set_xlim(0.0, 1.05e-2)
    #ax.set_ylim(-0.0005, 0.0062)
    
    leg = ax.legend(scatterpoints = 1,
                     markerscale = 1.0,
                     ncol = 1,
                     fontsize = 18)
    for i, legobj in enumerate(leg.legendHandles):
        legobj.set_linewidth(1.5)
    leg.draw_frame(False)
    return None


# In[32]:


stepSizes_ex3 = np.array([1.0e-3, 2.0e-3, 3.0e-3, 5.0e-3, 7.5e-3, 1.0e-2])
plot_msd_FJC_a(stepSizes_ex3, res_ex3, N = 2000.0)


# In[33]:


stepSizes_ex3 = np.array([1.0e-3, 2.0e-3, 3.0e-3, 5.0e-3, 7.5e-3, 1.0e-2])
plot_msd_FJC_a(stepSizes_ex3, res_ex3_vec, N = 2000.0)


# ## Assay 4 FRC mean-squared-end-to-end vector as a function of N

# In[34]:


get_ipython().run_cell_magic('time', '', "np.random.seed(923456789) # fix random seed\nm = 2000 # number of independent configurations\nsampleLengths_ex4 = np.array([10, 500, 1000, 2500, 5000])\nX_ex4, Y_ex4 = FRC(sampleLengths_ex4, m)\n\nnp.savetxt('./assay_4_xdata.txt', X_ex4, fmt = '%.8f')\nnp.savetxt('./assay_4_ydata.txt', Y_ex4, fmt = '%.8f')")


# In[36]:


get_ipython().run_cell_magic('time', '', '# vectorized version\nnp.random.seed(923456789) # fix random seed\nm = 2000 # number of independent configurations\nsampleLengths_ex4 = np.array([10, 500, 1000, 2500, 5000])\nX_ex4_vec, Y_ex4_vec = FRC_wrapper(sampleLengths_ex4, m)')


# In[37]:


def plot_msd_FRC(X, Y, a = 1.0e-3, theta = 0.2 * np.pi):
    
    f, ax = plt.subplots(1)
    f.set_size_inches(5.5, 4.0)
    
    ax.set_xlabel('chain length $N$', fontsize = 18)
    ax.set_ylabel(r'$\langle R^2\rangle$', fontsize = 18)
    labelfontsize = 15.0
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    
    xVals = np.linspace(0.0, 6000.0, 500)
    yVals = np.array([N * a ** 2 * FloryC(theta) for N in xVals])
    ax.plot(xVals, yVals, 
            lw = 2.0, 
            color = 'C3',
            label = r'$\langle R^2\rangle = C_{\infty}(\theta)Na^2$',
            zorder = 1)
    
    ax.scatter(X, Y, color = 'k', 
               s = 80,
               marker = 'o',
               facecolors = 'None',
               edgecolors = 'k',
               linewidth = 1.5,
               zorder = 2,
               label = r'FRC (numeric)')

    ax.set_xlim(-200.0, 5200.0)
    #ax.set_ylim(-0.0005, 0.0062)
    
    leg = ax.legend(scatterpoints = 1,
                     markerscale = 1.0,
                     ncol = 1,
                     fontsize = 18)
    for i, legobj in enumerate(leg.legendHandles):
        legobj.set_linewidth(1.5)
    leg.draw_frame(False)
    return None


# In[38]:


file = './assay_4_xdata.txt'
X_ex4 = np.genfromtxt(file)
file = './assay_4_ydata.txt'
Y_ex4 = np.genfromtxt(file)


# In[39]:


sampleLengths_ex4 = np.array([10, 500, 1000, 2500, 5000])
msd_ex4 = get_msd(X_ex4, Y_ex4, sampleLengths_ex4)
plot_msd_FRC(sampleLengths_ex4, msd_ex4)


# In[40]:


# visualization of the same result, using the vectorized FRC code
sampleLengths_ex4 = np.array([10, 500, 1000, 2500, 5000])
msd_ex4_vec = get_msd(X_ex4_vec, Y_ex4_vec, sampleLengths_ex4)
plot_msd_FRC(sampleLengths_ex4, msd_ex4_vec)


# The proportionalty factor here is Flory's characteristic ratio $C_{\infty}$, which for the freely rotating chain model is
# \begin{align}
# C_{\infty}(\theta) = \dfrac{1+\cos(\theta)}{1-\cos(\theta)} \, .
# \end{align}
# With this, the MSD for the FRC model is of course
# \begin{align}
# \bigl\langle
# \boldsymbol{R}^2\,\bigl\rangle = C_{\infty}(\theta)Na^2 = \dfrac{1+\cos(\theta)}{1-\cos(\theta)}\, Na^2 \, .
# \end{align}

# ## Assay 5 FRC mean-squared-end-to-end vector as a function of the rotation angle $\theta$

# In[41]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(123456789)\nm = 1000 # number of independent configurations\nN = 2000\n\nsampleLengths_ex5 = np.array([N])\n\nthetas = np.array([0.02 * np.pi, 0.05 * np.pi,\\\n                   0.1 * np.pi, 0.3 * np.pi, 0.6 * np.pi, np.pi])\n\nres_ex5 = np.zeros((len(thetas)))\n\nfor i, theta in enumerate(thetas):\n    print("theta", theta)\n    tmpX, tmpY = FRC(sampleLengths_ex5, m, a = 1.0e-3, theta = theta)\n    tmpMSD = get_msd(tmpX, tmpY, sampleLengths_ex5)\n    res_ex5[i] = tmpMSD[0]\n\nnp.savetxt(\'./assay_5_msd.txt\', res_ex5, fmt = \'%.8f\')')


# In[43]:


file = './assay_5_msd.txt'
res_ex5 = np.genfromtxt(file)
print(res_ex5.shape)


# In[44]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(123456789)\nm = 1000 # number of independent configurations\nN = 2000\n\nsampleLengths_ex5 = np.array([N])\n\nthetas = np.array([0.02 * np.pi, 0.05 * np.pi,\\\n                   0.1 * np.pi, 0.3 * np.pi, 0.6 * np.pi, np.pi])\n\nres_ex5_vec = np.zeros((len(thetas)))\n\nfor i, theta in enumerate(thetas):\n    print("theta", theta)\n    tmpX, tmpY = FRC_wrapper(sampleLengths_ex5, m, a = 1.0e-3, theta = theta)\n    tmpMSD = get_msd(tmpX, tmpY, sampleLengths_ex5)\n    res_ex5_vec[i] = tmpMSD[0]')


# In[45]:


def plot_msd_FRC_theta(X, Y, N = 2000.0, a = 1.0e-3):
    
    f, ax = plt.subplots(1)
    f.set_size_inches(5.5, 4.0)
    
    ax.set_xlabel(r'rotation angle $\theta$', fontsize = 18)
    ax.set_ylabel(r'$\langle R^2\rangle$', fontsize = 18)
    labelfontsize = 15.0
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    
    xVals = np.linspace(0.05, 1.2 * np.pi, 500)
    yVals = np.array([N * a ** 2 * FloryC(theta) for theta in xVals])
    ax.plot(xVals, yVals, 
            lw = 2.0, 
            color = 'C3', 
            label = r'$\langle R^2\rangle = C_{\infty}(\theta)Na^2$',
            zorder = 1)
    ax.scatter(X, Y, color = 'k', 
               s = 80,
               marker = 'o',
               facecolors = 'None',
               edgecolors = 'k',
               linewidth = 1.5,
               zorder = 2,
               label = r'FRC (numeric)')

    ax.set_xlim(-0.1, 1.05 * np.pi)
    ax.set_ylim(-0.085, 2.0062)
    
    leg = ax.legend(scatterpoints = 1,
                     markerscale = 1.0,
                     ncol = 1,
                     fontsize = 18)
    for i, legobj in enumerate(leg.legendHandles):
        legobj.set_linewidth(1.5)
    leg.draw_frame(False)
    return None


# In[46]:


thetas = np.array([0.02 * np.pi, 0.05 * np.pi,                   0.1 * np.pi, 0.3 * np.pi, 0.6 * np.pi, np.pi])
plot_msd_FRC_theta(thetas, res_ex5, N = 2000.0)


# In[47]:


# visualization of the same result using the vectorized version of the FRC model
thetas = np.array([0.02 * np.pi, 0.05 * np.pi,                   0.1 * np.pi, 0.3 * np.pi, 0.6 * np.pi, np.pi])
plot_msd_FRC_theta(thetas, res_ex5_vec, N = 2000.0)


# In[48]:


def plot_msd_FRC_theta_LOG(X, Y, N = 2000.0, a = 1.0e-3):
    
    f, ax = plt.subplots(1)
    f.set_size_inches(5.5, 4.0)
    
    ax.set_xlabel(r'rotation angle $\theta$', fontsize = 18)
    ax.set_ylabel(r'$\langle R^2\rangle$', fontsize = 18)
    labelfontsize = 15.0
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    
    xVals = np.linspace(0.01, 1.2 * np.pi, 500)
    yVals = np.array([N * a ** 2 * FloryC(theta) for theta in xVals])
    ax.plot(xVals, yVals, 
            lw = 2.0, 
            color = 'C3', 
            label = r'$\langle R^2\rangle = C_{\infty}(\theta)Na^2$',
            zorder = 1)
    ax.scatter(X, Y, color = 'k', 
               s = 80,
               marker = 'o',
               facecolors = 'None',
               edgecolors = 'k',
               linewidth = 1.5,
               zorder = 2,
               label = r'FRC (numeric)')

    ax.set_xlim(-0.1, 1.05 * np.pi)
    ax.set_ylim(1.0e-9, 5.1)
    
    ax.set_yscale('log')
    
    leg = ax.legend(scatterpoints = 1,
                     markerscale = 1.0,
                     ncol = 1,
                     fontsize = 18)
    for i, legobj in enumerate(leg.legendHandles):
        legobj.set_linewidth(1.5)
    leg.draw_frame(False)
    return None


# In[49]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(123456789)\nm = 10000 # number of independent configurations\nN = 2000\n\nsampleLengths_ex6 = np.array([N])\n\nthetas = np.array([0.02 * np.pi, 0.05 * np.pi,\\\n                   0.1 * np.pi, 0.2 * np.pi, 0.3 * np.pi, 0.4 * np.pi,\\\n                   0.5 * np.pi, 0.6 * np.pi, 0.7 * np.pi, 0.8 * np.pi,\\\n                   0.9 * np.pi, 0.95 * np.pi, 0.98 * np.pi, 0.99 * np.pi])\n\nres_ex6_vec = np.zeros((len(thetas)))\n\nfor i, theta in enumerate(thetas):\n    print("theta", theta)\n    tmpX, tmpY = FRC_wrapper(sampleLengths_ex6, m, a = 1.0e-3, theta = theta)\n    tmpMSD = get_msd(tmpX, tmpY, sampleLengths_ex6)\n    res_ex6_vec[i] = tmpMSD[0]')


# In[50]:


# visualization of the same result using the vectorized version of the FRC model
thetas = np.array([0.02 * np.pi, 0.05 * np.pi,                   0.1 * np.pi, 0.2 * np.pi, 0.3 * np.pi, 0.4 * np.pi,                   0.5 * np.pi, 0.6 * np.pi, 0.7 * np.pi, 0.8 * np.pi,                   0.9 * np.pi, 0.95 * np.pi, 0.98 * np.pi, 0.99 * np.pi])
plot_msd_FRC_theta_LOG(thetas, res_ex6_vec, N = 2000.0)
# print(res_ex6_vec)

