''' 
# PolarGraph Project
This Script: Reduces Images to Edges for Sketching
    [1] Liam S. -> hardware
    [2] Ben L.  -> software
'''
import numpy as np
import cv2
from tkinter.filedialog import askopenfilename
import PIL
from PIL import Image, ImageFilter
import PIL.ImageOps
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import PIL.ImageOps
import turtle
import pickle
import copy
 

PREVIEW = False

# Intiate Path Creation Process
def createPath(filename):
    # Image Processing
    edges = 255-draw(filename) 
    image = PIL.Image.open(filename[0:-4] + "_canny.png") # get cannied-iamge from local file system
    image.thumbnail((256,144), Image.NEAREST)
    simplified = singleLineImage(edges) # reduce to one-stroke drawable
    img = PIL.Image.fromarray(edges)    
 
    # Path Output
    # send  simplified-image to the path-finding algorithm,
    # which a drawing-path will be identified 
    PIL.Image.fromarray(simplified).convert('RGBA').save(filename[0:-4] + '_preview.png')
    pp,nn = pa_to_path(filename)
    with open(filename[0:-4]+'_path.p', 'wb') as fp: # output path-file
        pickle.dump(pp, fp)
    load_path(filename) # write file containing drawing-path coordinates 

 
# Apply Canny Edge Detection
def draw(filename):

    def autoCanny(image):
        sigma=0.33                  # sigma^ = threshold^
        med = np.median(image)      # px-intensity
        lower = int(max(0, med * (1.0-sigma)))
        upper = int(min(255, med * (1.0+sigma)))
        # Edge Detection using computed Median 
        edgedImage = cv2.Canny(image, lower, upper)
        return edgedImage

    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    auto = autoCanny(blurred)
    cv2.imwrite(filename[0:-4] + "_canny.png" , auto)
    cv2.waitKey(0)
    return auto
 
 
# Reduce Image to Usable Pixels
def singleLineImage(image):
    x,y = image.shape
    tol = 2 # max dist. btwn neighbor-nodes 
    image[:,0] = 255
    image[:,y-tol:]=255
    image[0,:] = 255
    image[x-tol:,:] = 255
    newImg = np.ones((x,y))*255
    imgSimple = image
 
    G = nx.Graph()
    for ix in range(x):
        for iy in range(y):
            if imgSimple[ix,iy] == 0: # draw black-px's
                # find neighbour black-px
                neighbours = []
                for subx in [-2,-1,0,1,2]: # legal distances between neighbour-nodes
                    for suby in [-2,-1,0,1,2]:
                        # pos. of neighbouring-node
                        iix = ix+subx; iiy = iy+suby
                        # check if edge is formed
                        if imgSimple[iix,iiy] == 0:
                            G.add_edge((ix,iy), (iix,iiy))
    
    # Longest Path max's Resolution
    G = max(nx.connected_component_subgraphs(G), key=len)
    for node in G.nodes:
        newImg[node[0], node[1]] = 0

    return newImg


# (helper) load drawing-path
def load_path(filename): # file = img_name 
    with open(filename[0:-4] + '_path.p', 'rb') as myfile:
        data = pickle.load(myfile)
    return data


# Return Best Drawing Path
def pa_to_path(filename): # file = img_name
    print("preview")
    f = filename[0:-4] + '_preview.png'
    image = PIL.Image.open(f)
    image = image.rotate(180)
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    imageArr = np.array(image)
    y,x,z = imageArr.shape
    startx,starty = None,None
 
    # Create Graph DS from simplified image
    G = nx.Graph();
    imgSimple = imageArr.sum(2);
    print("starting search")
    for ix in range(x):
        for iy in  range(y):
            if imgSimple[iy,ix] == 255: # [iy,ix] or [ix,iy] ?
                for subx in [-2,-1,0,1,2]: # displacement to neighbour pixels
                    for suby in [-2,-1,0,1,2]:
                        # pos. of neighbouring-node
                        iix = ix+subx
                        iiy = iy+suby

                        # check if edge is line
                        if imgSimple[iiy,iix] == 255:                       
                            # euclidean-dist 
                            G.add_edge((ix,iy), (iix,iiy), weight = (
                                (ix-iix)**2 + (iy-iiy)**2)**0.5
                            )
                # first/leftmost black-px
                if startx is None:
                    startx, starty = ix, iy
 
    # Determine drawing-path with DFS
    print("calling DFS")                                    
    path, nodes = dfstoPath(G,startx,starty)
    x_nodes = [z[0] for z in nodes]
    y_nodes = [z[1] for z in nodes]
    plt.scatter(x=x_nodes, y=y_nodes,s=1.)               
    plt.show()
    return path, nodes


# (helper) Depth First Search 
def dfstoPath(G, startx, starty):
    T = nx.dfs_tree(G, (startx, starty))
    print("A\nA\nA\nA\n")
    #print(T.edges())
    mvmts = []
    mvmts.append((startx,starty))
    nodes = []
    Tedges = T.edges()
    prevte = [None,[z for z in Tedges][0][0]]
    for te in Tedges:
        if te[0] == prevte[1]:
            pass # the node smoothly connects to the next node
        else:
            # we need to backtrack first before we can add the movement
            backtrack_start = prevte[1]
            backtrack_end = te[0]
            path_back = nx.shortest_path(G,backtrack_start,backtrack_end)
 
            for bb in range(len(path_back)-1):
                start_b = path_back[bb]
                dest_b  = path_back[bb+1]
                xdiff_b = dest_b[0] - start_b[0]
                ydiff_b = dest_b[1] - start_b[1]
                mvmts.append((xdiff_b, ydiff_b))
                nodes.append(dest_b)
 
        start = te[0]
        dest = te[1]
        xdiff = dest[0] - start[0]
        ydiff = dest[1] - start[1]
        mvmts.append((xdiff, ydiff))
        nodes.append(dest)
        prevte = te

    return mvmts, nodes
 
 
def turtleDraw(filename):
    # read the generated coords-file in dir
    file_no_extension = filename[0:-4]
    with open(file_no_extension+'_path.p', 'rb') as myfile:
        data = pickle.load(myfile)
 
    # Coordinates are relative, convert to absolute
    coords = [(-200,-200)]
    xyMax = -100000000000000000000000000
    xyMin = 100000000000000000000000000 # (bad practice)
    
    i=1
    while i < len(data):
        next = (coords[i-1][0] + data[i][0] , coords[i-1][1] + data[i][1])
        coords.append(next)
        # Get max/min to do feature-scaling
        if next[0] > xyMax: xyMax = next[0];    
        if next[0] < xyMin: xyMin = next[0];
        if next[1] > xyMax: xyMax = next[1];    
        if next[1] < xyMin: xyMin = next[1];
        i+=1
 
    # Generate Coords-File in Dir
    MotorCoords(coords, xyMax, xyMin)

    # Preview the Drawing Process in a Turtle Window
    if(PREVIEW):
        fast = turtle.Turtle() # drawing object
        fast.color("blue")
        fast.penup(); fast.setposition(coords[0]); fast.pendown();
     
        i=0
        scale = 300
        offset = 350
        
        while i < (len(data)):
            print(i,"/",len(data))
            if i == 0:
                fast.penup()
            # Normalize and Scale Coordinates
            x = coords[i][0]
            y = coords[i][1]
            x = ((x - xyMin)/(xyMax - xyMin)) *scale+offset 
            y = ((y- xyMin)/(xyMax - xyMin)) *scale+offset
            fast.goto(coords[i])

            if i == 0:
                fast.pendown()
            i+=1
            
        turtle.getscreen()._root.mainloop() # keep drawing open after it is complete


# Generate File for Motors to read Coordinates from
def MotorCoords(coords, xyMax, xyMin):
    # alter these depending on whiteboard size
    scale = 300
    offset = 350
    
    coords = coords[1:] 
    m = open("code.txt", "wt")
    m.write("P1\n")
    
    # Ensure Marker goes to Starting Position before Drawing
    _x = ((coords[0][0] - xyMin)/(xyMax - xyMin))*scale+offset
    _y = ((coords[0][1]- xyMin)/(xyMax - xyMin))*scale+offset
    instruction = "X" + str(int(_x)) + " Y" + str(int(_y)) + "\n"
    m.write(instruction)
    m.write("P0\n")
    
    # Motor Coordinates
    instructions_string = ""
    for xy in coords[1:]:
        # normalize xy for scaling in Arduino 
        x = ((xy[0] - xyMin)/(xyMax - xyMin))*scale+offset
        y = ((xy[1]- xyMin)/(xyMax - xyMin))*scale+offset
        instruction = "X" + str(int(x)) + " Y" + str(int(y)) + "\n"
        instructions_string += instruction
        m.write(instruction)
    
    m.write("\n") # newline indicates file's end
    m.close()
 

# Execute this
def run(uploaded_file):
        createPath(uploaded_file)
        turtleDraw(uploaded_file)
