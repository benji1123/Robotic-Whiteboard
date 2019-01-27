''' 
# PolarGraph Project
This Script: Reduces Images to Edges for Sketching
 
    [1] Liam S. -> hardware
    [2] Ben L.  -> software
'''
 
import numpy as np
import cv2
from tkinter import Tk
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
import readLine
 
# .............................GET IMAGE FROM LOCAL SYSTEM
#GUI Based Selection
Tk().withdraw()
 

# .............................................................. MAIN ALGORITHM .........................................................................
def reduceImage(filename):
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
# .......................................................................................................................................................
 
 
 
# .... CANNY (edge detection)
# produce an image-file that is just an outline of the original image, 
# which the reduceImage() algorithm will further compress.
def autoCanny(image):
    sigma=0.33                  # sigma^ = threshold^
    med = np.median(image)      # px-intensity
    lower = int(max(0, med * (1.0-sigma)))
    upper = int(min(255, med * (1.0+sigma)))
    # Edge Detection using computed Median 
    edgedImage = cv2.Canny(image, lower, upper)
    return edgedImage
 
 
# .... CANNY PRE-PROCESSING
# blur the original image and apply canny-edge detector on it
def draw(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0) # filter image-noise w. 3x3 Kernel
    auto = autoCanny(blurred) # canny the blurred image
    outName = filename[0:-4] + "_canny.png" 
    cv2.imwrite(outName, auto) # write image to local file system
    cv2.waitKey(0)
    return auto
 
 
# .... REDUCTION (to just image-nodes we'll draw) 
# reduce image to just a fraction of the original pixels, 
# which the path algorithm will find a drawing path for
def singleLineImage(image):
    x,y = image.shape
    tol = 2 # max dist. btwn neighbor-nodes 
 
    # outta-bounds prevention (canvas)
    image[:,0] = 255
    image[:,y-tol:]=255
    image[0,:] = 255
    image[x-tol:,:] = 255
    newImg = np.ones((x,y))*255
    imgSimple = image
 
    # Image-"Edges" = nodes w/in 2px of each other  
    G = nx.Graph()
    for ix in range(x):
        for iy in range(y):
            if imgSimple[ix,iy] == 0:   # draw black-px's
                # find neighbour black-px
                neighbours = []
                for subx in [-2,-1,0,1,2]:     # legal distances between neighbour-nodes
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
 
 
# .. ( X ) .. P A T H  P L A N N I N G 
# Compute the path the machine will follow, as an
# array of coordinates
def load_path(filename): # file = img_name 
    with open(filename[0:-4] + '_path.p', 'rb') as myfile:
        data = pickle.load(myfile)
    return data
 
def pa_to_path(filename): # file = img_name
    print("preview")
    f = filename[0:-4] + '_preview.png'
    image = PIL.Image.open(f)
    image = image.rotate(180)
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    imageArr = np.array(image)
    y,x,z = imageArr.shape
    startx,starty = None,None
 
    # begin path @ leftmost node
    G = nx.Graph();                 print("graph created");
    imgSimple = imageArr.sum(2);    print(imgSimple);
 
    print("starting search")
    for ix in range(x):
        for iy in  range(y):
            if imgSimple[iy,ix] == 255: # [iy,ix] or [ix,iy] ?
                neighbors = []
                for subx in [-2,-1,0,1,2]:     # legal distances between neighbour-nodes
                    for suby in [-2,-1,0,1,2]:
                        # pos. of neighbouring-node
                        iix = ix+subx
                        iiy = iy+suby
 
                        # check if edge is line
                        if imgSimple[iiy,iix] == 255:                       # euclidean-dist 
                            G.add_edge((ix,iy), (iix,iiy), weight = ((ix-iix)**2 + (iy-iiy)**2)**0.5)
 
                # first/leftmost black-px
                if startx is None:
                    startx, starty = ix, iy
 
    print("Starting at: [", startx, " ", starty, "]")
 
    # call (DFS) depth first search   
    print("calling DFS")                                    
    path, nodes = dfstoPath(G,startx,starty)
    print("DFS complete") 
    x_nodes = [z[0] for z in nodes]
    y_nodes = [z[1] for z in nodes]
#    plt.scatter(x=x_nodes, y=y_nodes,s=1.)               
#    plt.show()
    return path, nodes
 
 
 
# .. DEPTH-FIRST-SERACH
def dfstoPath(G, startx, starty):
<<<<<<< HEAD
	T = nx.dfs_tree(G, (startx, starty))
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
				

# DIGITAL DRAWER
# get set of absolute coordaintes, which
# will be converted to a motor-readable format .txt file
def turtleDraw(filename):
	file_no_extension = filename[0:-4]

	# Load path-file
	with open(file_no_extension+'_path.p', 'rb') as myfile:
		data = pickle.load(myfile) # un-compressed instructions [list of tuples]

	# Set starting Coordinates
	coords = [] 
	coords.append((-200,-200)) # start-coord

	# Build Array of (absolute) Coordinates 
	i=1 
	xyMax = -100000000000000000000000000
	xyMin = 100000000000000000000000000

	while i < len(data):
		next = (coords[i-1][0] + data[i][0] , coords[i-1][1] + data[i][1])
		coords.append(next)
		# Get max/min to do feature-scaling
		if next[0] > xyMax: xyMax = next[0];	
		if next[0] < xyMin: xyMin = next[0];
		if next[1] > xyMax: xyMax = next[1];	
		if next[1] < xyMin: xyMin = next[1];
		i+=1

	# DATA COMPRESSION --> COORDS
	dY, dX = coords[1][1]-coords[0][1], coords[1][0]-coords[0][0]
	pastSlope = 0
	if dX != 0:
		pastSlope = (dY)/(dX)

	i=3
	while i < len(coords):
		currSlope = 0
		deltaY, deltaX = coords[i][1]-coords[i-1][1], coords[i][0]-coords[i-1][0]
		if deltaX != 0:
				currSlope = (deltaY)/(deltaX)

		# Merge re-occuring slopes
		tolerance = 0.20*pastSlope
		if currSlope <= pastSlope+tolerance or currSlope >= pastSlope - tolerance:
			del coords[i-2:i]
		i += 1
		

	# CONVERT to MOTOR FILE FORMAT
	MotorCoords(coords, xyMax, xyMin)
	# SIMULATE DRAWING PATH
	fast = turtle.Turtle() # drawing object
	fast.color("purple")

	# draw image via coordinates-array
	i=0
	scale = 300
	offset = 350
	while i < (len(coords)):
		#print(i,"/",len(data))
		if i == 0:
			fast.penup()
		x = coords[i][0]
		y = coords[i][1]
		x = ((x - xyMin)/(xyMax - xyMin)) *scale+offset
		y = ((y- xyMin)/(xyMax - xyMin)) *scale+offset
		fast.goto((x,y))

		if i == 0:
			fast.pendown()
		i+=1
	turtle.getscreen()._root.mainloop() # keep drawing open after it is complete


def MotorCoords(coords, xyMax, xyMin):
	scale = 300
	offset = 350
	coords = coords[1:] 
	m = open("code.txt", "wt")
	m.write("P1\n")

	# Move Motor to Start-Coordinate
	_x = ((coords[0][0] - xyMin)/(xyMax - xyMin))*scale+offset
	_y = ((coords[0][1]- xyMin)/(xyMax - xyMin))*scale+offset
	instruction = "X" + str(int(_x)) + " Y" + str(int(_y)) + "\n"
	m.write(instruction)
	m.write("P0\n")

	# Store Array to String
	instructions_string = ""

	# Motor Coordinates
	counter = 0
	for xy in coords[1:]:
		# normalize xy for scaling in Arduino 
		print(counter," / ",len(coords))
		x = ((xy[0] - xyMin)/(xyMax - xyMin))*scale+offset
		y = ((xy[1]- xyMin)/(xyMax - xyMin))*scale+offset
		instruction = "X" + str(int(x)) + " Y" + str(int(y)) + "\n"
		instructions_string += instruction
		m.write(instruction)

	m.write("\n") # newline indicates file's end
	m.close()


reduceImage(filename)
turtleDraw(filename)
=======
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
    file_no_extension = filename[0:-4]
    # LOAD PATH-FILE, which contains list of relative-coordinates
    with open(file_no_extension+'_path.p', 'rb') as myfile:
        data = pickle.load(myfile) # un-compressed instructions [list of tuples]
 
    # Starting coordinates are first2 elements of list
    # Build Series of Absolute Coordinates from Relative-Coordinates
    coords = [] # store coordinates here
    startx, starty = -200,-200                      # the image is drawn starting at the bottom-left
    coords.append((startx,starty))
 
    i=1 # index-0 is the starting-coord, but we manually set this to top left of board
    xyMax = -100000000000000000000000000
    xyMin = 100000000000000000000000000
    while i < len(data):
        next = (coords[i-1][0] + data[i][0] , coords[i-1][1] + data[i][1])
        coords.append(next)
        # Get max/min to do feature-scaling
        if next[0] > xyMax: xyMax = next[0];    
        if next[0] < xyMin: xyMin = next[0];
        if next[1] > xyMax: xyMax = next[1];    
        if next[1] < xyMin: xyMin = next[1];
        i+=1
 
    # CONVERT to MOTOR FILE FORMAT
    MotorCoords(coords, xyMax, xyMin)



    # MOTOR INSTRUCTIONS TRANSLATION
def MotorCoords(coords, xyMax, xyMin):
    scale = 300
    offset = 350
    coords = coords[1:] 
    m = open("code.txt", "wt")
    m.write("P1\n")
        # Move Motor to Start-Coordinate
    _x = ((coords[0][0] - xyMin)/(xyMax - xyMin))*scale+offset
    _y = ((coords[0][1]- xyMin)/(xyMax - xyMin))*scale+offset
    instruction = "X" + str(int(_x)) + " Y" + str(int(_y)) + "\n"
    m.write(instruction)
    m.write("P0\n")
        # Store Array to String
    instructions_string = ""
        # Motor Coordinates
    for xy in coords[1:]:
        # normalize xy for scaling in Arduino 
        x = ((xy[0] - xyMin)/(xyMax - xyMin))*scale+offset
        y = ((xy[1]- xyMin)/(xyMax - xyMin))*scale+offset
        instruction = "X" + str(int(x)) + " Y" + str(int(y)) + "\n"
        instructions_string += instruction
        m.write(instruction)
    m.write("\n") # newline indicates file's end
    m.close()
 
 
def run(uploaded_file):
<<<<<<< HEAD
        reduceImage(uploaded_file)
        turtleDraw(uploaded_file)
>>>>>>> d168b754f86d6558531e38cac6666594abb44c38
=======
    reduceImage(uploaded_file)
    turtleDraw(uploaded_file)
    print("doneturtle")
    readLine.draw()
>>>>>>> 9874e2e518be1e44f0166c2ff0b0d58799c0961d
