
''' 
# PolarGraph Project
This Script: Reduces Images to Edges for Sketching

	[1]	Liam S. -> hardware
	[2]	Ben L.  -> software
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


# .............................GET IMAGE FROM LOCAL SYSTEM
#GUI Based Selection
Tk().withdraw()
file_address = askopenfilename()
original_img = cv2.imread(file_address)
cv2.imwrite("image_to_draw.png", original_img)
filename = "image_to_draw.png"


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
	sigma=0.33 					# sigma^ = threshold^
	med = np.median(image)		# px-intensity
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
	tol = 2	# max dist. btwn neighbor-nodes 

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
			if imgSimple[ix,iy] == 0: 	# draw black-px's
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
	G = nx.Graph(); 				print("graph created");
	imgSimple = imageArr.sum(2); 	print(imgSimple);

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
						if imgSimple[iiy,iix] == 255:						# euclidean-dist 
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
	plt.scatter(x=x_nodes, y=y_nodes,s=1.)               
	plt.show()
	return path, nodes



# .. DEPTH-FIRST-SERACH
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
	x_max = -100000000000000000000000000
	x_min = 100000000000000000000000000
	y_max = -100000000000000000000000000
	y_min = 1000000000000000000000000

	while i < len(data):
		next = (coords[i-1][0] + data[i][0] , coords[i-1][1] + data[i][1])
		coords.append(next)

		if next[0] > x_max: x_max = next[0];	#update max for scaling
		if next[0] < x_min: x_min = next[0];

		if next[1] > y_max: y_max = next[1];
		if next[1] < y_min: y_min = next[1];
		i+=1

	# CONVERT to MOTOR FILE FORMAT
	MotorCoords(coords, x_max, x_min, y_max, y_min)

	# SIMULATE DRAWING PATH
	fast = turtle.Turtle() # drawing object
	fast.color("purple")
	fast.penup(); fast.setposition(coords[0]); fast.pendown(); # go to starting-pos

	# draw image via coordinates-array
	i=0
	while i < (len(data)):
		#print(i,"/",len(data))
		fast.goto(coords[i])
		i+=1
	turtle.getscreen()._root.mainloop() # keep drawing open after it is complete


def MotorCoords(coords, xMax, xMin, yMax, yMin):
	print(xMax)
	print(yMax)
	coords = coords[1:] 
	m = open("motor_coords.txt", "wt")
	for xy in coords:
		# normalize xy for scaling in Arduino 
		x = (xy[0] - xMin)/(xMax - xMin)
		y = (xy[1]- yMin)/(yMax - yMin)
		instruction = "X" + str(x) + " Y" + str(y) + "\n"
		m.write(instruction)

	m.write("\n") # newline indicates file's end
	m.close()

reduceImage(filename)
turtleDraw(filename)