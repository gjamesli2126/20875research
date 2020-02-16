import numpy as np
import collections

#data point -- 2D
class Point :
	
	def __init__(self, x=0.0, y=0.0) :
		self.data = (x, y)
		
	def __getitem__(self, i) :
		return self.data[i]
		
	def __repr__(self) :
		retval = "Point: ({}, {})".format(self.data[0], self.data[1])
		return retval
		
	#compute distance from a single point
	def dist(self, other) :
		return np.sqrt((self[0] - other[0])**2 + (self[1] - other[1])**2)
		
	#compute whether any point p in box b can be within distance r of self
	def canIntersect(self, r, bbox) :# bbox= bounding box

		xcoord = self[0]
		ycoord = self[1]
		if (self[0] <= bbox[0]) :
			xcoord = bbox[0]
		elif (self[0] >= bbox[1]) :
			xcoord = bbox[1]
			
		if (self[1] <= bbox[2]) :
			ycoord = bbox[2]
		elif (self[1] >= bbox[3]) :
			ycoord = bbox[3]
			
		comp = Point(xcoord, ycoord)
			
		# if (abs(r - self.dist(comp)) < 0.0001) :
		# 	print "close call"
		
		return r >= self.dist(comp)
	
	def findNearest(self, points) :
		guess = points[0]
		guessDist = self.dist(guess)
		for p in points :
			tmp = self.dist(p)
			if guessDist > tmp :
				guess = p
				guessDist = tmp
		return guess
	
	#generate a list of points within a certain range
	@staticmethod
	def genPoints(numpoints, bbox = (-10.0, 10.0, -10.0, 10.0)) :
		retval = []
		for _ in range(0, numpoints):
			retval.append(Point(np.random.uniform(bbox[0], bbox[1]), np.random.uniform(bbox[2], bbox[3])))
		return retval
		
	@staticmethod
	def boundingBox(points) :
		if (len(points) == 0) : return [0.0, 0.0, 0.0, 0.0]
		minX = points[0][0]
		maxX = points[0][0]
		minY = points[0][1]
		maxY = points[0][1]		
		for p in points :
			xval = p[0]
			yval = p[1]			
			if (xval < minX) : minX = xval 
			if (xval > maxX) : maxX = xval 
			if (yval < minY) : minY = yval 
			if (yval > maxY) : maxY = yval 
		return [minX, maxX, minY, maxY]
		
#KDCell
class KDCell :
	
	def __init__(self, points, bounds = None) :
		self.points = list(points)
		if (bounds == None) :
			self.bounds = Point.boundingBox(self.points)
		else :
			self.bounds = bounds
			
		#figure out splitdim
		if ((self.bounds[1] - self.bounds[0]) > (self.bounds[3] - self.bounds[2])) :
			self.splitdim = 0
		else :
			self.splitdim = 1
			
		#sort points based on split dim
		self.points.sort(key = lambda x : x[self.splitdim])
			
		#figure out split point
		midpoint = len(self.points) / 2
		self.median = self.points[midpoint]
		
		leftpoints = self.points[:midpoint]
		rightpoints = self.points[midpoint + 1:]

		#compute new bounding boxes by splitting along the midpoint:
		leftbox = list(self.bounds)
		rightbox = list(self.bounds)
		#if splitdim = 0, leftbox's maxX (1) is median's x (median[0]) and rightbox's minX (0) is median's x (median[0])					
		#if splitdim = 1, leftbox's maxY (3) is median's y (median[1]) and rightbox's minY (2) is median's y (median[1])
		#so leftbox's [splitdim * 2 + 1] is median[splitdim] and rightbox's [splitdim * 2] is median[splitdim]
		# writing it out the long way:
		# if (splitdim == 0) :
		# 	leftbox[1] = self.median[0]
		# 	rightbox[0] = self.median[0]
		# else :
		# 	leftbox[3] = self.median[1]
		# 	rightbox[2] = self.median[1]
		leftbox[self.splitdim * 2 + 1] = self.median[self.splitdim]
		rightbox[self.splitdim * 2] = self.median[self.splitdim]
		
		#create child KDCells
		if (len(leftpoints) != 0) :
			self.left = KDCell(leftpoints, leftbox)
		else :
			self.left = None
			
		if (len(rightpoints) != 0) :
			self.right = KDCell(rightpoints, rightbox)
		else :
			self.right = None
			
	def findNearest(self, point) :
		guess, comps = self.countingFindNearest(point)
		return guess
			
	def countingFindNearest(self, point) :
		guess = self.median
		comps = 0
		guess, comps = self.findNearestRec(point, guess, comps)
		return (guess, comps)
			
	def findNearestRec(self, point, guess, comps) :
		#set the guess to the median if we don't already have one
		if (guess == None) :
			guess = self.median
		
		comps += 1
		
		#if the bounding box is farther than the distance to your guess, stop
		if (point.canIntersect(point.dist(guess), self.bounds) == False) :
			return (guess, comps)# stop the recursion if search Fin numbers.


			
		#if the current point is closer, change your guess
		if (point.dist(self.median) < point.dist(guess)) :
			guess = self.median
			
		#figure out which direction to go: left if "less" than the median
		#otherwise right. Don't forget to check along the split dimension!
		if (point[self.splitdim] < self.median[self.splitdim]) :
			#search left
			if (self.left != None) :
				guess, comps = self.left.findNearestRec(point, guess, comps)
	
			#search right
			if (self.right != None) :
				guess, comps = self.right.findNearestRec(point, guess, comps)
				
		else :
			#search right
			if (self.right != None) :
				guess, comps = self.right.findNearestRec(point, guess, comps)

			#search left
			if (self.left != None) :
				guess, comps = self.left.findNearestRec(point, guess, comps)
				
		return (guess, comps)