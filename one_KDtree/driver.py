from kdtree import Point
from kdtree import KDCell
import timeit

points = Point.genPoints(1000000)
tests = Point.genPoints(100)

total = 0
kdtime = 0
naivetime = 0

#use kdtree
kdguesses = []
start_time = timeit.default_timer()

cell = KDCell(points)

end_treebuild_time = timeit.default_timer()

for p in tests :
	kdguess, comps = cell.countingFindNearest(p)
	total += comps
	kdguesses.append(kdguess)

end_time = timeit.default_timer()

print( "Time to build KDTree: {}".format(end_treebuild_time - start_time))
print( "Time to perform NN search: {}".format(end_time - end_treebuild_time))
print( "Total elapsed time for kdtree: {}".format(end_time - start_time))
print( "Average number of comparisons ", (float(total) / len(tests)))

#use naive
naiveguesses = []
start_time = timeit.default_timer()

for p in tests :
	naiveguess = p.findNearest(points)
	naiveguesses.append(naiveguess)
	
print( "Elapsed time for naive: {}".format(timeit.default_timer() - start_time))

print( "Verifying answers ...")

for p, q in zip(kdguesses, naiveguesses) :
	assert(p == q)
	
print( "OK!")