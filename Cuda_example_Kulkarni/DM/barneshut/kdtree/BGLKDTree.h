/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef BGL_TREE_H
#define BGL_TREE_H
#include<boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include<boost/graph/distributed/adjacency_list.hpp>
#include"Point.h"

using namespace boost;
using boost::graph::distributed::mpi_process_group;

class BGLTreeNode 
{
friend class boost::serialization::access;
public:
	char uCount;
	BGLTreeNode():uCount(0)
	{
	}
	~BGLTreeNode(){};

  template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & uCount;
  }
};

class BGLTreeEdge
{
friend class boost::serialization::access;
public:
		bool left; //if true left edge otherwise right edge
		BGLTreeEdge(){}
		BGLTreeEdge(bool l):left(l){}
		~BGLTreeEdge(){}

template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & left;
  }

};


typedef adjacency_list<listS, distributedS<mpi_process_group,vecS>, undirectedS, BGLTreeNode, BGLTreeEdge> BGLGraph;
typedef graph_traits<BGLGraph>::vertex_descriptor BGLVertexdesc;
typedef graph_traits<BGLGraph>::edge_descriptor BGLEdgedesc;
typedef graph_traits<BGLGraph>::adjacency_iterator BGL_AdjVertexIterator; 
typedef graph_traits<BGLGraph>::vertex_iterator BGL_Vertexiter;


#endif
