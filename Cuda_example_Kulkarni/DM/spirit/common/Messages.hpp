/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef MESSAGES_HPP
#define MESSAGES_HPP
#include <boost/serialization/serialization.hpp>
#include "Point.hpp"
#include "../nnvptree/VptreeTypes.hpp"
#include "../nnkdtree/KdtreeTypes.hpp"
#include "../pckdtree/PCKdtreeTypes.hpp"
#include "../bhoctree/OctreeTypes.hpp"

enum SPIRIT_Message{
MESSAGE_UPDATE_PLEAVES_OCT,
MESSAGE_READYTOEXIT,
MESSAGE_BUILDSUBTREE,
MESSAGE_DONETREE,
MESSAGE_TRAVERSE,
MESSAGE_TRAVERSE_BACKWARD,
MESSAGE_REPLICATE_REQ,
MESSAGE_DONEOCTSUBTREE_ACK
};

class MsgBuildSubTree{
friend class boost::serialization::access;
public:
bool headerChunk;
bool moreData;
long int pLeafLabel; 
int depth;
TType type;
bool isleft;
VertexData* vParentData;
bool parentDataPresent;
TPointVector ptv;
Vec center;
long int numVertices;
float diameter;

~MsgBuildSubTree(){for(int i=0;i<ptv.size();i++) delete ptv[i];}
MsgBuildSubTree():type(VPTREE),vParentData(NULL),parentDataPresent(false), moreData(false), headerChunk(false){}

		template<class Archive>
		void save(Archive& ar, const unsigned version) const 
		{
			int size = ptv.size();
 			ar & depth & isleft & type & size & parentDataPresent & headerChunk & moreData & pLeafLabel & numVertices;
			if(type == VPTREE)
			{
				if(vParentData)
				{
					VptreeVertexData* vd = reinterpret_cast<VptreeVertexData*>(vParentData);
					ar & *vd;
				}
				for(int i=0;i<size;i++)
				{
					VptreePoint* c = reinterpret_cast<VptreePoint*>(ptv[i]);
					ar & *c;
				}
			}
			else if (type  == KDTREE)
			{
				for(int i=0;i<size;i++)
				{
					KdtreePoint* c = reinterpret_cast<KdtreePoint*>(ptv[i]);
					ar & *c;
				}
			}
			else if (type  == PCKDTREE)
			{
				for(int i=0;i<size;i++)
				{
					PCKdtreePoint* c = reinterpret_cast<PCKdtreePoint*>(ptv[i]);
					ar & *c;
				}
			}
			else if (type == OCTREE)
			{
				ar & diameter & center;			
				for(int i=0;i<size;i++)
				{
					OctreePoint* c = reinterpret_cast<OctreePoint*>(ptv[i]);
					ar & *c;
				}
			}
		}

		template<class Archive>
		void load(Archive& ar, const unsigned version) 
		{
			int size;
 			ar & depth & isleft & type & size & parentDataPresent & headerChunk & moreData & pLeafLabel & numVertices;
			if(type == VPTREE)
			{
				if(parentDataPresent)
				{
					VptreeVertexData vpd;
					ar & vpd;
					vParentData = new VptreeVertexData(&vpd);
				}
				for(int i=0;i<size;i++)
				{
					VptreePoint p;
					ar & p;
					VptreePoint* c = new VptreePoint(&p);
					ptv.push_back(c);
				}
			}
			else if (type  == KDTREE)
			{
				for(int i=0;i<size;i++)
				{
					KdtreePoint p;
					ar & p;
					KdtreePoint* c = new KdtreePoint(&p);
					ptv.push_back(c);
				}
			}
			else if (type  == PCKDTREE)
			{
				for(int i=0;i<size;i++)
				{
					PCKdtreePoint p;
					ar & p;
					PCKdtreePoint* c = new PCKdtreePoint(&p);
					ptv.push_back(c);
				}
			}
			else if (type == OCTREE)
			{
				ar & diameter & center;			
				for(int i=0;i<size;i++)
				{
					OctreePoint p;
					ar & p;
					OctreePoint* c = new OctreePoint(&p);
					ptv.push_back(c);
				}
			}
		}

		BOOST_SERIALIZATION_SPLIT_MEMBER();
};


class MsgTraverse{
public:
TType type;
TContextVector l;
TBlockId blkStart;
long int pRoot; //pseudo root
long int pLeaf; //pseudo leaf who is to be updated (Vertex* corresponding to parent)
long int pSibling;
int siblingDesc;
std::vector<long int> pSiblings;
std::vector<int> siblingDescs;
MsgTraverse():type(VPTREE){}
~MsgTraverse(){for(int i=0;i<l.size();i++) delete l[i];}
friend class boost::serialization::access;

		template<class Archive>
		void save(Archive& ar, const unsigned version) const 
		{
			int size = l.size();
    			ar & pRoot & pLeaf & blkStart & pSibling & siblingDesc & type & size & pSiblings & siblingDescs;
			if(type == VPTREE)
			{
				for(int i=0;i<size;i++)
				{
					VptreeContext* c = reinterpret_cast<VptreeContext*>(l[i]);
					//ar & c->index & c->nodesTraversed & c->tau & c->closestLabel;
					ar & *c;
				}
			}
			else if(type == KDTREE)
			{
				for(int i=0;i<size;i++)
				{
					KdtreeContext* c = reinterpret_cast<KdtreeContext*>(l[i]);
					ar & *c;
				}
			}
			else if(type == PCKDTREE)
			{
				for(int i=0;i<size;i++)
				{
					PCKdtreeContext* c = reinterpret_cast<PCKdtreeContext*>(l[i]);
					ar & *c;
				}
			}
			else if(type == OCTREE)
			{
				for(int i=0;i<size;i++)
				{
					OctreeContext* c = reinterpret_cast<OctreeContext*>(l[i]);
					ar & *c;
				}
			}
		}

		template<class Archive>
		void load(Archive& ar, const unsigned version) 
		{
			int size;
    			ar & pRoot & pLeaf & blkStart & pSibling & siblingDesc & type & size & pSiblings & siblingDescs;
			if(type == VPTREE)
			{
				long int index, nodesTraversed, closestLabel;
				float tau;
				for(int i=0;i<size;i++)
				{
					VptreeContext lc;
					//ar & index & nodesTraversed & tau & closestLabel;
					//VptreeContext* c = new VptreeContext(index, nodesTraversed, tau, closestLabel);
					ar & lc;
					VptreeContext* c = new VptreeContext(lc);
				
					l.push_back(c);
				}
			}
			else if(type == KDTREE)
			{
				for(int i=0;i<size;i++)
				{
					KdtreeContext lc;
					ar & lc;
					KdtreeContext* c = new KdtreeContext(lc);
				
					l.push_back(c);
				}
			}
			else if(type == PCKDTREE)
			{
				for(int i=0;i<size;i++)
				{
					PCKdtreeContext lc;
					ar & lc;
					PCKdtreeContext* c = new PCKdtreeContext(lc);
				
					l.push_back(c);
				}
			}
			else if(type == OCTREE)
			{
				for(int i=0;i<size;i++)
				{
					OctreeContext lc;
					ar & lc;
					OctreeContext* c = new OctreeContext(lc);
				
					l.push_back(c);
				}
			}
		}

		BOOST_SERIALIZATION_SPLIT_MEMBER();

};

typedef struct MsgUpdatePLeaf_Oct{
friend class boost::serialization::access;
std::vector<char> cell;
std::vector<int> descs;
std::vector<long int> children;
long int label;
long int numVertices;
Vec cofm;
float mass;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & cell & descs & children & label & numVertices & cofm & mass;
  }
}MsgUpdatePLeaf_Oct;

typedef struct MsgUpdatePLeaves_Oct{
friend class boost::serialization::access;
bool moreData;
int lastSent;
std::vector<MsgUpdatePLeaf_Oct> vPLeaves;
MsgUpdatePLeaves_Oct():moreData(false){}
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & vPLeaves & moreData & lastSent;
  }
}MsgUpdatePLeaves_Oct;

typedef struct MsgReplicateReq{
friend class boost::serialization::access;
std::vector<long int> pRoot;
long int pLeafLabel;
MsgReplicateReq(){}
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & pRoot & pLeafLabel;
  }
}MsgReplicateReq;

#endif
