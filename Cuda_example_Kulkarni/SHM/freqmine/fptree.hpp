/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef FPTREE_HPP
#define FPTREE_HPP

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <utility>

#define SPLICE_DEPTH 1
#define SUPP_THRESHOLD 5
#define NUM_THREADS 5

typedef std::string Item;
typedef std::vector<Item> Transaction;
typedef std::pair<std::vector<Item>, unsigned> TransformedPrefixPath;
typedef std::pair<std::set<Item>, unsigned> Pattern;


struct FPNode {
    const Item item;
    unsigned frequency;
    //FPNode* node_link;
    FPNode* parent;
    std::vector<FPNode*> children;
    FPNode(const Item&, FPNode*);
    short int height;
};



struct FPTree {
    FPNode* root;
    std::map<Item, std::vector<FPNode*> > header_table;
    //std::map<Item, FPNode*> header_table;

    unsigned minimum_support_treshold;
    
    FPTree(const std::vector<Transaction>&, unsigned);
    ~FPTree();
    void DestroyTree(FPNode* node);
    
    bool empty() const;
};


typedef std::vector<std::pair<FPTree*,Item> > TreesPendingMining;
//std::set<Pattern> fptree_growth(FPTree*, unsigned DOR, unsigned spliceDepth, const Item& prefix, TreesPendingMining& ttm);
void fptree_growth(FPTree*, unsigned DOR, unsigned spliceDepth, const Item& prefix, TreesPendingMining& ttm, std::set<Pattern>& patterns);

#ifdef METRICS
extern int numberOfTraversals;
extern int numberOfVertices;
extern bool beginTrackingVertices;
extern int splice_depth;
extern std::vector<FPNode*> subtrees;
#endif


#endif  // FPTREE_HPP
