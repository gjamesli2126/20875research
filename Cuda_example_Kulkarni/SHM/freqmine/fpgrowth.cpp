/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include"stdio.h"
#include <algorithm>
#include <cassert>
#include <utility>

#include "fptree.hpp"
int dbgCnt=0;

typedef struct decreasing_order_comparator {
	bool operator() (const std::pair<unsigned, Item>& lhs, const std::pair<unsigned, Item>& rhs) const {
	    return (lhs.first > rhs.first) || (!(lhs.first > rhs.first) && lhs.second < rhs.second);
	}
}decreasing_order_comparator;

class IsSameItem{
    public:
    const Item _item;
    IsSameItem(const Item&  item) : _item(item) {};
    bool operator()(const FPNode* node) const {
        return (node->item.compare(_item)==0)?true:false;
    }
};

class CompareNodeLinks{
    public:
    bool operator () (FPNode* const&  lhs, FPNode* const& rhs) const {
        return lhs->height < rhs->height;
    }
};

FPNode::FPNode(const Item& item, FPNode* parent) :
    //item( item ), frequency( 1 ), node_link(NULL), parent( parent ), children(), height(0)
    item( item ), frequency( 1 ), parent( parent ), children(), height(0)
{
#ifdef METRICS
	numberOfVertices++;
#endif
}

void FPTree::DestroyTree(FPNode* node)
{
	std::vector<FPNode*>::iterator childIter=node->children.begin();
	for(;childIter!=node->children.end();childIter++)
	{
		DestroyTree(*childIter);
	}
	delete node;
}
FPTree::~FPTree()
{
	//std::map<Item, std::vector<FPNode*> >::iterator	iter=header_table.begin();
	//for(;iter!=header_table.end();iter++)
	//	iter->second.erase((iter->second).begin(),(iter->second).end());
	DestroyTree(root);
}


FPTree::FPTree(const std::vector<Transaction>& transactions, unsigned minimum_support_treshold) :
    root(new FPNode( Item(), NULL )), header_table(), minimum_support_treshold( minimum_support_treshold )
{
	// scan the transactions counting the frequence of each item
	std::vector<Transaction>::const_iterator transIter = transactions.begin();
	std::vector<Item>::const_iterator itemIter;
	std::map<Item, unsigned> frequency_by_item;
    	for (;transIter!=transactions.end();transIter++)
	{
		itemIter=transIter->begin();
		for (;itemIter !=transIter->end();itemIter++) 
		    ++frequency_by_item[*itemIter];
	}
    
	std::map<Item, unsigned>::iterator suppIter=frequency_by_item.begin();
	// keep only items which have a frequency greater or equal than the minimum support treshold
	for (;suppIter!=frequency_by_item.end();) 
	{
		const unsigned item_frequency = (*suppIter).second;
		if ( item_frequency < minimum_support_treshold ) 
			frequency_by_item.erase(suppIter++); 
		else
			++suppIter;
	}
    
	// order items by decreasing frequency
	std::set<std::pair<unsigned, Item>, decreasing_order_comparator> items_ordered_by_frequency;
	for (suppIter=frequency_by_item.begin();suppIter!=frequency_by_item.end(); suppIter++) 
	{
		const Item& item = suppIter->first;
		const unsigned frequency = suppIter->second;
		items_ordered_by_frequency.insert(std::make_pair(frequency, item));
    	}

    // start tree construction
    
    // scan the transactions again
    for (transIter=transactions.begin();transIter!=transactions.end();transIter++)
    {
	FPNode* curr_fpnode = root;

        // select and sort the frequent items in transaction according to the order of items_ordered_by_frequency
	std::set<std::pair<unsigned, Item>, decreasing_order_comparator>::iterator freqIter=items_ordered_by_frequency.begin();
        for(freqIter=items_ordered_by_frequency.begin();freqIter!=items_ordered_by_frequency.end();freqIter++) 
	{
            const Item& item = freqIter->second;
            // check if item is contained in the current transaction
            if (std::find(transIter->begin(), transIter->end(), item ) != transIter->end() ) 
	    {
                // insert item in the tree
            
                // check if curr_fpnode has a child curr_fpnode_child such that curr_fpnode_child.item = item
                std::vector<FPNode*>::iterator it = std::find_if(curr_fpnode->children.begin(), curr_fpnode->children.end(), IsSameItem(item));
                if (it == curr_fpnode->children.end() ) {
                    // the child doesn't exist, create a new node
                    FPNode* curr_fpnode_new_child = new FPNode( item, curr_fpnode);
                    
		    curr_fpnode_new_child->height=curr_fpnode->height+1;
			#ifdef METRICS
			if(beginTrackingVertices)
			{
				if(curr_fpnode_new_child->height == splice_depth-1)
					subtrees.push_back(curr_fpnode_new_child);
			}	
			#endif
                    // add the new node to the tree
                    curr_fpnode->children.push_back( curr_fpnode_new_child );
                    
                    // update the node-link structure
                    if ( header_table.count( curr_fpnode_new_child->item ) ) {
                        /*FPNode* prev_fpnode = header_table[curr_fpnode_new_child->item];
                        while ( prev_fpnode->node_link ) { prev_fpnode = prev_fpnode->node_link; }
                        prev_fpnode->node_link = curr_fpnode_new_child;*/
			header_table[curr_fpnode_new_child->item].push_back(curr_fpnode_new_child);
                    }
                    else {
                        //header_table[curr_fpnode_new_child->item] = curr_fpnode_new_child;
			header_table[curr_fpnode_new_child->item].push_back(curr_fpnode_new_child);
                    }
                    
                    // advance to the next node of the current transaction
                    curr_fpnode = curr_fpnode_new_child;
                }
                else {
                    // the child exist, increment its frequency
                    FPNode* curr_fpnode_child = *it;
                    ++(curr_fpnode_child->frequency);
                    
                    // advance to the next node of the current transaction
                    curr_fpnode = curr_fpnode_child;
                }
            }
        }
    }
	
#ifdef METRICS
    if(beginTrackingVertices)
	beginTrackingVertices = false;
#endif
    /*std::map<Item, std::vector<FPNode*> >::iterator iIter=header_table.begin();
    for(;iIter!=header_table.end();iIter++)
   	 std::sort((iIter->second).begin(),(iIter->second).end(),CompareNodeLinks());*/
}

bool FPTree::empty() const {
    assert( root );
    return root->children.size() == 0;
}


bool contains_single_path(const FPNode* fpnode) {
    assert( fpnode );
    if ( fpnode->children.size() == 0 ) { return true; }
    if ( fpnode->children.size() > 1 ) { return false; }
    return contains_single_path( fpnode->children.front() );
}
bool contains_single_path(FPTree* fptree) {
    if ( fptree->empty() ) { return true; }
    return contains_single_path( fptree->root );
}
   
std::vector<std::string> string_split(const char *str, char c = ' ')
{
    std::vector<std::string> result;

    do
    {
        const char *begin = str;

        while(*str != c && *str)
            str++;

        result.push_back(std::string(begin, str));
    } while (0 != *str++);

    return result;
}
                       
void fptree_growth(FPTree* fptree, unsigned DOR, unsigned spliceDepth, const Item& prefix, TreesPendingMining& treesToBeMined, std::set<Pattern>& multi_path_patterns) {
//std::set<Pattern> fptree_growth(FPTree* fptree, unsigned DOR, unsigned spliceDepth, const Item& prefix, TreesPendingMining& treesToBeMined) {
    if ( fptree->empty() || (DOR == spliceDepth) ) 
    { 
	return; 
    }
    //if ( fptree->empty() || (DOR == spliceDepth) ) {std::set<Pattern> emptyPattern; return emptyPattern; }
    
    if ( contains_single_path( fptree ) ) {
        // generate all possible combinations of the items in the tree
        
        std::set<Pattern> single_path_patterns;
	
	std::vector<Item> prefixTokens = string_split(prefix.c_str(),' ');
        std::vector<Item>::iterator tIter=prefixTokens.begin();
	std::set<Item> curPrefixSet;
	for(;tIter!=prefixTokens.end();tIter++)
		curPrefixSet.insert(*tIter);

        // for each node in the tree
        assert( fptree->root->children.size() == 1 );
        FPNode* curr_fpnode = fptree->root->children.front();
        while ( curr_fpnode ) {
            const Item& curr_fpnode_item = curr_fpnode->item;
            const unsigned curr_fpnode_frequency = curr_fpnode->frequency;
            
            // add a pattern formed only by the item of the current node
            std::set<Item> tmpItem;
		tmpItem.insert(curPrefixSet.begin(),curPrefixSet.end());
		tmpItem.insert(curr_fpnode_item);
            Pattern new_pattern(std::make_pair(tmpItem, curr_fpnode_frequency));
            single_path_patterns.insert( new_pattern );
            
            // create a new pattern by adding the item of the current node to each pattern generated until now
            std::set<Pattern>::iterator pattIter=single_path_patterns.begin();

            for (pattIter=single_path_patterns.begin();pattIter!=single_path_patterns.end();pattIter++) 
	    {
                Pattern new_pattern;
                new_pattern.first.insert(pattIter->first.begin(),pattIter->first.end());
		new_pattern.first.insert(curPrefixSet.begin(),curPrefixSet.end());
                new_pattern.first.insert( curr_fpnode_item );
                assert( curr_fpnode_frequency <= pattIter->second);
                new_pattern.second = curr_fpnode_frequency;
                
                single_path_patterns.insert( new_pattern );
            }

            // advance to the next node until the end of the tree
            assert( curr_fpnode->children.size() <= 1 );
            if ( curr_fpnode->children.size() == 1 ) { curr_fpnode = curr_fpnode->children.front(); }
            else { curr_fpnode = NULL; }
        }
        
        //return single_path_patterns;
	multi_path_patterns.insert(single_path_patterns.begin(),single_path_patterns.end());
	return;
    }
    else {
        // generate conditional fptrees for each different item in the fptree, then join the results
        //std::set<Pattern> multi_path_patterns;
        //std::map<Item, FPNode*>::const_iterator startIter=fptree->header_table.begin();
        std::map<Item, std::vector<FPNode*> >::const_iterator startIter=fptree->header_table.begin();

	int k=0;
        // for each item in the fptree
        for (int i=0;i<fptree->header_table.size();i++)
        {
		int j=0;
		//printf("\nitem %d:",k++);
		//while(j < BLOCK_SIZE)
		{
			const Item& curr_item = startIter->first;
			unsigned curr_item_frequency = 0;
            
		    // build the conditional fptree relative to the current item
		    
		    // start by generating the conditional pattern base
		    std::vector<TransformedPrefixPath> conditional_pattern_base;
		    
		    // for each path in the header_table (relative to the current item)
		    //FPNode* path_starting_fpnode = startIter->second;
		    std::vector<FPNode*>::const_iterator path_starting_fpnode = (startIter->second).begin();
		    //while ( path_starting_fpnode) 
		    while ( path_starting_fpnode != (startIter->second).end())
		    {

			// construct the transformed prefix path
			
			// each item in th transformed prefix path has the same frequency (the frequency of path_starting_fpnode)
			const unsigned path_starting_fpnode_frequency = (*path_starting_fpnode)->frequency;
			//const unsigned path_starting_fpnode_frequency = (path_starting_fpnode)->frequency;

			//FPNode* curr_path_fpnode = (path_starting_fpnode)->parent;
			FPNode* curr_path_fpnode = (*path_starting_fpnode)->parent;
			// check if curr_path_fpnode is already the root of the fptree
			if ( curr_path_fpnode->parent ) {
			    // the path has at least one node (excluding the starting node and the root)
			    TransformedPrefixPath transformed_prefix_path;
			    transformed_prefix_path.second=path_starting_fpnode_frequency;
			    
			    while ( curr_path_fpnode->parent ) {
				assert( curr_path_fpnode->frequency >= path_starting_fpnode_frequency );
				transformed_prefix_path.first.push_back( curr_path_fpnode->item );
				
				// advance to the next node in the path
				curr_path_fpnode = curr_path_fpnode->parent;
			    }
			    
			    conditional_pattern_base.push_back( transformed_prefix_path );
			}
			
			// advance to the next path
			//printf("%d ", (path_starting_fpnode)->height);
			//path_starting_fpnode = path_starting_fpnode->node_link;
			//printf("%d ", (*path_starting_fpnode)->height);
			curr_item_frequency += (*path_starting_fpnode)->frequency;
			path_starting_fpnode++;
		    }
		
		    // the first pattern is made only by the current item
		    // compute the frequency of this pattern by summing the frequency of the nodes which have the same item (follow the node links)
		    {	
			std::set<Item> tmpItem;
		    	tmpItem.insert(curr_item);
		    	tmpItem.insert(prefix);
			    Pattern pattern(std::make_pair(tmpItem, curr_item_frequency));
			    multi_path_patterns.insert(pattern);
		    }

		#if 1
		    // generate the transactions that represent the conditional pattern base
		    std::vector<TransformedPrefixPath>::iterator cbpIter=conditional_pattern_base.begin();

		    std::vector<Transaction> conditional_fptree_transactions;
		    for (cbpIter=conditional_pattern_base.begin();cbpIter!=conditional_pattern_base.end();cbpIter++) 
		    {
			const std::vector<Item>& transformed_prefix_path_items = cbpIter->first;
			const unsigned transformed_prefix_path_items_frequency = cbpIter->second;
			
			Transaction transaction;
			std::vector<Item>::const_iterator tppIter = transformed_prefix_path_items.begin();
			for (tppIter = transformed_prefix_path_items.begin();tppIter!=transformed_prefix_path_items.end();tppIter++) { transaction.push_back( *tppIter); }
			
			// add the same transaction transformed_prefix_path_items_frequency times
			for ( int i = 0; i < transformed_prefix_path_items_frequency; ++i ) { conditional_fptree_transactions.push_back( transaction ); }
		    }
		   
		    if(conditional_fptree_transactions.size() > 0)
		    {
			Item prefixItem(prefix);
			prefixItem.append(" ");
			prefixItem.append(curr_item);
			//append prefix with curr_item
			// build the conditional fptree relative to the current item with the transactions just generated
		    	FPTree* conditional_fptree= new FPTree( conditional_fptree_transactions, fptree->minimum_support_treshold );
			if((DOR+1) == spliceDepth) 
			{
				treesToBeMined.push_back(std::make_pair(conditional_fptree,prefixItem));
			}
			else
			{
				// call recursively fptree_growth on the conditional fptree (empty fptree: no patterns)
				fptree_growth( conditional_fptree, DOR+1, spliceDepth, prefixItem, treesToBeMined, multi_path_patterns);
				//std::set<Pattern> conditional_patterns = fptree_growth( conditional_fptree, DOR+1, spliceDepth, prefixItem, treesToBeMined);
				delete conditional_fptree;
				
		    
			    /*// construct patterns relative to the current item using both the current item and the conditional patterns
			    std::set<Pattern> curr_item_patterns;
			    
			    std::set<Pattern>::iterator cpIter = conditional_patterns.begin(); 
			    for (cpIter = conditional_patterns.begin();cpIter !=conditional_patterns.end();cpIter++ ) {
				Pattern new_pattern;
				new_pattern.first.insert(cpIter->first.begin(),cpIter->first.end());
				new_pattern.first.insert( curr_item );
				assert( curr_item_frequency >= cpIter->second );
				new_pattern.second = cpIter->second;

				curr_item_patterns.insert(new_pattern);
			    }
			    
			    // join the patterns generated by the current item with all the other items of the fptree
			    multi_path_patterns.insert( curr_item_patterns.begin(), curr_item_patterns.end() );*/
			}
		    }
		#endif
		    j++;
#ifdef METRICS
		numberOfTraversals +=startIter->second.size();
		/*if(DOR == 0)
		{
			printf("%d set of traversals gave rise to %d vertex visits\n",startIter->second.size(), numberOfVertices);
			numberOfVertices=0;
		}*/
#endif

		    startIter++;
		    if(startIter == fptree->header_table.end())
			break;
        	} //while (j<BLOCKSIZE ...

	}
        //return multi_path_patterns;
    }
}
