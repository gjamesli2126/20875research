/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef _BH_BLOCK_INFO_H
#define _BH_BLOCK_INFO_H


#define BLOCK_INFO_SET_TYPE(bi, type) ((bi)->b0 = (((type << 30) & 0xC0000000) | ((bi)->b0 & 0x3FFFFFFF)))
#define BLOCK_INFO_SET_SIZE(bi, size) ((bi)->b0 = (((bi)->b0 & 0xC0000000) | (size & 0x3FFFFFFF)))
#define BLOCK_INFO_SET_TREE_INDEX(bi, index) (bi)->b1 = (index)
#define BLOCK_INFO_SET_DEPTH(bi, depth) (bi)->b2 = (depth)

#define BLOCK_INFO_TYPE(bi) ((block_type)(((bi)->b0 >> 30) & 0x3))
#define BLOCK_INFO_SIZE(bi) ((unsigned int)(((bi)->b0 & 0x3FFFFFFF)))
#define BLOCK_INFO_TREE_INDEX(bi) ((int)(bi)->b1)
#define BLOCK_INFO_DEPTH(bi) (bi)->b2

// block type is not required for Barnes-Hut but we will want to
// store it just incase we start mixing block configurations
typedef enum _block_type {
	btLinear = 1,
	btTriangle = 2,
	btLeaf = 3
} block_type;

typedef struct _block_info {
	unsigned int b0; // bits (31..29) type, (28..0) size
	int b1; // bits (30..0) index of root node in tree, negative means unset
	unsigned int b2; // bits (31..0) depth of root node of block
} block_info_t;


#endif
