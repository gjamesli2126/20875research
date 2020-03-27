
#ifndef APP_INTERSTATE_H_
#define APP_INTERSTATE_H_

// extra state which needs to be saved in pro/epilogue of intermediary methods
// is not used during recursive traversal, so does not need to be in Block
// Construction/destruction code is in ../../interstate.c
struct IntermediaryState {
	bool castRay_ret;
};

#endif /* APP_INTERSTATE_H_ */
