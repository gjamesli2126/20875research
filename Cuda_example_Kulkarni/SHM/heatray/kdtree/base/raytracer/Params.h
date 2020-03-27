/*
 * Params.h
 *
 *  Created on: Oct 27, 2012
 *      Author: yjo
 */

#ifndef PARAMS_H_
#define PARAMS_H_

enum Params {
	ModelFile = 0,
	PhotonsPerLight,
	LightPower,
	ScreenWidth,
	ScreenHeight,
	NumParams
};

inline void checkParams(int argc) {
	if (argc != NumParams) {
		cout << "usage: ./heat_ray [model file] [photons per light] [light power] [width] [height]" << endl;
		exit(0);
	}
}

#endif /* PARAMS_H_ */
