#pragma once

#ifndef MARCHING_CUBES_H
#define MARCHING_CUBES_H

#include "MCTables.h"
#include "Volume.h"
#include "SurfaceMeasurement.h"

#include "Eigen.h"

struct MC_Triangle {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		Vector3d p[3];
};

struct MC_Gridcell {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		Vector3d p[8];
	double val[8];
};

struct MC_Gridcell_2 {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		Vector3d p[8];
	double val[8];
	Vector3d ev0[8];
};

class MarchingCubes
{

public:
	//Eigen::Vector3d interpolate(double isolevel, const Vector3d& p1, const Vector3d& p2, double f1, double f2);

	int polygonise(MC_Gridcell grid, double isolevel, MC_Triangle* triangles);

	bool processVolumeCell(Volume* vol, int x, int y, int z, double iso, SurfaceMeasurement* mesh);

	static void extractMesh(Volume& volume, std::string fileName);

};

#endif // MARCHING_CUBES_H