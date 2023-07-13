#include "Ray.h"

Ray::Ray(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction): origin{origin}, direction{direction}
{
	//this->invDirection= Eigen::Vector3d(1.0) / this->direction;
	this->invDirection[0] = (1.0) / this->direction[0];
	this->invDirection[1] = (1.0) / this->direction[1];
	this->invDirection[2] = (1.0) / this->direction[2];
	
	this->sign[0] = (this->invDirection.x() < 0);
	this->sign[1] = (this->invDirection.y() < 0);
	this->sign[2] = (this->invDirection.z() < 0);
}
