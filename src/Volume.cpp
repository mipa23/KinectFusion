#include "Volume.h"

Volume::Volume(const Eigen::Vector3d origin, const Eigen::Vector3i volumeSize, const double voxelScale)
    : _volumeSize(volumeSize),
    _voxelScale(voxelScale),
    _volumeRange(volumeSize.cast<double>()* voxelScale),
    _origin(origin),
    _maxPoint(voxelScale* volumeSize.cast<double>())
{
    _voxelData.resize(volumeSize.x() * volumeSize.y() * volumeSize.z(), Voxel());

    Eigen::Vector3d half_voxelSize(voxelScale / 2, voxelScale / 2, voxelScale / 2);
    bounds[0] = _origin + half_voxelSize;
    bounds[1] = _maxPoint - half_voxelSize;
}

bool Volume::intersects(const Ray& r, float& entry_distance) const {

    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin = (bounds[r.sign[0]].x() - r.origin.x()) * r.invDirection.x();
    tmax = (bounds[1 - r.sign[0]].x() - r.origin.x()) * r.invDirection.x();
    tymin = (bounds[r.sign[1]].y() - r.origin.y()) * r.invDirection.y();
    tymax = (bounds[1 - r.sign[1]].y() - r.origin.y()) * r.invDirection.y();

    if ((tmin > tymax) || (tymin > tmax))
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    tzmin = (bounds[r.sign[2]].z() - r.origin.z()) * r.invDirection.z();
    tzmax = (bounds[1 - r.sign[2]].z() - r.origin.z()) * r.invDirection.z();

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    entry_distance = tmin;

    return true;
}

const Eigen::Vector3d& Volume::getOrigin() const {
    return _origin;
}


std::vector<Voxel>& Volume::getVoxelData() {
    return _voxelData;
}

const Eigen::Vector3i& Volume::getVolumeSize() const {
    return _volumeSize;
}

float Volume::getVoxelScale() const {
    return _voxelScale;
}

bool Volume::contains(const Eigen::Vector3d global_point) {
    Eigen::Vector3d volumeCoord = (global_point - _origin);

    return !(volumeCoord.x() < 0 || volumeCoord.x() >= _volumeRange.x() || volumeCoord.y() < 0 ||
        volumeCoord.y() >= _volumeRange.y() ||
        volumeCoord.z() < 0 || volumeCoord.z() >= _volumeRange.z());

}

Eigen::Vector3d Volume::getGlobalCoordinate(int voxelIdx_x, int voxelIdx_y, int voxelIdx_z) {
    const Eigen::Vector3d position((static_cast<double>(voxelIdx_x) + 0.5) * _voxelScale,
        (static_cast<double>(voxelIdx_y) + 0.5) * _voxelScale,
        (static_cast<double>(voxelIdx_z) + 0.5) * _voxelScale);
    return position + _origin;
}

double Volume::getTSDF(Eigen::Vector3d global) {
    Eigen::Vector3d shifted = (global - _origin) / _voxelScale;
    Eigen::Vector3i currentPosition;
    currentPosition.x() = int(shifted.x());
    currentPosition.y() = int(shifted.y());
    currentPosition.z() = int(shifted.z());

    Voxel voxelData = getVoxelData()[currentPosition.x() + currentPosition.y() * _volumeSize.x()
        + currentPosition.z() * _volumeSize.x() * _volumeSize.y()];
    return voxelData.tsdf;
}

double Volume::getTSDF(int x, int y, int z) {
    return getVoxelData()[x + y * _volumeSize.x() + z * _volumeSize.x() * _volumeSize.y()].tsdf;
}

Vector4uc Volume::getColor(Eigen::Vector3d global) {
    Eigen::Vector3d shifted = (global - _origin) / _voxelScale;
    Eigen::Vector3i currentPosition;
    currentPosition.x() = int(shifted.x());
    currentPosition.y() = int(shifted.y());
    currentPosition.z() = int(shifted.z());

    Voxel voxelData = getVoxelData()[currentPosition.x() + currentPosition.y() * _volumeSize.x()
        + currentPosition.z() * _volumeSize.x() * _volumeSize.y()];
    return voxelData.color;
}

Eigen::Vector3d Volume::getTSDFGrad(Eigen::Vector3d global) {
    Eigen::Vector3d shifted = (global - _origin) / _voxelScale;
    Eigen::Vector3i currentPosition;
    currentPosition.x() = int(shifted.x());
    currentPosition.y() = int(shifted.y());
    currentPosition.z() = int(shifted.z());

    // TODO: double check

    double tsdf_x0 = getVoxelData()[(currentPosition.x() - 1) + currentPosition.y() * _volumeSize.x()
        + currentPosition.z() * _volumeSize.x() * _volumeSize.y()].tsdf;
    double tsdf_x1 = getVoxelData()[(currentPosition.x() + 1) + currentPosition.y() * _volumeSize.x()
        + currentPosition.z() * _volumeSize.x() * _volumeSize.y()].tsdf;
    double tsdf_y0 = getVoxelData()[currentPosition.x() + (currentPosition.y() - 1) * _volumeSize.x()
        + currentPosition.z() * _volumeSize.x() * _volumeSize.y()].tsdf;
    double tsdf_y1 = getVoxelData()[currentPosition.x() + (currentPosition.y() + 1) * _volumeSize.x()
        + currentPosition.z() * _volumeSize.x() * _volumeSize.y()].tsdf;
    double tsdf_z0 = getVoxelData()[currentPosition.x() + currentPosition.y() * _volumeSize.x()
        + (currentPosition.z() - 1) * _volumeSize.x() * _volumeSize.y()].tsdf;
    double tsdf_z1 = getVoxelData()[currentPosition.x() + currentPosition.y() * _volumeSize.x()
        + (currentPosition.z() + 1) * _volumeSize.x() * _volumeSize.y()].tsdf;
    return Eigen::Vector3d(tsdf_x1 - tsdf_x0, tsdf_y1 - tsdf_y0, tsdf_z1 - tsdf_z0) / (_voxelScale * 2);
}
