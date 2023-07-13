
#pragma once
#include <algorithm>
#include <fstream>

#include <limits>
#include <cmath>

#include <vector>
#include "DataTypes.h"
#include "Eigen.h"
#include "VirtualSensor.h"

#ifndef MINF
#define MINF -std::numeric_limits<double>::infinity()
#endif

struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Position stored as 4 floats (4th component is supposed to be 1.0)
    Vector4d position;
    // Color stored as 4 unsigned char
    Vector4uc color;
};

struct Triangle {
    unsigned int idx0;
    unsigned int idx1;
    unsigned int idx2;

    Triangle() : idx0{ 0 }, idx1{ 0 }, idx2{ 0 } {}

    Triangle(unsigned int _idx0, unsigned int _idx1, unsigned int _idx2) :
        idx0(_idx0), idx1(_idx1), idx2(_idx2) {}
};

struct CameraParametersPyramid {
    int imageWidth, imageHeight;
    float fovX, fovY;
    float cX, cY;

    /**
     * Returns camera parameters for a specified pyramid level; each level corresponds to a scaling of pow(.5, level)
     * @param level The pyramid level to get the parameters for with 0 being the non-scaled version,
     * higher levels correspond to smaller spatial size
     * @return A CameraParameters structure containing the scaled values
     */
    CameraParametersPyramid level(const size_t level) const
    {
        if (level == 0) return *this;

        const float scale_factor = powf(0.5f, static_cast<float>(level));
        return CameraParametersPyramid{ imageWidth >> level, imageHeight >> level,
                                  fovX * scale_factor, fovY * scale_factor,
                                  (cX + 0.5f) * scale_factor - 0.5f,
                                  (cY + 0.5f) * scale_factor - 0.5f };
    }
};

class SurfaceMeasurement {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SurfaceMeasurement(VirtualSensor& sensor, CameraParametersPyramid cameraParams, const float* depthMap, const BYTE* colorMap, const Eigen::Matrix3d& depthIntrinsics, const Eigen::Matrix3d& colorIntrinsics,
            const Eigen::Matrix4d& d2cExtrinsics,
            const unsigned int width, const unsigned int height, int num_levels, int bfilter_kernel_size = 5, float bfilter_color_sigma = 1.f,
            float bfilter_spatial_sigma = 1.f, double maxDistance = 2, unsigned downsampleFactor = 1);

    void applyGlobalPose(Eigen::Matrix4d& estimated_pose);

    const std::vector<Eigen::Vector3d>& getVertexMap() const;

    const std::vector<Eigen::Vector3d>& getNormalMap() const;

    const std::vector<Eigen::Vector3d>& getGlobalVertexMap() const;

    void setGlobalVertexMap(const Eigen::Vector3d& point, size_t u, size_t v, CameraParametersPyramid cameraParams);

    const std::vector<Eigen::Vector3d>& getGlobalNormalMap() const;

    void setGlobalNormalMap(const Eigen::Vector3d& normal, size_t u, size_t v, CameraParametersPyramid cameraParams);

    void computeGlobalNormalMap(CameraParametersPyramid cameraParams);

    const Eigen::Matrix4d& getGlobalPose() const;

    void setGlobalPose(const Eigen::Matrix4d& pose);

    void setDepthMapPyramid(const cv::Mat& depthMap);

    const cv::Mat& getDepthMapPyramid() const;

    const std::vector<Vector4uc>& getColorMap() const;

    void setColor(const Vector4uc& color, size_t u, size_t v, CameraParametersPyramid cameraParams);

    bool contains(const Eigen::Vector2i& point, CameraParametersPyramid cameraParams);

    Eigen::Vector3d projectIntoCamera(const Eigen::Vector3d& globalCoord);

    Eigen::Vector2i projectOntoDepthPlane(const Eigen::Vector3d& cameraCoord, CameraParametersPyramid cameraParams);
    Eigen::Vector2i projectOntoColorPlane(const Eigen::Vector3d& cameraCoord, CameraParametersPyramid cameraParams);

    Eigen::Matrix3d intrinsicPyramid(CameraParametersPyramid cameraParams);

    unsigned int addFace(unsigned int idx0, unsigned int idx1, unsigned int idx2) {
        unsigned int fId = (unsigned int)triangles.size();
        Triangle triangle(idx0, idx1, idx2);
        triangles.push_back(triangle);
        return fId;
    }

private:
    Eigen::Vector2i projectOntoPlane(const Eigen::Vector3d& cameraCoord, Eigen::Matrix3d& intrinsics);

    void computeColorMap(const BYTE* colorMap, CameraParametersPyramid cameraParams, int numLevels);

    void computeTriangles(std::vector<Eigen::Vector3d> rawVertexMap, CameraParametersPyramid cameraParams, float edgeThreshold);

    std::vector<Eigen::Vector3d> computeVertexMap(cv::Mat& depthMapConvert, CameraParametersPyramid cameraParams);

    std::vector<Eigen::Vector3d> computeNormalMap(std::vector<Eigen::Vector3d> vertexMap, CameraParametersPyramid cameraParams, double maxDistance);

    void filterMask(std::vector<Eigen::Vector3d> rawVertexMap, std::vector<Eigen::Vector3d> rawNormalMap, int downsampleFactor);
    std::vector<Eigen::Vector3d> transformPoints(std::vector<Eigen::Vector3d>& points, Eigen::Matrix4d& transformation);

    std::vector<Eigen::Vector3d> rotatePoints(std::vector<Eigen::Vector3d>& points, Eigen::Matrix3d& rotation);


    std::vector<Eigen::Vector3d> vertexMap;
    std::vector<Eigen::Vector3d> normalMap;

    const unsigned int m_width;
    const unsigned int m_height;

    std::vector<Eigen::Vector3d> globalVertexMap;
    std::vector<Eigen::Vector3d> globalNormalMap;
    Eigen::Matrix4d m_global_pose;
    Eigen::Matrix3d m_intrinsic_matrix;
    Eigen::Matrix3d m_color_intrinsic_matrix;
    Eigen::Matrix4d m_d2cExtrinsics;

    cv::Mat smoothedDepthMapConvert;

    cv::Mat depthPyramidLevel1;
    cv::Mat depthPyramidLevel2;
    cv::Mat depthPyramidLevel3;

    cv::Mat smoothedDepthPyramidLevel1;
    cv::Mat smoothedDepthPyramidLevel2;
    cv::Mat smoothedDepthPyramidLevel3;

    cv::Mat m_depth_map_pyramid;

    std::vector<double> m_depth_map;
    std::vector<Vector4uc> m_color_map;

    int bfilter_kernel_size{ 5 };
    float bfilter_color_sigma{ 1.f };
    float bfilter_spatial_sigma{ 1.f };

    // The number of pyramid levels to generate for each frame, including the original frame level
    int num_levels{ 3 };

    double m_maxDistance;

    std::vector<Triangle> triangles;
};
