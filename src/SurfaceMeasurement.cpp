
#include "SurfaceMeasurement.h"
#include "Mesh.h"

#include <iostream>


SurfaceMeasurement::SurfaceMeasurement(VirtualSensor& sensor, CameraParametersPyramid cameraParams, const float* depthMap, const BYTE* colorMap,
    const Eigen::Matrix3d& depthIntrinsics, const Eigen::Matrix3d& colorIntrinsics,
    const Eigen::Matrix4d& d2cExtrinsics,
    const unsigned int width, const unsigned int height, int numLevels,
    int kernelSize, float colorSigma,
    float spatialSigma, double maxDistance, unsigned downsampleFactor)
    : m_width(width), m_height(height), m_intrinsic_matrix(depthIntrinsics), m_color_intrinsic_matrix(colorIntrinsics),
    m_d2cExtrinsics(d2cExtrinsics), m_maxDistance(maxDistance)
{

    cv::Mat depthMapConvert = cv::Mat(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_32FC1, sensor.getDepth());

    cv::resize(depthMapConvert, depthPyramidLevel1, cv::Size(sensor.getDepthImageWidth() / 2, sensor.getDepthImageHeight() / 2));
    cv::resize(depthPyramidLevel1, depthPyramidLevel2, cv::Size(sensor.getDepthImageWidth() / 2, sensor.getDepthImageHeight() / 2));
    cv::resize(depthPyramidLevel2, depthPyramidLevel3, cv::Size(sensor.getDepthImageWidth() / 2, sensor.getDepthImageHeight() / 2));

    cv::bilateralFilter(depthMapConvert, // source
        smoothedDepthMapConvert, // destination
        kernelSize,
        colorSigma,
        spatialSigma,
        cv::BORDER_DEFAULT);

    cv::bilateralFilter(depthPyramidLevel1, // source
        smoothedDepthPyramidLevel1, // destination
        kernelSize,
        colorSigma,
        spatialSigma,
        cv::BORDER_DEFAULT);

    cv::bilateralFilter(depthPyramidLevel2, // source
        smoothedDepthPyramidLevel2, // destination
        kernelSize,
        colorSigma,
        spatialSigma,
        cv::BORDER_DEFAULT);

    cv::bilateralFilter(depthPyramidLevel3, // source
        smoothedDepthPyramidLevel3, // destination
        kernelSize,
        colorSigma,
        spatialSigma,
        cv::BORDER_DEFAULT);

    setDepthMapPyramid(depthMapConvert);

    std::vector<Eigen::Vector3d> rawVertexMap = computeVertexMap(depthMapConvert, cameraParams.level(numLevels));

    std::vector<Eigen::Vector3d> rawNormalMap = computeNormalMap(rawVertexMap, cameraParams.level(numLevels), maxDistance);
    filterMask(rawVertexMap, rawNormalMap, downsampleFactor);
    setGlobalPose(Eigen::Matrix4d::Identity());

    computeColorMap(colorMap, cameraParams.level(numLevels), numLevels);
}

std::vector<Eigen::Vector3d> SurfaceMeasurement::computeVertexMap(cv::Mat& depthMapConvert, CameraParametersPyramid cameraParams) {

    // Back-project the pixel depths into the camera space.
    std::vector<Vector3d> rawVertexMap(cameraParams.imageWidth * cameraParams.imageHeight);

    for (int v = 0; v < cameraParams.imageHeight; ++v) {
        // For every pixel in a row.
        for (int u = 0; u < cameraParams.imageWidth; ++u) {
            unsigned int idx = v * cameraParams.imageWidth + u; // linearized index
            float depth = depthMapConvert.at<float>(v, u);
            if (depth == MINF) {
                rawVertexMap[idx] = Vector3d(MINF, MINF, MINF);
            }
            else {
                // Back-projection to camera space.
                rawVertexMap[idx] = Vector3d((u - cameraParams.cX) / cameraParams.fovX * depth, (v - cameraParams.cY) / cameraParams.fovY * depth, depth);
            }
        }
    }

    return rawVertexMap;
}

std::vector<Eigen::Vector3d> SurfaceMeasurement::computeNormalMap(std::vector<Eigen::Vector3d> vertexMap, CameraParametersPyramid cameraParams, double maxDistance) {

    // We need to compute derivatives and then the normalized normal vector (for valid pixels).
    std::vector<Eigen::Vector3d> rawNormalMap(cameraParams.imageWidth * cameraParams.imageHeight);
    const float maxDistanceHalved = maxDistance / 2.f;

    for (int v = 1; v < cameraParams.imageHeight - 1; ++v) {
        for (int u = 1; u < cameraParams.imageWidth - 1; ++u) {
            unsigned int idx = v * cameraParams.imageWidth + u; // linearized index

            const Eigen::Vector3d du = vertexMap[idx + 1] - vertexMap[idx - 1];
            const Eigen::Vector3d dv = vertexMap[idx + cameraParams.imageWidth] - vertexMap[idx - cameraParams.imageWidth];
            if (!du.allFinite() || !dv.allFinite() || du.norm() > maxDistanceHalved || dv.norm() > maxDistanceHalved) {
                rawNormalMap[idx] = Eigen::Vector3d(MINF, MINF, MINF);
                continue;
            }

            // Compute the normals using central differences. 
            rawNormalMap[idx] = du.cross(dv);
            rawNormalMap[idx].normalize();
        }
    }

    // We set invalid normals for border regions.
    for (int u = 0; u < cameraParams.imageWidth; ++u) {
        rawNormalMap[u] = Eigen::Vector3d(MINF, MINF, MINF);
        rawNormalMap[u + (cameraParams.imageHeight - 1) * cameraParams.imageWidth] = Eigen::Vector3d(MINF, MINF, MINF);
    }
    for (int v = 0; v < cameraParams.imageHeight; ++v) {
        rawNormalMap[v * cameraParams.imageWidth] = Eigen::Vector3d(MINF, MINF, MINF);
        rawNormalMap[(cameraParams.imageWidth - 1) + v * cameraParams.imageWidth] = Eigen::Vector3d(MINF, MINF, MINF);
    }

    return rawNormalMap;
}

void SurfaceMeasurement::filterMask(std::vector<Eigen::Vector3d> rawVertexMap, std::vector<Eigen::Vector3d> rawNormalMap, int downsampleFactor)
{
    // We filter out measurements where either point or normal is invalid.
    const unsigned nVertices = rawVertexMap.size();
    vertexMap.reserve(std::floor(float(nVertices) / downsampleFactor));
    normalMap.reserve(std::floor(float(nVertices) / downsampleFactor));

    for (int i = 0; i < nVertices; i = i + downsampleFactor) {
        const auto& vertex = rawVertexMap[i];
        const auto& normal = rawNormalMap[i];

        if ((vertex.allFinite() && normal.allFinite())) {
            vertexMap.push_back(vertex);
            normalMap.push_back(normal);
        }
        else {
            vertexMap.emplace_back(Eigen::Vector3d(MINF, MINF, MINF));
            normalMap.emplace_back(Eigen::Vector3d(MINF, MINF, MINF));
        }
    }
}

void SurfaceMeasurement::computeColorMap(const BYTE* colorMap, CameraParametersPyramid cameraParams, int numLevels) {
    const auto rotation = m_d2cExtrinsics.block(0, 0, 3, 3);
    const auto translation = m_d2cExtrinsics.block(0, 3, 3, 1);

    std::vector<Vector4uc> colors(cameraParams.imageWidth * cameraParams.imageHeight);
    for (size_t i = 0; i < cameraParams.imageWidth * cameraParams.imageHeight; i++)
        colors[i] = Vector4uc(colorMap[i * 4], colorMap[i * 4 + 1], colorMap[i * 4 + 2], colorMap[i * 4 + 3]);

    BYTE zero = (BYTE)255;

    m_color_map.reserve(vertexMap.size());
    for (size_t i = 0; i < vertexMap.size(); ++i) {
        Eigen::Vector2i coord = projectOntoColorPlane(rotation * vertexMap[i] + translation, cameraParams.level(numLevels));
        if (contains(coord, cameraParams.level(numLevels)))
            m_color_map.push_back(colors[coord.x() + coord.y() * cameraParams.imageWidth]);
        else
            m_color_map.push_back(Vector4uc(zero, zero, zero, zero));
    }
}

void SurfaceMeasurement::computeTriangles(std::vector<Eigen::Vector3d> rawVertexMap, CameraParametersPyramid cameraParams, float edgeThreshold) {
    
    Matrix4d depthExtrinsicsInv = m_d2cExtrinsics.inverse();
    
    std::vector<Eigen::Vector3d> globalRawVertexMap = transformPoints(rawVertexMap, depthExtrinsicsInv);

    // Compute inverse camera pose (mapping from camera CS to world CS).
    //Matrix4d cameraPoseInverse = cameraPose.inverse();
    
    triangles.reserve((cameraParams.imageHeight - 1) * (cameraParams.imageWidth - 1) * 2);

    // Compute triangles (faces).
    for (unsigned int i = 0; i < cameraParams.imageHeight - 1; i++) {
        for (unsigned int j = 0; j < cameraParams.imageWidth - 1; j++) {
            unsigned int i0 = i * cameraParams.imageWidth + j;
            unsigned int i1 = (i + 1) * cameraParams.imageWidth + j;
            unsigned int i2 = i * cameraParams.imageWidth + j + 1;
            unsigned int i3 = (i + 1) * cameraParams.imageWidth + j + 1;

            bool valid0 = globalRawVertexMap[i0].allFinite();
            bool valid1 = globalRawVertexMap[i1].allFinite();
            bool valid2 = globalRawVertexMap[i2].allFinite();
            bool valid3 = globalRawVertexMap[i3].allFinite();

            if (valid0 && valid1 && valid2) {
                float d0 = (globalRawVertexMap[i0] - globalRawVertexMap[i1]).norm();
                float d1 = (globalRawVertexMap[i0] - globalRawVertexMap[i2]).norm();
                float d2 = (globalRawVertexMap[i1] - globalRawVertexMap[i2]).norm();
                if (edgeThreshold > d0 && edgeThreshold > d1 && edgeThreshold > d2)
                    addFace(i0, i1, i2);
            }
            if (valid1 && valid2 && valid3) {
                float d0 = (globalRawVertexMap[i3] - globalRawVertexMap[i1]).norm();
                float d1 = (globalRawVertexMap[i3] - globalRawVertexMap[i2]).norm();
                float d2 = (globalRawVertexMap[i1] - globalRawVertexMap[i2]).norm();
                if (edgeThreshold > d0 && edgeThreshold > d1 && edgeThreshold > d2)
                    addFace(i1, i3, i2);
            }
        }
    }
}

Eigen::Matrix3d SurfaceMeasurement::intrinsicPyramid(CameraParametersPyramid cameraParams) {

    Eigen::Matrix3d intrinsics;
    intrinsics << cameraParams.fovX, 0.0f, cameraParams.cX, 0.0f, cameraParams.fovY, cameraParams.cY, 0.0f, 0.0f, 1.0f;

    return intrinsics;
}

Eigen::Vector3d SurfaceMeasurement::projectIntoCamera(const Eigen::Vector3d& globalCoord) {
    Eigen::Matrix4d pose_inverse = m_global_pose.inverse();
    const auto rotation_inv = pose_inverse.block(0, 0, 3, 3);
    const auto translation_inv = pose_inverse.block(0, 3, 3, 1);
    return rotation_inv * globalCoord + translation_inv;
}

bool SurfaceMeasurement::contains(const Eigen::Vector2i& img_coord, CameraParametersPyramid cameraParams) {
    return img_coord[0] < cameraParams.imageWidth && img_coord[1] < cameraParams.imageHeight && img_coord[0] >= 0 && img_coord[1] >= 0;
}

Eigen::Vector2i SurfaceMeasurement::projectOntoPlane(const Eigen::Vector3d& cameraCoord, Eigen::Matrix3d& intrinsics) {
    Eigen::Vector3d projected = (intrinsics * cameraCoord);
    if (projected[2] == 0) {
        return Eigen::Vector2i(MINF, MINF);
    }
    projected /= projected[2];
    return (Eigen::Vector2i((int)round(projected.x()), (int)round(projected.y())));
}
Eigen::Vector2i SurfaceMeasurement::projectOntoDepthPlane(const Eigen::Vector3d& cameraCoord, CameraParametersPyramid cameraParams) {
    m_intrinsic_matrix = intrinsicPyramid(cameraParams);

    return projectOntoPlane(cameraCoord, m_intrinsic_matrix);
}

Eigen::Vector2i SurfaceMeasurement::projectOntoColorPlane(const Eigen::Vector3d& cameraCoord, CameraParametersPyramid cameraParams) {
    m_color_intrinsic_matrix = intrinsicPyramid(cameraParams);

    return projectOntoPlane(cameraCoord, m_color_intrinsic_matrix);
}

void SurfaceMeasurement::computeGlobalNormalMap(CameraParametersPyramid cameraParams) {
    std::vector<Eigen::Vector3d> camera_points;
    for (const auto& global : globalVertexMap) {
        camera_points.emplace_back(projectIntoCamera(global));
    }
    globalNormalMap = computeNormalMap(camera_points, cameraParams, m_maxDistance);
}

void SurfaceMeasurement::applyGlobalPose(Eigen::Matrix4d& estimated_pose) {
    Eigen::Matrix3d rotation = estimated_pose.block(0, 0, 3, 3);

    globalVertexMap = transformPoints(vertexMap, estimated_pose);
    globalNormalMap = rotatePoints(normalMap, rotation);
}

std::vector<Eigen::Vector3d> SurfaceMeasurement::transformPoints(std::vector<Eigen::Vector3d>& points, Eigen::Matrix4d& transformation) {
    const Eigen::Matrix3d rotation = transformation.block(0, 0, 3, 3);
    const Eigen::Vector3d translation = transformation.block(0, 3, 3, 1);
    std::vector<Eigen::Vector3d> transformed(points.size());

    for (size_t idx = 0; idx < points.size(); ++idx) {
        if (points[idx].allFinite())
            transformed[idx] = rotation * points[idx] + translation;
        else
            transformed[idx] = (Eigen::Vector3d(MINF, MINF, MINF));
    }
    return transformed;
}

std::vector<Eigen::Vector3d> SurfaceMeasurement::rotatePoints(std::vector<Eigen::Vector3d>& points, Eigen::Matrix3d& rotation) {
    std::vector<Eigen::Vector3d> transformed(points.size());

    for (size_t idx = 0; idx < points.size(); ++idx) {
        if (points[idx].allFinite())
            transformed[idx] = rotation * points[idx];
        else
            transformed[idx] = (Eigen::Vector3d(MINF, MINF, MINF));
    }
    return transformed;
}

const std::vector<Eigen::Vector3d>& SurfaceMeasurement::getVertexMap() const {
    return vertexMap;
}

const std::vector<Eigen::Vector3d>& SurfaceMeasurement::getNormalMap() const {
    return normalMap;
}

const std::vector<Eigen::Vector3d>& SurfaceMeasurement::getGlobalNormalMap() const {
    return globalNormalMap;
}

void SurfaceMeasurement::setGlobalNormalMap(const Eigen::Vector3d& normal, size_t u, size_t v, CameraParametersPyramid cameraParams) {
    size_t idx = v * cameraParams.imageWidth + u;
    globalNormalMap[idx] = normal;
}

const std::vector<Eigen::Vector3d>& SurfaceMeasurement::getGlobalVertexMap() const {
    return globalVertexMap;
}

void SurfaceMeasurement::setGlobalVertexMap(const Eigen::Vector3d& point, size_t u, size_t v, CameraParametersPyramid cameraParams) {
    size_t idx = v * cameraParams.imageWidth + u;
    globalVertexMap[idx] = point;
}

const Eigen::Matrix4d& SurfaceMeasurement::getGlobalPose() const {
    return m_global_pose;
}

void SurfaceMeasurement::setGlobalPose(const Eigen::Matrix4d& pose) {
    m_global_pose = pose;
    applyGlobalPose(m_global_pose);
}

const std::vector<Vector4uc>& SurfaceMeasurement::getColorMap() const {
    return m_color_map;
}

void SurfaceMeasurement::setColor(const Vector4uc& color, size_t u, size_t v, CameraParametersPyramid cameraParams) {
    size_t idx = v * cameraParams.imageWidth + u;
    m_color_map[idx] = color;
}

void SurfaceMeasurement::setDepthMapPyramid(const cv::Mat& depthMap) {
    m_depth_map_pyramid = depthMap;
}

const cv::Mat& SurfaceMeasurement::getDepthMapPyramid() const {
    return m_depth_map_pyramid;
}