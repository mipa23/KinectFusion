#include "icp.h"
#include "iterator"

// Linear Solver : linear least-squares optimization of ICP
// This class serves the purpose of solving for the pose of a camera given the point correspondences
// Reference Paper : Linear least-squares optimization for point-to-plane icp surface registration by Kok-Lim Low

// Creates a linear system of equations fulfilling the constraints of point-to-plane error metric
// Solves the created linear system for the camera pose
void LinearSolver::solvePoint2Plane(const std::vector<Eigen::Vector3d>& sourcePoints,
    const std::vector<Eigen::Vector3d>& destPoints,
    const std::vector<Eigen::Vector3d> destNormals,
    const std::vector<std::pair<size_t, size_t>>& correspondence) {

    const size_t N = correspondence.size();
    Eigen::MatrixXd A(N, 6);
    Eigen::MatrixXd b(N, 1);

    for (size_t i = 0; i < correspondence.size(); ++i) {
        auto match = correspondence[i];

        Eigen::Vector3d d_i = destPoints[match.first];
        Eigen::Vector3d n_i = destNormals[match.first];
        Eigen::Vector3d s_i = sourcePoints[match.second];

        Eigen::Matrix<double, 6, 1> A_i;
        A_i << s_i.cross(n_i), n_i;
        A.row(i) = A_i;
        b(i) = n_i.dot(d_i) - n_i.dot(s_i);
    }

    Eigen::Matrix<double, 6, 1> x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    solution = x;
}

// Creates a linear system of equations fulfilling the constraints of point-to-point error metric
// Solves the created linear system for the camera pose
void LinearSolver::solvePoint2Point(const std::vector<Eigen::Vector3d>& sourcePoints,
    const std::vector<Eigen::Vector3d>& destPoints,
    const std::vector<std::pair<size_t, size_t>>& correspondence) {
    const size_t N = correspondence.size();
    Eigen::MatrixXd A(N * 3, 6);
    Eigen::MatrixXd b(N * 3, 1);
    for (size_t i = 0; i < correspondence.size(); ++i) {
        auto match = correspondence[i];

        Eigen::Vector3d d_i = destPoints[match.first];
        Eigen::Vector3d s_i = sourcePoints[match.second];

        Eigen::Matrix<double, 3, 6> A_i;
        A_i << 0, s_i.z(), -s_i.y(), 1, 0, 0,
            -s_i.z(), 0, s_i.x(), 0, 1, 0,
            s_i.y(), -s_i.x(), 0, 0, 0, 1;

        A.row(i * 3) = A_i.row(0);
        A.row(i * 3 + 1) = A_i.row(1);
        A.row(i * 3 + 2) = A_i.row(2);

        b(i * 3) = d_i.x() - s_i.x();
        b(i * 3 + 1) = d_i.y() - s_i.y();
        b(i * 3 + 2) = d_i.z() - s_i.z();
    }

    Eigen::Matrix<double, 6, 1> x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    solution = x;
}

// Creates the pose from the rotation angles and translation vector obtained after solving the linear system
// Rotation matrix Approximation as mentioned in the reference paper has been used
const Eigen::Matrix4d LinearSolver::getApproximatePose() {
    Eigen::Matrix4d transformation;
    double alpha = solution[0];
    double beta = solution[1];
    double gamma = solution[2];

    transformation << 1, alpha* beta - gamma, alpha* gamma + beta, solution[3],
        gamma, alpha* beta* gamma + 1, beta* gamma - alpha, solution[4],
        -beta, alpha, 1, solution[5],
        0, 0, 0, 1;
    return transformation;
}

// Creates the pose from the rotation angles and translation vector obtained after solving the linear system
// Unit Rotation matrices created from the angles is used
const Eigen::Matrix4d LinearSolver::getPose() {
    double alpha = solution[0];
    double beta = solution[1];
    double gamma = solution[2];
    Vector3d translation = solution.tail(3);

    Matrix3d rotation = AngleAxisd(alpha, Vector3d::UnitX()).toRotationMatrix() *
        AngleAxisd(beta, Vector3d::UnitY()).toRotationMatrix() *
        AngleAxisd(gamma, Vector3d::UnitZ()).toRotationMatrix();

    Matrix4d pose = Matrix4d::Identity();

    pose.block(0, 0, 3, 3) = rotation;
    pose.block(0, 3, 3, 1) = translation;

    return pose;
}

icp::icp(double dist_thresh, double normal_thresh)
    :dist_threshold(dist_thresh), normal_threshold(normal_thresh)
{}

// Checks whether the Euclidean distance between two points is within a certain threshold or not
bool icp::hasValidDistance(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2) {
    return (point1 - point2).norm() < dist_threshold;
}

// Checks whether the angle between the two normals is within a certain threshold or not
bool icp::hasValidAngle(const Eigen::Vector3d& normal1, const Eigen::Vector3d& normal2) {
    return std::abs(normal1.dot(normal2)) > normal_threshold;
}

// Find corresponding points between current frame and previous frame
// Method Used : Projective Point-Plane data association
// Return : vector of pairs of source and target vertex indices
// Reference Paper : Efficient variants of the ICP algorithm by Rusinkiewicz, Szymon and Levoy, Marc
void icp::findCorrespondence(CameraParametersPyramid cameraParams, std::shared_ptr<SurfaceMeasurement> prev_frame, std::shared_ptr<SurfaceMeasurement> curr_frame, std::vector<std::pair<size_t, size_t>>& corresponding_points, Eigen::Matrix4d& estimated_pose, int numLevels) {

    std::vector<Eigen::Vector3d> prev_frame_global_points = prev_frame->getGlobalVertexMap();
    std::vector<Eigen::Vector3d> prev_frame_global_normals = prev_frame->getGlobalNormalMap();

    std::vector<Eigen::Vector3d> curr_frame_points = curr_frame->getVertexMap();
    std::vector<Eigen::Vector3d> curr_frame_normals = curr_frame->getNormalMap();

    const auto rotation = estimated_pose.block(0, 0, 3, 3);
    const auto translation = estimated_pose.block(0, 3, 3, 1);

    for (size_t idx = 0; idx < curr_frame_points.size(); idx++) {

        Eigen::Vector3d curr_point = curr_frame_points[idx];
        Eigen::Vector3d curr_normal = curr_frame_normals[idx];

        if (curr_point.allFinite() && curr_normal.allFinite()) {
            const Eigen::Vector3d curr_global_point = rotation * curr_point + translation;
            const Eigen::Vector3d curr_global_normal = rotation * curr_normal;

            const Eigen::Vector3d curr_point_prev_frame = prev_frame->projectIntoCamera(curr_global_point);
            const Eigen::Vector2i curr_point_img_coord = prev_frame->projectOntoDepthPlane(curr_point_prev_frame, cameraParams.level(numLevels));

            if (prev_frame->contains(curr_point_img_coord, cameraParams.level(numLevels))) {

                size_t prev_idx = curr_point_img_coord[1] * cameraParams.level(numLevels).imageWidth + curr_point_img_coord[0];

                Eigen::Vector3d prev_global_point = prev_frame_global_points[prev_idx];
                Eigen::Vector3d prev_global_normal = prev_frame_global_normals[prev_idx];

                if (prev_global_point.allFinite() && prev_global_normal.allFinite()) {

                    if (hasValidDistance(prev_global_point, curr_global_point) &&
                        hasValidAngle(prev_global_normal, curr_global_normal)) {

                        corresponding_points.push_back(std::make_pair(prev_idx, idx));
                    }
                }
            }
        }
    }
}

// API to be called from outside the class
// Input : Two frames to be aligned
// Result : estimated pose
bool icp::estimatePose(CameraParametersPyramid cameraParams, int frame_cnt, std::shared_ptr<SurfaceMeasurement> prev_frame, std::shared_ptr<SurfaceMeasurement> curr_frame, size_t m_nIterations, Eigen::Matrix4d& estimated_pose, int numLevels)
{
    for (size_t i = 0; i < m_nIterations; ++i) {

        std::vector<std::pair<size_t, size_t>> corresponding_points;
        findCorrespondence(cameraParams, prev_frame, curr_frame, corresponding_points, estimated_pose, numLevels);

        LinearSolver solver;
        solver.solvePoint2Plane(curr_frame->getGlobalVertexMap(), prev_frame->getGlobalVertexMap(),
            prev_frame->getGlobalNormalMap(),
            corresponding_points);

        estimated_pose = solver.getApproximatePose() * estimated_pose;

        curr_frame->setGlobalPose(estimated_pose);
    }
    return true;
}