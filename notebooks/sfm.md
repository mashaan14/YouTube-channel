# Structure-from-Motion (SfM): A Tutorial

## SfM initialization in colmap
Colmap implements an incremental SfM in a [cpp file](https://github.com/colmap/colmap/blob/cb02ca13a57e565c6bfb56f5f88d65dab222cd7b/src/colmap/sfm/incremental_mapper_impl.cc) under `colmap/src/colmap/sfm/incremental_mapper_impl.cc`. It finds a first image by sorting all images in a way that priortize images with a large number of correspondences and have camera calibration priors. Large number of correspondences would make it easy to pair it with a second image, and having camera calibration priors would allow colmap to use the essential matrix $E$ to estimate the camera poses.

Here's 

inside `colmap/src/colmap/sfm/incremental_mapper_impl.cc` there is a `FindInitialImagePair`. It finds first image using `FindInitialImagePair`, then a second image using `FindSecondInitialImage`

```cpp
if (geometry->config == TwoViewGeometry::ConfigurationType::CALIBRATED ||
      geometry->config == TwoViewGeometry::ConfigurationType::UNCALIBRATED) {
    // Try to recover relative pose for calibrated and uncalibrated
    // configurations. In the uncalibrated case, this most likely leads to a
    // ill-defined reconstruction, but sometimes it succeeds anyways after e.g.
    // subsequent bundle-adjustment etc.
    PoseFromEssentialMatrix(geometry->E,
                            inlier_points1_normalized,
                            inlier_points2_normalized,
                            &geometry->cam2_from_cam1,
                            &points3D);
  } else if (geometry->config == TwoViewGeometry::ConfigurationType::PLANAR ||
             geometry->config ==
                 TwoViewGeometry::ConfigurationType::PANORAMIC ||
             geometry->config ==
                 TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC) {
    Eigen::Vector3d normal;
    PoseFromHomographyMatrix(geometry->H,
                             camera1.CalibrationMatrix(),
                             camera2.CalibrationMatrix(),
                             inlier_points1_normalized,
                             inlier_points2_normalized,
                             &geometry->cam2_from_cam1,
                             &normal,
                             &points3D);
  } else {
    return false;
  }
```

![Screenshot 2024-12-29 102507](https://github.com/user-attachments/assets/cbcfd067-2cd9-43e1-b5d7-9db3228a4a4a)

