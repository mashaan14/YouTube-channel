# SfM

Vkmfiovfkjmv

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

