use arrayvec::ArrayVec;
use cv_core::{
    nalgebra::{DMatrix, Matrix3, OMatrix, OVector, Point2, SVD, U8, U9},
    sample_consensus::Estimator,
    CameraToCamera, FeatureMatch,
};
use cv_optimize::HomographyOptimizer;
use cv_pinhole::HomographyMatrix;

#[derive(Copy, Clone, Debug)]
pub struct HomographyEstimatorDLT {
    pub epsilon: f64,
    pub iterations: usize,
}

impl HomographyEstimatorDLT {
    pub fn new() -> Self {
        Default::default()
    }

    /// A version that directly uses the SVD function, but is slightly slower
    pub fn _from_matches_svd<I>(&self, data: I) -> Option<HomographyMatrix>
    where
        I: Iterator<Item = FeatureMatch> + Clone,
    {
        let count = data.clone().count();
        assert!(count >= 4);

        let data = data.map(|FeatureMatch(src, dst)| {
            (
                Point2::from_homogeneous(src.xyz())
                    .expect("No points should be at infinity")
                    .coords,
                Point2::from_homogeneous(dst.xyz())
                    .expect("No points should be at infinity")
                    .coords,
            )
        });

        // Construct design matrix
        let design = data
            .clone()
            .flat_map(|(src, dst)| {
                vec![
                    0.0,
                    0.0,
                    0.0,
                    -src.x,
                    -src.y,
                    -1.0,
                    dst.y * src.x,
                    dst.y * src.y,
                    dst.y,
                    src.x,
                    src.y,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    -dst.x * src.x,
                    -dst.x * src.y,
                    -dst.x,
                ]
            })
            .collect::<Vec<_>>();

        let design_t = DMatrix::from_vec(9, 2 * count, design);

        let dtd = &design_t * design_t.transpose();

        let svd = SVD::try_new(dtd, false, true, self.epsilon, self.iterations).unwrap();

        let h = Matrix3::from_row_slice(svd.v_t.unwrap().row(8).transpose().as_slice());
        let h = h / h[(2, 2)];

        // Optimize homography matrix
        let mut h8 = OVector::<f64, U8>::zeros();
        OVector::<f64, U8>::copy_from_slice(&mut h8, &h.transpose().data.as_slice()[..8]);
        let optimizer = HomographyOptimizer {
            data: data.clone(),
            n: count,
            h: h8,
            w: HomographyOptimizer::calc_w(data.clone(), &h8),
        };
        let result = optimizer.optimize();

        let h = Matrix3::from_iterator(result.h.iter().chain([1.0].iter()).copied()).transpose();

        Some(HomographyMatrix(h))
    }

    /// A slightly faster version, inspired by the OpenCV code
    pub fn from_matches_opencv<I>(&self, data: I) -> Option<HomographyMatrix>
    where
        I: Iterator<Item = FeatureMatch> + Clone,
    {
        let count = data.clone().count();
        assert!(count >= 4);

        let data = data.clone().map(|FeatureMatch(src, dst)| {
            (
                Point2::from_homogeneous(src.xyz())
                    .expect("No points should be at infinity")
                    .coords,
                Point2::from_homogeneous(dst.xyz())
                    .expect("No points should be at infinity")
                    .coords,
            )
        });

        // Calculate the center of the source and destination points
        let (src_center, dst_center) = data
            .clone()
            .reduce(|(src_center, dst_center), (src, dst)| (src_center + src, dst_center + dst))
            .unwrap();
        let (src_center, dst_center) = (src_center / count as f64, dst_center / count as f64);

        // Center the data
        let data_centered = data
            .clone()
            .map(|(src, dst)| (src - src_center, dst - dst_center));

        // Calculate the standard deviation of the source and destination points
        let (src_std, dst_std) = data_centered.clone().fold(
            (Point2::default().coords, Point2::default().coords),
            |(src_std, dst_std), (src_centered, dst_centered)| {
                (src_std + src_centered.abs(), dst_std + dst_centered.abs())
            },
        );
        let src_std = src_std.map(|x| count as f64 / x);
        let dst_std = dst_std.map(|x| count as f64 / x);

        // Manually calculate the SVD
        let mut design = OMatrix::<f64, U9, U9>::zeros();
        for (p, q) in data_centered
            .map(|(src, dst)| (src.component_mul(&src_std), dst.component_mul(&dst_std)))
        {
            let row_x = vec![p.x, p.y, 1.0, 0.0, 0.0, 0.0, -q.x * p.x, -q.x * p.y, -q.x];
            let row_y = vec![0.0, 0.0, 0.0, p.x, p.y, 1.0, -q.y * p.x, -q.y * p.y, -q.y];

            for i in 0..9 {
                for j in i..9 {
                    design[(i, j)] += row_x[i] * row_x[j] + row_y[i] * row_y[j];
                }
            }
        }
        // Complete symmetry
        for i in 0..9 {
            for j in (i + 1)..9 {
                design[(j, i)] = design[(i, j)];
            }
        }

        let m = design
            .try_symmetric_eigen(self.epsilon, self.iterations)
            .unwrap();

        // We need to sort the eigenvectors by their corresponding eigenvalue.
        let mut sources = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        sources.sort_unstable_by_key(|&ix| float_ord::FloatOrd(m.eigenvalues[ix]));

        let h_0 =
            Matrix3::from_iterator(m.eigenvectors.column(sources[0]).iter().copied()).transpose();

        let inv_h_norm = Matrix3::new(
            1.0 / dst_std.x,
            0.0,
            dst_center.x,
            0.0,
            1.0 / dst_std.y,
            dst_center.y,
            0.0,
            0.0,
            1.0,
        );

        let h_norm2 = Matrix3::new(
            src_std.x,
            0.0,
            -src_center.x * src_std.x,
            0.0,
            src_std.y,
            -src_center.y * src_std.y,
            0.0,
            0.0,
            1.0,
        );

        let h_temp = inv_h_norm * h_0;

        let h = h_temp * h_norm2;

        let h_norm = h / h[(2, 2)];

        // Optimize homography matrix
        let mut h8 = OVector::<f64, U8>::zeros();
        OVector::<f64, U8>::copy_from_slice(&mut h8, &h_norm.transpose().data.as_slice()[..8]);
        let optimizer = HomographyOptimizer {
            data: data.clone(),
            n: count,
            h: h8,
            w: HomographyOptimizer::calc_w(data.clone(), &h8),
        };
        let result = optimizer.optimize();

        let h = Matrix3::from_iterator(result.h.iter().chain([1.0].iter()).copied()).transpose();

        Some(HomographyMatrix(h))
    }
}

impl Default for HomographyEstimatorDLT {
    fn default() -> Self {
        Self {
            epsilon: 1e-12,
            iterations: 1000,
        }
    }
}

impl Estimator<FeatureMatch> for HomographyEstimatorDLT {
    type Model = CameraToCamera;
    type ModelIter = ArrayVec<CameraToCamera, 4>;
    const MIN_SAMPLES: usize = 4;

    fn estimate<I>(&self, data: I) -> Self::ModelIter
    where
        I: Iterator<Item = FeatureMatch> + Clone,
    {
        self.from_matches_opencv(data)
            .and_then(|homography| {
                homography.possible_unscaled_poses(self.epsilon, self.iterations)
            })
            .map(Into::into)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{
        nalgebra::{IsometryMatrix3, Point3, Rotation3, UnitVector3, Vector3},
        sample_consensus::Model,
        CameraPoint, CameraToCamera, FeatureMatch, Pose, Projective,
    };

    const SAMPLE_POINTS: usize = 16;
    // TODO: I'm not sure what an acceptable threshold is
    const RESIDUAL_THRESHOLD: f64 = 1e-4;

    const ROT_MAGNITUDE: f64 = 0.2;
    const POINT_BOX_SIZE: f64 = 2.0;
    const POINT_DISTANCE: f64 = 3.0;

    #[test]
    fn randomized() {
        let successes = (0..1000).filter(|_| run_round()).count();
        assert!(successes > 950);
    }

    fn run_round() -> bool {
        let mut success = true;
        let (_, aps, bps) = some_test_data();
        let matches = aps.iter().zip(&bps).map(|(&a, &b)| FeatureMatch(a, b));
        let homography_estimator = HomographyEstimatorDLT::new();
        let homography = homography_estimator
            .from_matches_opencv(matches.clone())
            .expect("didn't get a homography matrix");
        for m in matches.clone() {
            if homography.residual(&m).abs() > RESIDUAL_THRESHOLD {
                success = false;
            }
        }
        success
    }

    /// Gets a random relative pose, input points A, input points B, and A point depths.
    fn some_test_data() -> (
        CameraToCamera,
        [UnitVector3<f64>; SAMPLE_POINTS],
        [UnitVector3<f64>; SAMPLE_POINTS],
    ) {
        // The relative pose orientation is fixed and translation is random.
        let relative_pose = CameraToCamera(IsometryMatrix3::from_parts(
            Vector3::new_random().into(),
            Rotation3::new(Vector3::new_random() * std::f64::consts::PI * 2.0 * ROT_MAGNITUDE),
        ));

        // Generate A's camera points.
        let cams_a = (0..SAMPLE_POINTS)
            .map(|_| {
                let mut a = Point3::from(Vector3::new_random() * POINT_BOX_SIZE);
                a.x -= 0.5 * POINT_BOX_SIZE;
                a.y -= 0.5 * POINT_BOX_SIZE;
                a.z += POINT_DISTANCE;
                CameraPoint::from_point(a)
            })
            .collect::<Vec<_>>()
            .into_iter();

        // Generate B's camera points.
        let cams_b = cams_a.clone().map(|a| relative_pose.transform(a));

        let mut kps_a = [UnitVector3::new_normalize(Vector3::z()); SAMPLE_POINTS];
        for (keypoint, camera) in kps_a.iter_mut().zip(cams_a) {
            *keypoint = camera.bearing();
        }
        let mut kps_b = [UnitVector3::new_normalize(Vector3::z()); SAMPLE_POINTS];
        for (keypoint, camera) in kps_b.iter_mut().zip(cams_b.clone()) {
            *keypoint = camera.bearing();
        }

        (relative_pose, kps_a, kps_b)
    }
}
