use cv_core::{
    nalgebra::{Matrix3, Rotation3, UnitVector3, Vector3, SVD},
    sample_consensus::Model,
    CameraToCamera, FeatureMatch, Pose,
};
use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};

/// This stores a homography matrix, which is satisfied by the following constraint:
///
/// s * x' = H * x
///
/// where `x` and `x'` are homogeneous normalized image coordinates. `s` is a scalar.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct HomographyMatrix(pub Matrix3<f64>);

impl HomographyMatrix {
    pub fn possible_rotations_unscaled_translation(
        &self,
        epsilon: f64,
        max_iterations: usize,
    ) -> Option<(Rotation3<f64>, Rotation3<f64>, Vector3<f64>)> {
        let Self(homography) = *self;

        // normalize H to ensure that ||c1|| = 1
        let norm =
            (homography[(0, 0)].powi(2) + homography[(0, 1)].powi(2) + homography[(0, 2)].powi(2))
                .sqrt();

        let homography = homography / norm;

        let t = homography.column(2).into_owned();

        let c1 = homography.column(0).into_owned();
        let c2 = homography.column(1).into_owned();
        let c3 = c1.cross(&c2);

        let rotation = Matrix3::from_columns(&[c1, c2, c3]);

        let svd = SVD::try_new(rotation, true, true, epsilon, max_iterations).unwrap();
        let u = svd.u.unwrap();
        let v_t = svd.v_t.unwrap();

        let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        let r1 = u * w * v_t;
        let r2 = u * w.transpose() * v_t;

        Some((
            Rotation3::from_matrix_unchecked(r1),
            Rotation3::from_matrix_unchecked(r2),
            t,
        ))
    }

    pub fn possible_rotations(
        &self,
        epsilon: f64,
        max_iterations: usize,
    ) -> Option<[Rotation3<f64>; 2]> {
        self.possible_rotations_unscaled_translation(epsilon, max_iterations)
            .map(|(rot_a, rot_b, _)| [rot_a, rot_b])
    }

    pub fn possible_unscaled_poses(
        &self,
        epsilon: f64,
        max_iterations: usize,
    ) -> Option<[CameraToCamera; 4]> {
        self.possible_rotations_unscaled_translation(epsilon, max_iterations)
            .map(|(rot_a, rot_b, t)| {
                [
                    CameraToCamera::from_parts(t, rot_a),
                    CameraToCamera::from_parts(t, rot_b),
                    CameraToCamera::from_parts(-t, rot_a),
                    CameraToCamera::from_parts(-t, rot_b),
                ]
            })
    }

    pub fn possible_unscaled_poses_bearing(
        &self,
        epsilon: f64,
        max_iterations: usize,
    ) -> Option<[CameraToCamera; 2]> {
        self.possible_rotations_unscaled_translation(epsilon, max_iterations)
            .map(|(rot_a, rot_b, t)| {
                [
                    CameraToCamera::from_parts(t, rot_a),
                    CameraToCamera::from_parts(t, rot_b),
                ]
            })
    }
}

impl From<CameraToCamera> for HomographyMatrix {
    fn from(pose: CameraToCamera) -> Self {
        Self(pose.0.translation.vector.cross_matrix() * *pose.0.rotation.matrix())
    }
}

impl Model<FeatureMatch> for HomographyMatrix {
    fn residual(&self, data: &FeatureMatch) -> f64 {
        let Self(mat) = *self;
        let &FeatureMatch(a, b) = data;
        let normalized = |p: UnitVector3<f64>| p.into_inner() / p.z;

        let residual = mat * normalized(a) - normalized(b);
        residual.norm()
    }
}
