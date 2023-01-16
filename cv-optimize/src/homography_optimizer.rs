use cv_core::nalgebra::{Dynamic, Matrix, OMatrix, OVector, Owned, VecStorage, U1, U2, U8};
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};

type DataIterator = (OVector<f64, U2>, OVector<f64, U2>);

pub struct HomographyOptimizer<I>
where
    I: Iterator<Item = DataIterator> + Clone,
{
    pub h: OVector<f64, U8>,
    pub n: usize,
    pub data: I,
    pub w: Vec<f64>,
}

impl<I> LeastSquaresProblem<f64, Dynamic, U8> for HomographyOptimizer<I>
where
    I: Iterator<Item = DataIterator> + Clone,
{
    type ParameterStorage = Owned<f64, U8>;
    type ResidualStorage = Owned<f64, Dynamic>;
    type JacobianStorage = Owned<f64, Dynamic, U8>;

    fn set_params(&mut self, h: &OVector<f64, U8>) {
        self.h.copy_from(h);
        self.w = Self::calc_w(self.data.clone(), &self.h);
    }

    fn params(&self) -> OVector<f64, U8> {
        self.h
    }

    fn residuals(&self) -> Option<Matrix<f64, Dynamic, U1, VecStorage<f64, Dynamic, U1>>> {
        let mut residuals = Vec::with_capacity(2 * self.n);
        for (i, (src, dst)) in self.data.clone().enumerate() {
            let x = (self.h[0] * src.x + self.h[1] * src.y + self.h[2]) * self.w[i];
            let y = (self.h[3] * src.x + self.h[4] * src.y + self.h[5]) * self.w[i];
            residuals.push(x - dst.x);
            residuals.push(y - dst.y);
        }
        Some(OVector::from(residuals))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dynamic, U8, VecStorage<f64, Dynamic, U8>>> {
        let mut jacobian = OMatrix::<f64, Dynamic, U8>::zeros(2 * self.n);
        for (i, (src, _)) in self.data.clone().enumerate() {
            let x = (self.h[0] * src.x + self.h[1] * src.y + self.h[2]) * self.w[i];
            let y = (self.h[3] * src.x + self.h[4] * src.y + self.h[5]) * self.w[i];

            jacobian[(2 * i, 0)] = src.x * self.w[i];
            jacobian[(2 * i, 1)] = src.y * self.w[i];
            jacobian[(2 * i, 2)] = self.w[i];
            jacobian[(2 * i, 6)] = -src.x * x * self.w[i];
            jacobian[(2 * i, 7)] = -src.y * x * self.w[i];

            jacobian[(2 * i + 1, 3)] = src.x * self.w[i];
            jacobian[(2 * i + 1, 4)] = src.y * self.w[i];
            jacobian[(2 * i + 1, 5)] = self.w[i];
            jacobian[(2 * i + 1, 6)] = -src.x * y * self.w[i];
            jacobian[(2 * i + 1, 7)] = -src.y * y * self.w[i];
        }
        Some(jacobian)
    }
}

impl<I> HomographyOptimizer<I>
where
    I: Iterator<Item = DataIterator> + Clone,
{
    pub fn calc_w(data: I, h: &OVector<f64, U8>) -> Vec<f64> {
        data.map(|(src, _)| {
            let w = h[6] * src.x + h[7] * src.y + 1.0;
            if w.abs() > f64::EPSILON {
                1.0 / w
            } else {
                0.0
            }
        })
        .collect()
    }

    pub fn optimize(self) -> Self {
        LevenbergMarquardt::new().with_gtol(0.0).minimize(self).0
    }
}
