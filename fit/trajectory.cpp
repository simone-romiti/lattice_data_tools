#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>

#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

// Define matrix/vector types using Eigen
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;

nb::dict fit_trajectory(
    std::function<Vector(const Vector&, const Vector&)> ansatz,
    const Matrix& x,
    const Matrix& ex,
    const Matrix& y,
    const Matrix& ey,
    const Vector& guess,
    const std::string& method = "BFGS",
    const nb::object& cov_inv_obj = nb::none()
) {

  std::cout << "c++ fitting routine with nanobind!\n";
  
    if (x.rows() != y.rows()) {
        throw std::invalid_argument("x and y must have the same number of rows.");
    }

    const int64_t N_pts = x.rows();
    const int64_t Nx_cols = x.cols();
    const int64_t Ny_cols = y.cols();
    const int64_t N_par = guess.size();
    const int64_t N_dof = N_pts - N_par;

    // Identify indices where ex > 0 and ey > 0
    std::vector<std::pair<int64_t, int64_t>> ix_with_err;
    for (int64_t r = 0; r < x.rows(); ++r) {
        for (int64_t c = 0; c < x.cols(); ++c) {
            if (ex(r, c) > 0.0) {
                ix_with_err.emplace_back(r, c);
            }
        }
    }

    std::vector<std::pair<int64_t, int64_t>> iy_with_err;
    for (int64_t r = 0; r < y.rows(); ++r) {
        for (int64_t c = 0; c < y.cols(); ++c) {
            if (ey(r, c) > 0.0) {
                iy_with_err.emplace_back(r, c);
            }
        }
    }

    // Build flattened vector of guess parameters: [p_ansatz, p_x_initial]
    Vector full_guess(N_par + ix_with_err.size());
    full_guess.head(N_par) = guess;
    for (size_t k = 0; k < ix_with_err.size(); ++k) {
        auto [r, c] = ix_with_err[k];
        full_guess(N_par + k) = x(r, c);
    }

    // Determine whether Cov_inv was passed
    bool has_cov_inv = !cov_inv_obj.is_none();
    Matrix cov_inv;

    if (has_cov_inv) {
        cov_inv = nb::cast<Matrix>(cov_inv_obj);
        int64_t total_dim = x.size() + y.size();
        if (cov_inv.rows() != cov_inv.cols() || cov_inv.rows() != total_dim) {
            throw std::invalid_argument("Cov_inv must be a square matrix of size (Nx + Ny, Nx + Ny)");
        }
    }

    // Chi-squared objective function evaluation
    auto eval_chi2 = [&](const Vector& p_all) -> double {
        Vector p_ansatz = p_all.head(N_par);

        if (!has_cov_inv) {
            double ch2_x = 0.0;
            for (size_t k = 0; k < ix_with_err.size(); ++k) {
                auto [r, c] = ix_with_err[k];
                double val = (x(r, c) - p_all(N_par + k)) / ex(r, c);
                ch2_x += val * val;
            }

            // Construct trajectory evaluation X_th
            Matrix X_th = x;
            for (size_t k = 0; k < ix_with_err.size(); ++k) {
                auto [r, c] = ix_with_err[k];
                X_th(r, c) = p_all(N_par + k);
            }

            double ch2_y = 0.0;
            for (int64_t i = 0; i < N_pts; ++i) {
                Vector x_row = X_th.row(i);
                Vector y_th = ansatz(x_row, p_ansatz);

                for (int64_t c = 0; c < Ny_cols; ++c) {
                    if (ey(i, c) > 0.0) {
                        double val = (y(i, c) - y_th(c)) / ey(i, c);
                        ch2_y += val * val;
                    }
                }
            }
            return ch2_x + ch2_y;
        } else {
            Matrix X_th = x;
            for (size_t k = 0; k < ix_with_err.size(); ++k) {
                auto [r, c] = ix_with_err[k];
                X_th(r, c) = p_all(N_par + k);
            }

            Matrix Y_th(N_pts, Ny_cols);
            for (int64_t i = 0; i < N_pts; ++i) {
                Vector x_row = X_th.row(i);
                Y_th.row(i) = ansatz(x_row, p_ansatz);
            }

            // Build dx and dy flattened vectors column-major/transposed matching NumPy logic
            Matrix dx = (x - X_th).transpose();
            Matrix dy = (y - Y_th).transpose();

            Vector z(dx.size() + dy.size());
            z.head(dx.size()) = Eigen::Map<const Vector>(dx.data(), dx.size());
            z.tail(dy.size()) = Eigen::Map<const Vector>(dy.data(), dy.size());

            return z.transpose() * cov_inv * z;
        }
    };

    // Callback to Python scipy.optimize for numerical minimization
    nb::object opt = nb::module_::import_("scipy.optimize");
    nb::object py_eval_chi2 = nb::cpp_function(eval_chi2);

    nb::object mini = opt.attr("minimize")(
        py_eval_chi2,
        full_guess,
        nb::arg("method") = method
    );

    Vector opt_par = nb::cast<Vector>(mini.attr("x"));
    double ch2_value = eval_chi2(opt_par);

    double ch2_dof = (N_dof > 0) ? (ch2_value / static_cast<double>(N_dof)) : std::numeric_limits<double>::quiet_NaN();

    // Prepare response dictionary matching original structure
    nb::dict res;
    res["ansatz"] = ansatz;
    res["N_par"] = N_par;
    res["N_pts"] = N_pts;
    res["par"] = opt_par;
    res["ch2"] = ch2_value;
    res["N_dof"] = N_dof;
    res["ch2_dof"] = ch2_dof;

    return res;
}

NB_MODULE(fit_module, m) {
    m.def("fit_trajectory", &fit_trajectory,
        nb::arg("ansatz"),
        nb::arg("x"),
        nb::arg("ex"),
        nb::arg("y"),
        nb::arg("ey"),
        nb::arg("guess"),
        nb::arg("method") = "BFGS",
        nb::arg("Cov_inv") = nb::none(),
        R"pbdoc(
            Fit a function f: R^n -> R^m with the trajectory method using nanobind and Eigen.
        )pbdoc"
    );
}

