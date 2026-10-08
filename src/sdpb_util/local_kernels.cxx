#include "local_kernels.hxx"

#include <cstring>

namespace
{
  using local_kernels::Matrix;

  // Scratch variables at the working precision (one set per thread).
  struct Scratch
  {
    mpf_t t, gamma, u, one, half;
    mp_bitcnt_t bits = 0;
    Scratch() = default;
    Scratch(const Scratch &) = delete;
    Scratch &operator=(const Scratch &) = delete;
    ~Scratch()
    {
      if(bits != 0)
        clear();
    }
    void clear()
    {
      mpf_clear(t);
      mpf_clear(gamma);
      mpf_clear(u);
      mpf_clear(one);
      mpf_clear(half);
    }
    // BigFloat() is created at El::gmp::Precision(); so are these.
    void ensure()
    {
      const mp_bitcnt_t precision = El::gmp::Precision();
      if(bits == precision)
        return;
      if(bits != 0)
        clear();
      mpf_init2(t, precision);
      mpf_init2(gamma, precision);
      mpf_init2(u, precision);
      mpf_init2(one, precision);
      mpf_init2(half, precision);
      mpf_set_ui(one, 1);
      mpf_set_d(half, 0.5);
      bits = precision;
    }
  };

  Scratch &scratch()
  {
    thread_local Scratch s;
    s.ensure();
    return s;
  }

  inline mpf_ptr ptr(El::BigFloat &x)
  {
    return x.gmp_float.get_mpf_t();
  }
  inline mpf_srcptr ptr(const El::BigFloat &x)
  {
    return x.gmp_float.get_mpf_t();
  }

  // x := x * 1, i.e. what mpf_mul does to a value that carries the extra
  // limb: keep the top _mp_prec limbs (and give zero the exponent 0).
  inline void truncate_to_precision(mpf_ptr x)
  {
    const mp_size_t size = x->_mp_size;
    const mp_size_t abs_size = size < 0 ? -size : size;
    const mp_size_t prec = x->_mp_prec;
    if(abs_size == 0)
      {
        x->_mp_exp = 0;
        return;
      }
    if(abs_size > prec)
      {
        std::memmove(x->_mp_d, x->_mp_d + (abs_size - prec),
                     prec * sizeof(mp_limb_t));
        x->_mp_size = size < 0 ? -prec : prec;
      }
  }

  // x := alpha x exactly as BigFloat's "x *= alpha" does
  inline void scale(mpf_ptr x, mpf_srcptr alpha, const bool alpha_is_one)
  {
    if(alpha_is_one)
      truncate_to_precision(x);
    else
      mpf_mul(x, x, alpha);
  }

  // Scale all of C by beta as Elemental's generic Gemm/Syrk do
  void scale_all(const El::BigFloat &beta, Matrix &C)
  {
    const El::Int m = C.Height(), n = C.Width(), ldc = C.LDim();
    El::BigFloat *c = C.Buffer();
    if(beta == El::BigFloat(0))
      {
        for(El::Int j = 0; j < n; ++j)
          for(El::Int i = 0; i < m; ++i)
            mpf_set_ui(ptr(c[i + j * ldc]), 0);
      }
    else if(beta != El::BigFloat(1))
      {
        for(El::Int j = 0; j < n; ++j)
          for(El::Int i = 0; i < m; ++i)
            mpf_mul(ptr(c[i + j * ldc]), ptr(c[i + j * ldc]), ptr(beta));
      }
  }

  // Elemental's generic Gemm loops for the rows [row_begin(j), row_end(j))
  // of every column j of C
  template <class Begin, class End>
  void gemm_rows(const El::Orientation orientation_A,
                 const El::Orientation orientation_B,
                 const El::BigFloat &alpha, const Matrix &A, const Matrix &B,
                 const El::BigFloat &beta, Matrix &C, const Begin &row_begin,
                 const End &row_end)
  {
    const bool normal_A = orientation_A == El::NORMAL;
    const bool normal_B = orientation_B == El::NORMAL;
    const El::Int n = C.Width();
    const El::Int k = normal_A ? A.Width() : A.Height();
    const El::Int lda = A.LDim(), ldb = B.LDim(), ldc = C.LDim();
    const El::BigFloat *a = A.LockedBuffer();
    const El::BigFloat *b = B.LockedBuffer();
    El::BigFloat *c = C.Buffer();

    scale_all(beta, C);
    if(k == 0)
      return;
    auto &s = scratch();
    if(normal_A)
      {
        // C(:,j) += A(:,l) (alpha op(B)(l,j)): an axpy per (l, j)
        for(El::Int j = 0; j < n; ++j)
          for(El::Int l = 0; l < k; ++l)
            {
              const El::BigFloat &b_lj
                = normal_B ? b[l + j * ldb] : b[j + l * ldb];
              mpf_mul(s.gamma, ptr(alpha), ptr(b_lj));
              for(El::Int i = row_begin(j); i < row_end(j); ++i)
                {
                  mpf_mul(s.t, ptr(a[i + l * lda]), s.gamma);
                  mpf_add(ptr(c[i + j * ldc]), ptr(c[i + j * ldc]), s.t);
                }
            }
      }
    else
      {
        // C(i,j) += alpha (A(:,i) . op(B)(:,j)): a dot product per (i, j)
        for(El::Int j = 0; j < n; ++j)
          for(El::Int i = row_begin(j); i < row_end(j); ++i)
            {
              mpf_set_ui(s.gamma, 0);
              for(El::Int l = 0; l < k; ++l)
                {
                  const El::BigFloat &b_lj
                    = normal_B ? b[l + j * ldb] : b[j + l * ldb];
                  mpf_mul(s.t, ptr(a[l + i * lda]), ptr(b_lj));
                  mpf_add(s.gamma, s.gamma, s.t);
                }
              mpf_mul(s.gamma, s.gamma, ptr(alpha));
              mpf_add(ptr(c[i + j * ldc]), ptr(c[i + j * ldc]), s.gamma);
            }
      }
  }
}

namespace local_kernels
{
  void gemm(const El::Orientation orientation_A,
            const El::Orientation orientation_B, const El::BigFloat &alpha,
            const Matrix &A, const Matrix &B, const El::BigFloat &beta,
            Matrix &C)
  {
    gemm_rows(orientation_A, orientation_B, alpha, A, B, beta, C,
              [](El::Int) { return El::Int(0); },
              [&C](El::Int) { return C.Height(); });
  }

  void gemm_triangle(const El::UpperOrLower uplo,
                     const El::Orientation orientation_A,
                     const El::Orientation orientation_B,
                     const El::BigFloat &alpha, const Matrix &A,
                     const Matrix &B, const El::BigFloat &beta, Matrix &C)
  {
    const El::Int m = C.Height();
    if(uplo == El::LOWER)
      gemm_rows(orientation_A, orientation_B, alpha, A, B, beta, C,
                [](El::Int j) { return j; }, [m](El::Int) { return m; });
    else
      gemm_rows(orientation_A, orientation_B, alpha, A, B, beta, C,
                [](El::Int) { return El::Int(0); },
                [m](El::Int j) { return std::min(j + 1, m); });
  }

  void syrk(const El::UpperOrLower uplo, const El::Orientation orientation,
            const El::BigFloat &alpha, const Matrix &A,
            const El::BigFloat &beta, Matrix &C)
  {
    const bool normal = orientation == El::NORMAL;
    const bool lower = uplo == El::LOWER;
    const El::Int n = C.Height();
    const El::Int k = normal ? A.Width() : A.Height();
    const El::Int lda = A.LDim(), ldc = C.LDim();
    const El::BigFloat *a = A.LockedBuffer();
    El::BigFloat *c = C.Buffer();

    scale_all(beta, C);
    auto &s = scratch();
    for(El::Int j = 0; j < n; ++j)
      {
        const El::Int i_begin = lower ? j : 0;
        const El::Int i_end = lower ? n : j + 1;
        for(El::Int i = i_begin; i < i_end; ++i)
          {
            mpf_set_ui(s.gamma, 0);
            for(El::Int l = 0; l < k; ++l)
              {
                // normal: A(j,l) A(i,l); transpose: A(l,i) A(l,j)
                if(normal)
                  mpf_mul(s.t, ptr(a[j + l * lda]), ptr(a[i + l * lda]));
                else
                  mpf_mul(s.t, ptr(a[l + i * lda]), ptr(a[l + j * lda]));
                mpf_add(s.gamma, s.gamma, s.t);
              }
            mpf_mul(s.gamma, s.gamma, ptr(alpha));
            mpf_add(ptr(c[i + j * ldc]), ptr(c[i + j * ldc]), s.gamma);
          }
      }
  }

  void trsm(const El::LeftOrRight side, const El::UpperOrLower uplo,
            const El::Orientation orientation, const El::UnitOrNonUnit diag,
            const El::BigFloat &alpha, const Matrix &A, Matrix &B,
            const bool reciprocal)
  {
    const bool left = side == El::LEFT;
    const bool lower = uplo == El::LOWER;
    const bool normal = orientation == El::NORMAL;
    const bool unit_diag = diag == El::UNIT;
    const El::Int m = B.Height(), n = B.Width();
    const El::Int lda = A.LDim(), ldb = B.LDim();
    const El::BigFloat *a = A.LockedBuffer();
    El::BigFloat *b = B.Buffer();
    auto A_ = [&](const El::Int i, const El::Int j) { return ptr(a[i + j * lda]); };
    auto B_ = [&](const El::Int i, const El::Int j) { return ptr(b[i + j * ldb]); };

    // B := alpha B
    {
      const bool alpha_is_one = alpha == El::BigFloat(1);
      for(El::Int j = 0; j < n; ++j)
        for(El::Int i = 0; i < m; ++i)
          scale(B_(i, j), ptr(alpha), alpha_is_one);
    }
    auto &s = scratch();
    // Divide row (left) or column (right) k of B by A(k,k)
    auto divide = [&](const El::Int k) {
      if(unit_diag)
        return;
      const El::Int count = left ? n : m;
      if(reciprocal)
        mpf_ui_div(s.gamma, 1, A_(k, k));
      for(El::Int r = 0; r < count; ++r)
        {
          mpf_ptr x = left ? B_(k, r) : B_(r, k);
          if(reciprocal)
            mpf_mul(x, x, s.gamma);
          else
            mpf_div(x, x, A_(k, k));
        }
    };
    // B(i,j) -= x * y, the rank-one update of Elemental's Geru with alpha -1
    auto update = [&](mpf_ptr target, mpf_srcptr x, mpf_srcptr y) {
      mpf_mul(s.t, x, y);
      mpf_sub(target, target, s.t);
    };

    if(left)
      {
        // B is m x n, A is m x m
        if(normal == lower)
          {
            // forward: (N, lower) or (T, upper)
            for(El::Int k = 0; k < m; ++k)
              {
                divide(k);
                for(El::Int j = 0; j < n; ++j)
                  for(El::Int i = k + 1; i < m; ++i)
                    update(B_(i, j), lower ? A_(i, k) : A_(k, i), B_(k, j));
              }
          }
        else
          {
            // backward: (N, upper) or (T, lower)
            for(El::Int k = m - 1; k >= 0; --k)
              {
                divide(k);
                for(El::Int j = 0; j < n; ++j)
                  for(El::Int i = 0; i < k; ++i)
                    update(B_(i, j), lower ? A_(k, i) : A_(i, k), B_(k, j));
              }
          }
      }
    else
      {
        // B is m x n, A is n x n
        if(normal != lower)
          {
            // forward: (N, upper) or (T, lower)
            for(El::Int k = 0; k < n; ++k)
              {
                divide(k);
                for(El::Int j = k + 1; j < n; ++j)
                  for(El::Int i = 0; i < m; ++i)
                    update(B_(i, j), B_(i, k), lower ? A_(j, k) : A_(k, j));
              }
          }
        else
          {
            // backward: (N, lower) or (T, upper)
            for(El::Int k = n - 1; k >= 0; --k)
              {
                divide(k);
                for(El::Int j = 0; j < k; ++j)
                  for(El::Int i = 0; i < m; ++i)
                    update(B_(i, j), B_(i, k), lower ? A_(k, j) : A_(j, k));
              }
          }
      }
  }

  void axpy(const El::BigFloat &alpha, const Matrix &X, Matrix &Y)
  {
    const El::Int m = X.Height(), n = X.Width();
    const El::Int ldx = X.LDim(), ldy = Y.LDim();
    const El::BigFloat *x = X.LockedBuffer();
    El::BigFloat *y = Y.Buffer();
    auto &s = scratch();
    for(El::Int j = 0; j < n; ++j)
      for(El::Int i = 0; i < m; ++i)
        {
          mpf_mul(s.t, ptr(alpha), ptr(x[i + j * ldx]));
          mpf_add(ptr(y[i + j * ldy]), ptr(y[i + j * ldy]), s.t);
        }
  }

  void symmetrize(Matrix &A)
  {
    // h = A * 0.5 (El::Scale); A(i,j) = h(i,j) + 1 * h(j,i) (El::Axpy with
    // alpha 1, whose product truncates h(j,i) to the working precision)
    const El::Int n = A.Height(), lda = A.LDim();
    El::BigFloat *a = A.Buffer();
    auto &s = scratch();
    for(El::Int j = 0; j < n; ++j)
      {
        mpf_ptr a_jj = ptr(a[j + j * lda]);
        mpf_mul(s.t, a_jj, s.half);
        mpf_mul(s.u, s.one, s.t);
        mpf_add(a_jj, s.t, s.u);
        for(El::Int i = j + 1; i < n; ++i)
          {
            mpf_ptr a_ij = ptr(a[i + j * lda]);
            mpf_ptr a_ji = ptr(a[j + i * lda]);
            mpf_mul(s.t, a_ij, s.half);
            mpf_mul(s.gamma, a_ji, s.half);
            mpf_mul(s.u, s.one, s.gamma);
            mpf_add(a_ij, s.t, s.u);
            mpf_mul(s.u, s.one, s.t);
            mpf_add(a_ji, s.gamma, s.u);
          }
      }
  }

  void make_symmetric(const El::UpperOrLower uplo, Matrix &A)
  {
    const El::Int n = A.Height(), lda = A.LDim();
    El::BigFloat *a = A.Buffer();
    auto &s = scratch();
    for(El::Int j = 0; j < n; ++j)
      for(El::Int i = j + 1; i < n; ++i)
        {
          // (i, j) is below the diagonal, (j, i) above it
          if(uplo == El::LOWER)
            mpf_mul(ptr(a[j + i * lda]), s.one, ptr(a[i + j * lda]));
          else
            mpf_mul(ptr(a[i + j * lda]), s.one, ptr(a[j + i * lda]));
        }
  }
}
