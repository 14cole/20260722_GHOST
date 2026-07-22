/*
 * Phase-7c native sampling kernel for the BoR streaming assembly
 * (bor_streaming.py).  Fills the azimuthal integrand tiles that dominate
 * the streamed far-block build:
 *
 *   sample_g     g(xi)  = exp(-j k R)/(4 pi R)          (EFIE Green orders)
 *   sample_mfie  the four MFIE bracket functions * p(R)
 *   sample_ibc   the four IBC (rotated-PV) bracket functions * p(R)
 *   with p(R) = (1 + j k R) exp(-j k R)/(4 pi R^3)
 *
 * Real wavenumber only (the streaming path serves the exterior/air region;
 * bor_streaming falls back to the NumPy sampler for complex k).  Output
 * arrays are interleaved complex doubles laid out [rows, np, nxi].
 * Coincident points are clamped to R = 1e-30 -> large-but-finite garbage;
 * the caller zeroes every near/adjacent-element pair after the FFT exactly
 * as the table path does.
 *
 * Compile (see bor_streaming._load_native for the expected name):
 *   cc -O3 -shared -fPIC -o bor_stream_kernel.$(uname -s | tr 'A-Z' 'a-z')-$(uname -m).so bor_stream_kernel.c -lm
 */
#include <math.h>
#include <stddef.h>

#define R_MIN 1e-30

void sample_g(int nr, int np_, int nxi,
              const double *rho_p, const double *z_p,
              const double *rho_q, const double *z_q,
              double k, const double *sin2_tab, double *out)
{
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < np_; j++) {
            double dr = rho_p[i] - rho_q[j];
            double dz = z_p[i] - z_q[j];
            double d2 = dr * dr + dz * dz;
            double rr4 = 4.0 * rho_p[i] * rho_q[j];
            double *o = out + 2 * (size_t)nxi * ((size_t)i * np_ + j);
            for (int l = 0; l < nxi; l++) {
                double R = sqrt(d2 + rr4 * sin2_tab[l]);
                if (R < R_MIN) R = R_MIN;
                double a = 1.0 / (4.0 * M_PI * R);
                double kr = k * R;
                o[2 * l] = a * cos(kr);
                o[2 * l + 1] = -a * sin(kr);
            }
        }
    }
}

/* shared bracket-point core: computes p(R) (complex) and R components */
static inline void pR(double Rx, double Ry, double Rz, double k,
                      double *R_out, double *p_re, double *p_im)
{
    double R = sqrt(Rx * Rx + Ry * Ry + Rz * Rz);
    if (R < R_MIN) R = R_MIN;
    double pre = 1.0 / (4.0 * M_PI * R * R * R);
    double kr = k * R;
    double c = cos(kr), s = sin(kr);
    /* (1 + j kr)(c - j s) = (c + kr s) + j (kr c - s) */
    *p_re = pre * (c + kr * s);
    *p_im = pre * (kr * c - s);
    *R_out = R;
}

void sample_mfie(int nr, int np_, int nxi,
                 const double *rho_p, const double *z_p,
                 const double *tr_p, const double *tz_p,
                 const double *rho_q, const double *z_q,
                 const double *tr_q, const double *tz_q,
                 double k, const double *cx_tab, const double *sx_tab,
                 double *o_tt, double *o_tf, double *o_ft, double *o_ff)
{
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < np_; j++) {
            double Rz = z_p[i] - z_q[j];
            size_t base = 2 * (size_t)nxi * ((size_t)i * np_ + j);
            double *tt = o_tt + base, *tf = o_tf + base;
            double *ft = o_ft + base, *ff = o_ff + base;
            for (int l = 0; l < nxi; l++) {
                double cx = cx_tab[l], sx = sx_tab[l];
                double Rx = rho_p[i] - rho_q[j] * cx;
                double Ry = rho_q[j] * sx;
                double R, p_re, p_im;
                pR(Rx, Ry, Rz, k, &R, &p_re, &p_im);
                double WtR = tr_p[i] * Rx + tz_p[i] * Rz;
                double WfR = Ry;
                double nR = -tz_p[i] * Rx + tr_p[i] * Rz;
                double n_tq = -tz_p[i] * tr_q[j] * cx + tr_p[i] * tz_q[j];
                double n_fq = -tz_p[i] * sx;
                double Wt_tq = tr_p[i] * tr_q[j] * cx + tz_p[i] * tz_q[j];
                double Wt_fq = tr_p[i] * sx;
                double Wf_tq = -tr_q[j] * sx;
                double Wf_fq = cx;
                double f;
                f = -(WtR * n_tq - Wt_tq * nR);
                tt[2 * l] = f * p_re; tt[2 * l + 1] = f * p_im;
                f = -(WtR * n_fq - Wt_fq * nR);
                tf[2 * l] = f * p_re; tf[2 * l + 1] = f * p_im;
                f = -(WfR * n_tq - Wf_tq * nR);
                ft[2 * l] = f * p_re; ft[2 * l + 1] = f * p_im;
                f = -(WfR * n_fq - Wf_fq * nR);
                ff[2 * l] = f * p_re; ff[2 * l + 1] = f * p_im;
            }
        }
    }
}

void sample_ibc(int nr, int np_, int nxi,
                const double *rho_p, const double *z_p,
                const double *tr_p, const double *tz_p,
                const double *rho_q, const double *z_q,
                const double *tr_q, const double *tz_q,
                double k, const double *cx_tab, const double *sx_tab,
                double *o_tt, double *o_tf, double *o_ft, double *o_ff)
{
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < np_; j++) {
            double Rz = z_p[i] - z_q[j];
            size_t base = 2 * (size_t)nxi * ((size_t)i * np_ + j);
            double *tt = o_tt + base, *tf = o_tf + base;
            double *ft = o_ft + base, *ff = o_ff + base;
            for (int l = 0; l < nxi; l++) {
                double cx = cx_tab[l], sx = sx_tab[l];
                double Rx = rho_p[i] - rho_q[j] * cx;
                double Ry = rho_q[j] * sx;
                double R, p_re, p_im;
                pR(Rx, Ry, Rz, k, &R, &p_re, &p_im);
                double Wt_nq = -tr_p[i] * tz_q[j] * cx + tz_p[i] * tr_q[j];
                double Wf_nq = tz_q[j] * sx;
                double D = rho_p[i] * cx - rho_q[j];
                double R_tq = tr_q[j] * D + tz_q[j] * Rz;
                double R_fq = rho_p[i] * sx;
                double R_nq = -tz_q[j] * D + tr_q[j] * Rz;
                double Wt_tq = tr_p[i] * tr_q[j] * cx + tz_p[i] * tz_q[j];
                double Wt_fq = tr_p[i] * sx;
                double Wf_tq = -tr_q[j] * sx;
                double Wf_fq = cx;
                double f;
                f = Wt_nq * R_tq - Wt_tq * R_nq;
                tt[2 * l] = f * p_re; tt[2 * l + 1] = f * p_im;
                f = Wt_nq * R_fq - Wt_fq * R_nq;
                tf[2 * l] = f * p_re; tf[2 * l + 1] = f * p_im;
                f = Wf_nq * R_tq - Wf_tq * R_nq;
                ft[2 * l] = f * p_re; ft[2 * l + 1] = f * p_im;
                f = Wf_nq * R_fq - Wf_fq * R_nq;
                ff[2 * l] = f * p_re; ff[2 * l + 1] = f * p_im;
            }
        }
    }
}
