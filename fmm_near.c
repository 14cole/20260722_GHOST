/*
 * Fast near-field element-pair block computation for 2D Helmholtz BIE.
 * Supports variable Gauss-Legendre quadrature order (2-16 points).
 * Compile: gcc -O3 -shared -fPIC -o fmm_near.so fmm_near.c -lm
 */
#include <math.h>
#include <string.h>

/* Gauss-Legendre nodes/weights on [0,1], orders 2-16.
   Generated from numpy.polynomial.legendre.leggauss, shifted to [0,1]. */
#define MAX_ORDER 16

static int gauss_offsets[MAX_ORDER+1];  /* offset into gauss_data for order q */
static double gauss_t[200];  /* quadrature nodes, packed */
static double gauss_w[200];  /* quadrature weights, packed */
static int gauss_initialized = 0;

/* Initialize on first use from precomputed 8-point rule (higher orders added dynamically) */
static const double G8T[] = {
    0.01985507175123188, 0.10166676129318691, 0.23723379504183550,
    0.40828267875217510, 0.59171732124782490, 0.76276620495816450,
    0.89833323870681309, 0.98014492824876812
};
static const double G8W[] = {
    0.05061426814518813, 0.11119051722668724, 0.15685332293894344,
    0.18134189168918100, 0.18134189168918100, 0.15685332293894344,
    0.11119051722668724, 0.05061426814518813
};

/*
 * compute_sk_block: 2x2 SLP and K'/K blocks for one element pair.
 *
 * Uses nq-point Gauss-Legendre quadrature on [0,1]×[0,1].
 * The quadrature nodes/weights are passed in via qt/qw arrays.
 *
 * For real wavenumber k:
 *   G(r) = (j/4) H_0^(2)(kr) = (-Y0(kr)/4) + i*(J0(kr)/4)
 *   dG/dn_obs = (-j*k/4) H_1^(2)(kr) * (r·n_obs)/|r|
 *   dG/dn_src = (j*k/4) H_1^(2)(kr) * (n_src·r)/|r|
 */
void compute_sk_block_q(
    int nq, const double *qt, const double *qw,
    double obs_p0x, double obs_p0y, double obs_sx, double obs_sy,
    double obs_nx, double obs_ny, double obs_len,
    double src_p0x, double src_p0y, double src_sx, double src_sy,
    double src_nx, double src_ny, double src_len,
    double k, int obs_nd,
    double *s_re, double *s_im, double *k_re, double *k_im)
{
    int i;
    for (i = 0; i < 4; i++) { s_re[i]=0; s_im[i]=0; k_re[i]=0; k_im[i]=0; }

    for (int qi = 0; qi < nq; qi++) {
        double to = qt[qi], wo = qw[qi];
        double po[2] = {1.0-to, to};
        double rx = obs_p0x + to*obs_sx, ry = obs_p0y + to*obs_sy;
        for (int qj = 0; qj < nq; qj++) {
            double ts = qt[qj], ws = qw[qj];
            double ps[2] = {1.0-ts, ts};
            double sx = src_p0x + ts*src_sx, sy = src_p0y + ts*src_sy;
            double dx = rx-sx, dy = ry-sy;
            double dist = sqrt(dx*dx + dy*dy);
            if (dist < 1e-15) dist = 1e-15;
            double kr = k * dist;
            double J0v=j0(kr), Y0v=y0(kr), J1v=j1(kr), Y1v=y1(kr);
            double g_re=0.25*Y0v, g_im=0.25*J0v;
            double h1_re=J1v, h1_im=-Y1v;
            double dk_re2, dk_im2;
            if (obs_nd) {
                double proj = (dx*obs_nx + dy*obs_ny)/dist;
                dk_re2 = ( 0.25*k*h1_im)*proj;
                dk_im2 = (-0.25*k*h1_re)*proj;
            } else {
                double proj = (src_nx*dx + src_ny*dy)/dist;
                dk_re2 = (-0.25*k*h1_im)*proj;
                dk_im2 = ( 0.25*k*h1_re)*proj;
            }
            double w = wo*ws*obs_len*src_len;
            for (int a=0; a<2; a++) for (int b=0; b<2; b++) {
                double c = po[a]*ps[b]*w;
                int idx = a*2+b;
                s_re[idx]+=c*g_re; s_im[idx]+=c*g_im;
                k_re[idx]+=c*dk_re2; k_im[idx]+=c*dk_im2;
            }
        }
    }
}

/*
 * Batch: compute N element-pair blocks.
 * All arrays flat, row-major. qt/qw have nq entries each.
 */
void compute_sk_blocks_batch_q(
    int n_pairs, int nq, const double *qt, const double *qw,
    const double *obs_p0, const double *obs_seg, const double *obs_n, const double *obs_len,
    const double *src_p0, const double *src_seg, const double *src_n, const double *src_len,
    double k, int obs_nd,
    double *s_re, double *s_im, double *k_re, double *k_im)
{
    for (int p = 0; p < n_pairs; p++) {
        compute_sk_block_q(nq, qt, qw,
            obs_p0[2*p], obs_p0[2*p+1], obs_seg[2*p], obs_seg[2*p+1],
            obs_n[2*p], obs_n[2*p+1], obs_len[p],
            src_p0[2*p], src_p0[2*p+1], src_seg[2*p], src_seg[2*p+1],
            src_n[2*p], src_n[2*p+1], src_len[p],
            k, obs_nd,
            &s_re[4*p], &s_im[4*p], &k_re[4*p], &k_im[4*p]);
    }
}
