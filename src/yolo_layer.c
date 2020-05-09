#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(total*2, sizeof(float));
    if(mask) l.mask = mask;
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.truths = 90*(4 + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

// where c is the smallest box that fully encompases a and b
boxabs box_c(box a, box b) {
    boxabs ba = { 0 };
    ba.top = fmin(a.y - a.h / 2, b.y - b.h / 2);
    ba.bot = fmax(a.y + a.h / 2, b.y + b.h / 2);
    ba.left = fmin(a.x - a.w / 2, b.x - b.w / 2);
    ba.right = fmax(a.x + a.w / 2, b.x + b.w / 2);
    return ba;
}

// representation from x, y, w, h to top, left, bottom, right
boxabs to_tblr(box a) {
    boxabs tblr = { 0 };
    float t = a.y - (a.h / 2);
    float b = a.y + (a.h / 2);
    float l = a.x - (a.w / 2);
    float r = a.x + (a.w / 2);
    tblr.top = t;
    tblr.bot = b;
    tblr.left = l;
    tblr.right = r;
    return tblr;
}

static float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

static float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_giou(box a, box b)
{
    boxabs ba = box_c(a, b);
    float w = ba.right - ba.left;
    float h = ba.bot - ba.top;
    float c = w*h;
    float iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float u = box_union(a, b);
    float giou_term = (c - u) / c;
#ifdef DEBUG_PRINTS
    printf("  c: %f, u: %f, giou_term: %f\n", c, u, giou_term);
#endif
    return iou - giou_term;
}

// https://github.com/Zzh-tju/DIoU-darknet
// https://arxiv.org/abs/1911.08287
float box_diou(box a, box b)
{
    boxabs ba = box_c(a, b);
    float w = ba.right - ba.left;
    float h = ba.bot - ba.top;
    float c = w * w + h * h;
    float iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    float u = pow(d / c, 0.6);
    float diou_term = u;
#ifdef DEBUG_PRINTS
    printf("  c: %f, u: %f, riou_term: %f\n", c, u, diou_term);
#endif
    return iou - diou_term;
}

// https://github.com/Zzh-tju/DIoU-darknet
// https://arxiv.org/abs/1911.08287
float box_ciou(box a, box b)
{
    boxabs ba = box_c(a, b);
    float w = ba.right - ba.left;
    float h = ba.bot - ba.top;
    float c = w * w + h * h;
    float iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float u = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    float d = u / c;
    float ar_gt = b.w / b.h;
    float ar_pred = a.w / a.h;
    float ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred));
    float alpha = ar_loss / (1 - iou + ar_loss + 0.000001);
    float ciou_term = d + alpha * ar_loss;                   //ciou
#ifdef DEBUG_PRINTS
    printf("  c: %f, u: %f, riou_term: %f\n", c, u, ciou_term);
#endif
    return iou - ciou_term;
}

dxrep dx_box_iou(box pred, box truth, IOU_LOSS iou_loss) {
 boxabs pred_tblr = to_tblr(pred);
    float pred_t = fmin(pred_tblr.top, pred_tblr.bot);
    float pred_b = fmax(pred_tblr.top, pred_tblr.bot);
    float pred_l = fmin(pred_tblr.left, pred_tblr.right);
    float pred_r = fmax(pred_tblr.left, pred_tblr.right);
    //dbox dover = derivative(pred,truth);
    //dbox diouu = diou(pred, truth);
    boxabs truth_tblr = to_tblr(truth);
#ifdef DEBUG_PRINTS
    printf("\niou: %f, giou: %f\n", box_iou(pred, truth), box_giou(pred, truth));
    printf("pred: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)\n", pred.x, pred.y, pred.w, pred.h, pred_tblr.top, pred_tblr.bot, pred_tblr.left, pred_tblr.right);
    printf("truth: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)\n", truth.x, truth.y, truth.w, truth.h, truth_tblr.top, truth_tblr.bot, truth_tblr.left, truth_tblr.right);
#endif
    //printf("pred (t,b,l,r): (%f, %f, %f, %f)\n", pred_t, pred_b, pred_l, pred_r);
    //printf("trut (t,b,l,r): (%f, %f, %f, %f)\n", truth_tblr.top, truth_tblr.bot, truth_tblr.left, truth_tblr.right);
    dxrep ddx = {0};
    float X = (pred_b - pred_t) * (pred_r - pred_l);
    float Xhat = (truth_tblr.bot - truth_tblr.top) * (truth_tblr.right - truth_tblr.left);
    float Ih = fmin(pred_b, truth_tblr.bot) - fmax(pred_t, truth_tblr.top);
    float Iw = fmin(pred_r, truth_tblr.right) - fmax(pred_l, truth_tblr.left);
    float I = Iw * Ih;
    float U = X + Xhat - I;
    float S = (pred.x-truth.x)*(pred.x-truth.x)+(pred.y-truth.y)*(pred.y-truth.y);
    float giou_Cw = fmax(pred_r, truth_tblr.right) - fmin(pred_l, truth_tblr.left);
    float giou_Ch = fmax(pred_b, truth_tblr.bot) - fmin(pred_t, truth_tblr.top);
    float giou_C = giou_Cw * giou_Ch;
    //float IoU = I / U;
//#ifdef DEBUG_PRINTS
    //printf("X: %f", X);
    //printf(", Xhat: %f", Xhat);
    //printf(", Ih: %f", Ih);
    //printf(", Iw: %f", Iw);
    //printf(", I: %f", I);
    //printf(", U: %f", U);
    //printf(", IoU: %f\n", I / U);
//#endif

    //Partial Derivatives, derivatives
    float dX_wrt_t = -1 * (pred_r - pred_l);
    float dX_wrt_b = pred_r - pred_l;
    float dX_wrt_l = -1 * (pred_b - pred_t);
    float dX_wrt_r = pred_b - pred_t;
    // UNUSED
    //// Ground truth
    //float dXhat_wrt_t = -1 * (truth_tblr.right - truth_tblr.left);
    //float dXhat_wrt_b = truth_tblr.right - truth_tblr.left;
    //float dXhat_wrt_l = -1 * (truth_tblr.bot - truth_tblr.top);
    //float dXhat_wrt_r = truth_tblr.bot - truth_tblr.top;

    // gradient of I min/max in IoU calc (prediction)
    float dI_wrt_t = pred_t > truth_tblr.top ? (-1 * Iw) : 0;
    float dI_wrt_b = pred_b < truth_tblr.bot ? Iw : 0;
    float dI_wrt_l = pred_l > truth_tblr.left ? (-1 * Ih) : 0;
    float dI_wrt_r = pred_r < truth_tblr.right ? Ih : 0;
    // derivative of U with regard to x
    float dU_wrt_t = dX_wrt_t - dI_wrt_t;
    float dU_wrt_b = dX_wrt_b - dI_wrt_b;
    float dU_wrt_l = dX_wrt_l - dI_wrt_l;
    float dU_wrt_r = dX_wrt_r - dI_wrt_r;
    // gradient of C min/max in IoU calc (prediction)
    float dC_wrt_t = pred_t < truth_tblr.top ? (-1 * giou_Cw) : 0;
    float dC_wrt_b = pred_b > truth_tblr.bot ? giou_Cw : 0;
    float dC_wrt_l = pred_l < truth_tblr.left ? (-1 * giou_Ch) : 0;
    float dC_wrt_r = pred_r > truth_tblr.right ? giou_Ch : 0;

    float p_dt = 0;
    float p_db = 0;
    float p_dl = 0;
    float p_dr = 0;
    if (U > 0 ) {
      p_dt = ((U * dI_wrt_t) - (I * dU_wrt_t)) / (U * U);
      p_db = ((U * dI_wrt_b) - (I * dU_wrt_b)) / (U * U);
      p_dl = ((U * dI_wrt_l) - (I * dU_wrt_l)) / (U * U);
      p_dr = ((U * dI_wrt_r) - (I * dU_wrt_r)) / (U * U);
    }
    // apply grad from prediction min/max for correct corner selection
    p_dt = pred_tblr.top < pred_tblr.bot ? p_dt : p_db;
    p_db = pred_tblr.top < pred_tblr.bot ? p_db : p_dt;
    p_dl = pred_tblr.left < pred_tblr.right ? p_dl : p_dr;
    p_dr = pred_tblr.left < pred_tblr.right ? p_dr : p_dl;

    if (iou_loss == GIOU) {
      if (giou_C > 0) {
        // apply "C" term from gIOU
        p_dt += ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
        p_db += ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
        p_dl += ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
        p_dr += ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
      }
	  if (Iw<=0||Ih<=0) {
		p_dt = ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
        p_db = ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
        p_dl = ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
        p_dr = ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
	  }
    }

    float Ct = fmin(pred.y - pred.h / 2,truth.y - truth.h / 2);
    float Cb = fmax(pred.y + pred.h / 2,truth.y + truth.h / 2);
    float Cl = fmin(pred.x - pred.w / 2,truth.x - truth.w / 2);
    float Cr = fmax(pred.x + pred.w / 2,truth.x + truth.w / 2);
    float Cw = Cr - Cl;
    float Ch = Cb - Ct;
    float C = Cw * Cw + Ch * Ch;

    float dCt_dx = 0;
    float dCt_dy = pred_t < truth_tblr.top ? 1 : 0;
    float dCt_dw = 0;
    float dCt_dh = pred_t < truth_tblr.top ? -0.5 : 0;

    float dCb_dx = 0;
    float dCb_dy = pred_b > truth_tblr.bot ? 1 : 0;
    float dCb_dw = 0;
    float dCb_dh = pred_b > truth_tblr.bot ? 0.5: 0;

    float dCl_dx = pred_l < truth_tblr.left ? 1 : 0;
    float dCl_dy = 0;
    float dCl_dw = pred_l < truth_tblr.left ? -0.5 : 0;
    float dCl_dh = 0;

    float dCr_dx = pred_r > truth_tblr.right ? 1 : 0;
    float dCr_dy = 0;
    float dCr_dw = pred_r > truth_tblr.right ? 0.5 : 0;
    float dCr_dh = 0;

    float dCw_dx = dCr_dx - dCl_dx;
    float dCw_dy = dCr_dy - dCl_dy;
    float dCw_dw = dCr_dw - dCl_dw;
    float dCw_dh = dCr_dh - dCl_dh;

    float dCh_dx = dCb_dx - dCt_dx;
    float dCh_dy = dCb_dy - dCt_dy;
    float dCh_dw = dCb_dw - dCt_dw;
    float dCh_dh = dCb_dh - dCt_dh;

    // UNUSED
    //// ground truth
    //float dI_wrt_xhat_t = pred_t < truth_tblr.top ? (-1 * Iw) : 0;
    //float dI_wrt_xhat_b = pred_b > truth_tblr.bot ? Iw : 0;
    //float dI_wrt_xhat_l = pred_l < truth_tblr.left ? (-1 * Ih) : 0;
    //float dI_wrt_xhat_r = pred_r > truth_tblr.right ? Ih : 0;

    // Final IOU loss (prediction) (negative of IOU gradient, we want the negative loss)
    float p_dx = 0;
    float p_dy = 0;
    float p_dw = 0;
    float p_dh = 0;

    p_dx = p_dl + p_dr;           //p_dx, p_dy, p_dw and p_dh are the gradient of IoU or GIoU.
    p_dy = p_dt + p_db;
    p_dw = (p_dr - p_dl);         //For dw and dh, we do not divided by 2.
    p_dh = (p_db - p_dt);

    // https://github.com/Zzh-tju/DIoU-darknet
    // https://arxiv.org/abs/1911.08287
    if (iou_loss == DIOU) {
        if (C > 0) {
            p_dx += (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
            p_dy += (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
            p_dw += (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C);
            p_dh += (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C);
        }
	if (Iw<=0||Ih<=0){
            p_dx = (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
            p_dy = (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
            p_dw = (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C);
            p_dh = (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C);
        }
    }
	//The following codes are calculating the gradient of ciou.

    if (iou_loss == CIOU) {
	float ar_gt = truth.w / truth.h;
        float ar_pred = pred.w / pred.h;
        float ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred));
	float alpha = ar_loss / (1 - I/U + ar_loss + 0.000001);
	float ar_dw=8/(M_PI*M_PI)*(atan(ar_gt)-atan(ar_pred))*pred.h;
        float ar_dh=-8/(M_PI*M_PI)*(atan(ar_gt)-atan(ar_pred))*pred.w;
        if (C > 0) {
        // dar*
            p_dx += (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
            p_dy += (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
            p_dw += (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C) + alpha * ar_dw;
            p_dh += (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C) + alpha * ar_dh;
        }
	if (Iw<=0||Ih<=0){
            p_dx = (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
            p_dy = (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
            p_dw = (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C) + alpha * ar_dw;
            p_dh = (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C) + alpha * ar_dh;
        }
    }

    ddx.dt = p_dx;      //We follow the original code released from GDarknet. So in yolo_layer.c, dt, db, dl, dr are already dx, dy, dw, dh.
    ddx.db = p_dy;
    ddx.dl = p_dw;
    ddx.dr = p_dh;

    // UNUSED
    //// ground truth
    //float gt_dt = ((U * dI_wrt_xhat_t) - (I * (dXhat_wrt_t - dI_wrt_xhat_t))) / (U * U);
    //float gt_db = ((U * dI_wrt_xhat_b) - (I * (dXhat_wrt_b - dI_wrt_xhat_b))) / (U * U);
    //float gt_dl = ((U * dI_wrt_xhat_l) - (I * (dXhat_wrt_l - dI_wrt_xhat_l))) / (U * U);
    //float gt_dr = ((U * dI_wrt_xhat_r) - (I * (dXhat_wrt_r - dI_wrt_xhat_r))) / (U * U);

    // no min/max grad applied
    //dx.dt = dt;
    //dx.db = db;
    //dx.dl = dl;
    //dx.dr = dr;

    //// sum in gt -- THIS DOESNT WORK
    //dx.dt += gt_dt;
    //dx.db += gt_db;
    //dx.dl += gt_dl;
    //dx.dr += gt_dr;

    //// instead, look at the change between pred and gt, and weight t/b/l/r appropriately...
    //// need the real derivative here (I think?)
    //float delta_t = fmax(truth_tblr.top, pred_t) - fmin(truth_tblr.top, pred_t);
    //float delta_b = fmax(truth_tblr.bot, pred_b) - fmin(truth_tblr.bot, pred_b);
    //float delta_l = fmax(truth_tblr.left, pred_l) - fmin(truth_tblr.left, pred_l);
    //float delta_r = fmax(truth_tblr.right, pred_r) - fmin(truth_tblr.right, pred_r);

    //dx.dt *= delta_t / (delta_t + delta_b);
    //dx.db *= delta_b / (delta_t + delta_b);
    //dx.dl *= delta_l / (delta_l + delta_r);
    //dx.dr *= delta_r / (delta_l + delta_r);

    // UNUSED
    //// ground truth
    //float gt_dt = ((U * dI_wrt_xhat_t) - (I * (dXhat_wrt_t - dI_wrt_xhat_t))) / (U * U);
    //float gt_db = ((U * dI_wrt_xhat_b) - (I * (dXhat_wrt_b - dI_wrt_xhat_b))) / (U * U);
    //float gt_dl = ((U * dI_wrt_xhat_l) - (I * (dXhat_wrt_l - dI_wrt_xhat_l))) / (U * U);
    //float gt_dr = ((U * dI_wrt_xhat_r) - (I * (dXhat_wrt_r - dI_wrt_xhat_r))) / (U * U);

    // no min/max grad applied
    //dx.dt = dt;
    //dx.db = db;
    //dx.dl = dl;
    //dx.dr = dr;

    // apply grad from prediction min/max for correct corner selection
    //dx.dt = pred_tblr.top < pred_tblr.bot ? p_dt : p_db;
    //dx.db = pred_tblr.top < pred_tblr.bot ? p_db : p_dt;
    //dx.dl = pred_tblr.left < pred_tblr.right ? p_dl : p_dr;
    //dx.dr = pred_tblr.left < pred_tblr.right ? p_dr : p_dl;

    //// sum in gt -- THIS DOESNT WORK
    //dx.dt += gt_dt;
    //dx.db += gt_db;
    //dx.dl += gt_dl;
    //dx.dr += gt_dr;

    //// instead, look at the change between pred and gt, and weight t/b/l/r appropriately...
    //// need the real derivative here (I think?)
    //float delta_t = fmax(truth_tblr.top, pred_t) - fmin(truth_tblr.top, pred_t);
    //float delta_b = fmax(truth_tblr.bot, pred_b) - fmin(truth_tblr.bot, pred_b);
    //float delta_l = fmax(truth_tblr.left, pred_l) - fmin(truth_tblr.left, pred_l);
    //float delta_r = fmax(truth_tblr.right, pred_r) - fmin(truth_tblr.right, pred_r);

    //dx.dt *= delta_t / (delta_t + delta_b);
    //dx.db *= delta_b / (delta_t + delta_b);
    //dx.dl *= delta_l / (delta_l + delta_r);
    //dx.dr *= delta_r / (delta_l + delta_r);

//#ifdef DEBUG_PRINTS
    /*printf("  directions dt: ");
    if ((pred_tblr.top < truth_tblr.top && dx.dt > 0) || (pred_tblr.top > truth_tblr.top && dx.dt < 0)) {
      printf("✓");
    } else {
      printf("𝒙");
    }
    printf(", ");
    if ((pred_tblr.bot < truth_tblr.bot && dx.db > 0) || (pred_tblr.bot > truth_tblr.bot && dx.db < 0)) {
      printf("✓");
    } else {
      printf("𝒙");
    }
    printf(", ");
    if ((pred_tblr.left < truth_tblr.left && dx.dl > 0) || (pred_tblr.left > truth_tblr.left && dx.dl < 0)) {
      printf("✓");
    } else {
      printf("𝒙");
    }
    printf(", ");
    if ((pred_tblr.right < truth_tblr.right && dx.dr > 0) || (pred_tblr.right > truth_tblr.right && dx.dr < 0)) {
      printf("✓");
    } else {
      printf("𝒙");
    }
    printf("\n");

    printf("dx dt:%f", dx.dt);
    printf(", db: %f", dx.db);
    printf(", dl: %f", dx.dl);
    printf(", dr: %f | ", dx.dr);
#endif

#ifdef DEBUG_NAN
    if (isnan(dx.dt)) { printf("dt isnan\n"); }
    if (isnan(dx.db)) { printf("db isnan\n"); }
    if (isnan(dx.dl)) { printf("dl isnan\n"); }
    if (isnan(dx.dr)) { printf("dr isnan\n"); }
#endif

//    // No update if 0 or nan
//    if (dx.dt == 0 || isnan(dx.dt)) { dx.dt = 1; }
//    if (dx.db == 0 || isnan(dx.db)) { dx.db = 1; }
//    if (dx.dl == 0 || isnan(dx.dl)) { dx.dl = 1; }
//    if (dx.dr == 0 || isnan(dx.dr)) { dx.dr = 1; }
//
//#ifdef DEBUG_PRINTS
//    printf("dx dt:%f (t: %f, p: %f)", dx.dt, gt_dt, p_dt);
//    printf(", db: %f (t: %f, p: %f)", dx.db, gt_db, p_db);
//    printf(", dl: %f (t: %f, p: %f)", dx.dl, gt_dl, p_dl);
//    printf(", dr: %f (t: %f, p: %f) | ", dx.dr, gt_dr, p_dr);
//#endif */
    return ddx;
}

static inline float fix_nan_inf(float val)
{
    if (isnan(val) || isinf(val)) val = 0;
    return val;
}

static inline float clip_value(float val, const float max_val)
{
    if (val > max_val) {
        //printf("\n val = %f > max_val = %f \n", val, max_val);
        val = max_val;
    }
    else if (val < -max_val) {
        //printf("\n val = %f < -max_val = %f \n", val, -max_val);
        val = -max_val;
    }
    return val;
}

ious delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride, float iou_normalizer, IOU_LOSS iou_loss, int accumulate, float max_delta)
{
    ious all_ious = { 0 };
    // i - step in layer width
    // j - step in layer height
    //  Returns a box in absolute coordinates
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    all_ious.iou = box_iou(pred, truth);
    all_ious.giou = box_giou(pred, truth);
    all_ious.diou = box_diou(pred, truth);
    all_ious.ciou = box_ciou(pred, truth);
    // avoid nan in dx_box_iou
    if (pred.w == 0) { pred.w = 1.0; }
    if (pred.h == 0) { pred.h = 1.0; }
    if (iou_loss == MSE)    // old loss
    {
        float tx = (truth.x*lw - i);
        float ty = (truth.y*lh - j);
        float tw = log(truth.w*w / biases[2 * n]);
        float th = log(truth.h*h / biases[2 * n + 1]);

        // accumulate delta
        delta[index + 0 * stride] += scale * (tx - x[index + 0 * stride]) * iou_normalizer;
        delta[index + 1 * stride] += scale * (ty - x[index + 1 * stride]) * iou_normalizer;
        delta[index + 2 * stride] += scale * (tw - x[index + 2 * stride]) * iou_normalizer;
        delta[index + 3 * stride] += scale * (th - x[index + 3 * stride]) * iou_normalizer;
    }
    else {
        // https://github.com/generalized-iou/g-darknet
        // https://arxiv.org/abs/1902.09630v2
        // https://giou.stanford.edu/
        all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);

        // jacobian^t (transpose)
        //float dx = (all_ious.dx_iou.dl + all_ious.dx_iou.dr);
        //float dy = (all_ious.dx_iou.dt + all_ious.dx_iou.db);
        //float dw = ((-0.5 * all_ious.dx_iou.dl) + (0.5 * all_ious.dx_iou.dr));
        //float dh = ((-0.5 * all_ious.dx_iou.dt) + (0.5 * all_ious.dx_iou.db));

        // jacobian^t (transpose)
        float dx = all_ious.dx_iou.dt;
        float dy = all_ious.dx_iou.db;
        float dw = all_ious.dx_iou.dl;
        float dh = all_ious.dx_iou.dr;

        // predict exponential, apply gradient of e^delta_t ONLY for w,h
        dw *= exp(x[index + 2 * stride]);
        dh *= exp(x[index + 3 * stride]);

        // normalize iou weight
        dx *= iou_normalizer;
        dy *= iou_normalizer;
        dw *= iou_normalizer;
        dh *= iou_normalizer;


        dx = fix_nan_inf(dx);
        dy = fix_nan_inf(dy);
        dw = fix_nan_inf(dw);
        dh = fix_nan_inf(dh);

        if (max_delta != FLT_MAX) {
            dx = clip_value(dx, max_delta);
            dy = clip_value(dy, max_delta);
            dw = clip_value(dw, max_delta);
            dh = clip_value(dh, max_delta);
        }


        if (!accumulate) {
            delta[index + 0 * stride] = 0;
            delta[index + 1 * stride] = 0;
            delta[index + 2 * stride] = 0;
            delta[index + 3 * stride] = 0;
        }

        // accumulate delta
        delta[index + 0 * stride] += dx;
        delta[index + 1 * stride] += dy;
        delta[index + 2 * stride] += dw;
        delta[index + 3 * stride] += dh;
    }

    return all_ious;
}

/*
float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}*/


void averages_yolo_deltas(int class_index, int box_index, int stride, int classes, float *delta)
{

    int classes_in_one_box = 0;
    int c;
    for (c = 0; c < classes; ++c) {
        if (delta[class_index + stride*c] > 0) classes_in_one_box++;
    }

    if (classes_in_one_box > 0) {
        delta[box_index + 0 * stride] /= classes_in_one_box;
        delta[box_index + 1 * stride] /= classes_in_one_box;
        delta[box_index + 2 * stride] /= classes_in_one_box;
        delta[box_index + 3 * stride] /= classes_in_one_box;
    }
}

void delta_yolo_class(float *output, float *delta, int index, int class_id, int classes, int stride, float *avg_cat, int focal_loss, float label_smooth_eps, float *classes_multipliers)
{
    int n;
    if (delta[index + stride*class_id]){
        delta[index + stride*class_id] = (1 - label_smooth_eps) - output[index + stride*class_id];
        if (classes_multipliers) delta[index + stride*class_id] *= classes_multipliers[class_id];
        if(avg_cat) *avg_cat += output[index + stride*class_id];
        return;
    }
    // Focal loss
    if (focal_loss) {
        // Focal Loss
        float alpha = 0.5;    // 0.25 or 0.5
        //float gamma = 2;    // hardcoded in many places of the grad-formula

        int ti = index + stride*class_id;
        float pt = output[ti] + 0.000000000000001F;
        // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
        float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
        //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = (((n == class_id) ? 1 : 0) - output[index + stride*n]);

            delta[index + stride*n] *= alpha*grad;

            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
    else {
        // default
        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = ((n == class_id) ? (1 - label_smooth_eps) : (0 + label_smooth_eps/classes)) - output[index + stride*n];
            if (classes_multipliers && n == class_id) delta[index + stride*class_id] *= classes_multipliers[class_id];
            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
}

/*
void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat)
{
    int n;
    if (delta[index]){
        delta[index + stride*class] = 1 - output[index + stride*class];
        if(avg_cat) *avg_cat += output[index + stride*class];
        return;
    }
    for(n = 0; n < classes; ++n){
        delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
        if(n == class && avg_cat) *avg_cat += output[index + stride*n];
    }
}*/

int compare_yolo_class(float *output, int classes, int class_index, int stride, float objectness, int class_id, float conf_thresh)
{
    int j;
    for (j = 0; j < classes; ++j) {
        //float prob = objectness * output[class_index + stride*j];
        float prob = output[class_index + stride*j];
        if (prob > conf_thresh) {
            return 1;
        }
    }
    return 0;
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

/*void forward_yolo_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
                    float best_iou = 0;
                    int best_t = 0;
                    for(t = 0; t < l.max_boxes; ++t){
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = 0 - l.output[obj_index];
                    if (best_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0;
                    }
                    if (best_iou > l.truth_thresh) {
                        l.delta[obj_index] = 1 - l.output[obj_index];

                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class = l.map[class];
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                    }
                }
            }
        }
        for(t = 0; t < l.max_boxes; ++t){
            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for(n = 0; n < l.total; ++n){
                box pred = {0};
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);
            if(mask_n >= 0){
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);

                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = 1 - l.output[obj_index];

                int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class = l.map[class];
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %4d Avg IOU: %9.4f, Class: %9.4f, Obj: %9.4f, No Obj: %9.4f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}*/

int check_mistakes=0;
box float_to_box_stride(float *f, int stride)
{
    box b = { 0 };
    b.x = f[0];
    b.y = f[1 * stride];
    b.w = f[2 * stride];
    b.h = f[3 * stride];
    return b;
}

void scal_add_cpu(int N, float ALPHA, float BETA, float *X, int INCX)
{
    int i;
    for (i = 0; i < N; ++i) X[i*INCX] = X[i*INCX] * ALPHA + BETA;
}

void forward_yolo_layer(const layer l, network state)
{
    int i, j, b, t, n;
    memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);        // x,y,
            scal_add_cpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output + index, 1);    // scale x,y
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    // delta is zeroed
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if (!state.train) return;
    //float avg_iou = 0;
    float tot_iou = 0;
    float tot_giou = 0;
    float tot_diou = 0;
    float tot_ciou = 0;
    float tot_iou_loss = 0;
    float tot_giou_loss = 0;
    float tot_diou_loss = 0;
    float tot_ciou_loss = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.w, state.h, l.w*l.h);
                    float best_match_iou = 0;
                    float best_iou = 0;
                    int best_t = 0;
                    for (t = 0; t < l.max_boxes; ++t) {
                        box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
                        int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
                        if (class_id >= l.classes) {
                            printf("\n Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes - 1);
                            printf("\n truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f, class_id = %d \n", truth.x, truth.y, truth.w, truth.h, class_id);
                            if (check_mistakes) getchar();
                            continue; // if label contains class_id more than number of classes in the cfg-file
                        }
                        if (!truth.x) break;  // continue;

                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                        float objectness = l.output[obj_index];
                        int class_id_match = compare_yolo_class(l.output, l.classes, class_index, l.w*l.h, objectness, class_id, 0.25f);

                        float iou = box_iou(pred, truth);
                        if (iou > best_match_iou && class_id_match == 1) {
                            best_match_iou = iou;
                        }
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = l.cls_normalizer * (0 - l.output[obj_index]);
                    if (best_match_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0;
                    }
                    if (best_iou > l.truth_thresh) {
                        l.delta[obj_index] = l.cls_normalizer * (1 - l.output[obj_index]);

                        int class_id = state.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class_id = l.map[class_id];
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, 0, l.focal_loss, l.label_smooth_eps, l.classes_multipliers);
                        box truth = float_to_box_stride(state.truth + best_t*(4 + 1) + b*l.truths, 1);
                        const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.w, state.h, l.delta, (2 - truth.w*truth.h), l.w*l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta);
                    }
                }
            }
        }
        for (t = 0; t < l.max_boxes; ++t) {
            box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
            if (truth.x < 0 || truth.y < 0 || truth.x > 1 || truth.y > 1 || truth.w < 0 || truth.h < 0) {
                char buff[256];
                printf(" Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f \n", truth.x, truth.y, truth.w, truth.h);
                sprintf(buff, "echo \"Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f\" >> bad_label.list",
                    truth.x, truth.y, truth.w, truth.h);
                system(buff);
            }
            int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
            if (class_id >= l.classes) continue; // if label contains class_id more than number of classes in the cfg-file

            if (!truth.x) break;  // continue;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for (n = 0; n < l.total; ++n) {
                box pred = { 0 };
                pred.w = l.biases[2 * n] / state.w;
                pred.h = l.biases[2 * n + 1] / state.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);
            if (mask_n >= 0) {
                int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class_id = l.map[class_id];

                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                ious all_ious = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.w, state.h, l.delta, (2 - truth.w*truth.h), l.w*l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta);

                // range is 0 <= 1
                tot_iou += all_ious.iou;
                tot_iou_loss += 1 - all_ious.iou;
                // range is -1 <= giou <= 1
                tot_giou += all_ious.giou;
                tot_giou_loss += 1 - all_ious.giou;

                tot_diou += all_ious.diou;
                tot_diou_loss += 1 - all_ious.diou;

                tot_ciou += all_ious.ciou;
                tot_ciou_loss += 1 - all_ious.ciou;

                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = class_multiplier * l.cls_normalizer * (1 - l.output[obj_index]);

                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers);

                ++count;
                ++class_count;
                if (all_ious.iou > .5) recall += 1;
                if (all_ious.iou > .75) recall75 += 1;
            }

            // iou_thresh
            for (n = 0; n < l.total; ++n) {
                int mask_n = int_index(l.mask, n, l.n);
                if (mask_n >= 0 && n != best_n && l.iou_thresh < 1.0f) {
                    box pred = { 0 };
                    pred.w = l.biases[2 * n] / state.w;
                    pred.h = l.biases[2 * n + 1] / state.h;
                    float iou = box_iou(pred, truth_shift);
                    // iou, n

                    if (iou > l.iou_thresh) {
                        int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class_id = l.map[class_id];

                        int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                        const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                        ious all_ious = delta_yolo_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, state.w, state.h, l.delta, (2 - truth.w*truth.h), l.w*l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta);

                        // range is 0 <= 1
                        tot_iou += all_ious.iou;
                        tot_iou_loss += 1 - all_ious.iou;
                        // range is -1 <= giou <= 1
                        tot_giou += all_ious.giou;
                        tot_giou_loss += 1 - all_ious.giou;

                        tot_diou += all_ious.diou;
                        tot_diou_loss += 1 - all_ious.diou;

                        tot_ciou += all_ious.ciou;
                        tot_ciou_loss += 1 - all_ious.ciou;

                        int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                        avg_obj += l.output[obj_index];
                        l.delta[obj_index] = class_multiplier * l.cls_normalizer * (1 - l.output[obj_index]);

                        int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers);

                        ++count;
                        ++class_count;
                        if (all_ious.iou > .5) recall += 1;
                        if (all_ious.iou > .75) recall75 += 1;
                    }
                }
            }
        }

        // averages the deltas obtained by the function: delta_yolo_box()_accumulate
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                    const int stride = l.w*l.h;

                    averages_yolo_deltas(class_index, box_index, stride, l.classes, l.delta);
                }
            }
        }
    }

    //*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    //printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", state.index, avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count);

    int stride = l.w*l.h;
    float* no_iou_loss_delta = (float *)calloc(l.batch * l.outputs, sizeof(float));
    memcpy(no_iou_loss_delta, l.delta, l.batch * l.outputs * sizeof(float));
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    no_iou_loss_delta[index + 0 * stride] = 0;
                    no_iou_loss_delta[index + 1 * stride] = 0;
                    no_iou_loss_delta[index + 2 * stride] = 0;
                    no_iou_loss_delta[index + 3 * stride] = 0;
                }
            }
        }
    }
    float classification_loss = l.cls_normalizer * pow(mag_array(no_iou_loss_delta, l.outputs * l.batch), 2);
    free(no_iou_loss_delta);
    float loss = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    float iou_loss = loss - classification_loss;

    float avg_iou_loss = 0;
    // gIOU loss + MSE (objectness) loss
    if (l.iou_loss == MSE) {
        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    }
    else {
        // Always compute classification loss both for iou + cls loss and for logging with mse loss
        // TODO: remove IOU loss fields before computing MSE on class
        //   probably split into two arrays
        if (l.iou_loss == GIOU) {
            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_giou_loss / count) : 0;
        }
        else {
            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_iou_loss / count) : 0;
        }
        *(l.cost) = avg_iou_loss + classification_loss;
    }

    loss /= l.batch;
    classification_loss /= l.batch;
    iou_loss /= l.batch;

    printf("Region %3d Avg Class: %9.6f Obj: %8.6f No Obj: %6.4f .5R: %6.4f .75R: %6.4f count:%3d class_loss=%.4f\tiou_loss=%.4f\n",
        state.index, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count,
        classification_loss, iou_loss);
    /*fprintf(stderr, "v3 (%s loss, Normalizer: (iou: %.2f, cls: %.2f) Region %d Avg (IOU: %f, GIOU: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d, class_loss = %f, iou_loss = %f, total_loss = %f \n",
        (l.iou_loss == MSE ? "mse" : (l.iou_loss == GIOU ? "giou" : "iou")), l.iou_normalizer, l.cls_normalizer, state.index, tot_iou / count, tot_giou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count,
        classification_loss, iou_loss, loss);*/
}

void backward_yolo_layer(const layer l, network net)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

#ifdef GPU

/*void forward_yolo_layer_gpu(const layer l, network_state state)
{
    //copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
    simple_copy_ongpu(l.batch*l.inputs, state.input, l.output_gpu);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            // y = 1./(1. + exp(-x))
            // x = ln(y/(1-y))  // ln - natural logarithm (base = e)
            // if(y->1) x -> inf
            // if(y->0) x -> -inf
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);    // x,y
            if (l.scale_x_y != 1) scal_add_ongpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output_gpu + index, 1);      // scale x,y
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC); // classes and objectness
        }
    }
    if(!state.train || l.onlyforward){
        //cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        cuda_pull_array_async(l.output_gpu, l.output, l.batch*l.outputs);
        CHECK_CUDA(cudaPeekAtLastError());
        return;
    }

    float *in_cpu = (float *)xcalloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    memcpy(in_cpu, l.output, l.batch*l.outputs*sizeof(float));
    float *truth_cpu = 0;
    if (state.truth) {
        int num_truth = l.batch*l.truths;
        truth_cpu = (float *)xcalloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.net = state.net;
    cpu_state.index = state.index;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_yolo_layer(l, cpu_state);
    //forward_yolo_layer(l, state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    free(in_cpu);
    if (cpu_state.truth) free(cpu_state.truth);
}*/ // AlexeyAB's yolo needs to change for cpu version state:-)

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

