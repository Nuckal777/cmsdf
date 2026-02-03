// SPDX-License-Identifier: BSD-3-Clause

#include "cmsdf.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freetype/ftimage.h"
#include "freetype/ftoutln.h"
#include "freetype/fttypes.h"

typedef struct {
    double x;
    double y;
} vec2;

static vec2 splat2(double v) {
    return (vec2){.x = v, .y = v};
}

static vec2 vec2_add(vec2 a, vec2 b) {
    return (vec2){.x = a.x + b.x, .y = a.y + b.y};
}

static vec2 vec2_sub(vec2 a, vec2 b) {
    return (vec2){.x = a.x - b.x, .y = a.y - b.y};
}

static vec2 vec2_mult(vec2 a, vec2 b) {
    return (vec2){.x = a.x * b.x, .y = a.y * b.y};
}

static double vec2_dot(vec2 a, vec2 b) {
    return a.x * b.x + a.y * b.y;
}

static double vec2_cross(vec2 a, vec2 b) {
    return a.x * b.y - a.y * b.x;
}

static double vec2_dist(vec2 a, vec2 b) {
    vec2 diff = vec2_sub(a, b);
    return sqrt(diff.x * diff.x + diff.y * diff.y);
}

static vec2 vec2_normalize(vec2 a) {
    double len = sqrt(a.x * a.x + a.y * a.y);
    return (vec2){.x = a.x / len, .y = a.y / len};
}

#define EDGE_TY_LINE 1
#define EDGE_TY_CONIC 2
#define EDGE_TY_CUBIC 3

#define EDGE_COLOR_CYAN (255 | 255 << 8)
#define EDGE_COLOR_MAGENTA (255 | 255 << 16)
#define EDGE_COLOR_YELLOW (255 << 8 | 255 << 16)
#define EDGE_COLOR_WHITE (255 | 255 << 8 | 255 << 16)

struct cmsdf_edge {
    vec2 start;
    vec2 end;
    vec2 control1;
    vec2 control2;
    uint32_t color;
};

static uint8_t cmsdf_edge_type(cmsdf_edge* e) {
    if (!isnan(e->control2.x)) {
        return EDGE_TY_CUBIC;
    }
    if (!isnan(e->control1.x)) {
        return EDGE_TY_CONIC;
    }
    return EDGE_TY_LINE;
}

static cmsdf_edge make_line(vec2 start, vec2 end) {
    return (cmsdf_edge){
        .start = start,
        .end = end,
        .control1 = (vec2){.x = NAN, .y = NAN},
        .control2 = (vec2){.x = NAN, .y = NAN},
        .color = 0,
    };
}

static cmsdf_edge make_conic(vec2 start, vec2 end, vec2 control1) {
    return (cmsdf_edge){
        .start = start,
        .end = end,
        .control1 = control1,
        .control2 = (vec2){.x = NAN, .y = NAN},
        .color = 0,
    };
}

static cmsdf_edge make_cubic(vec2 start, vec2 end, vec2 control1, vec2 control2) {
    return (cmsdf_edge){
        .start = start,
        .end = end,
        .control1 = control1,
        .control2 = control2,
        .color = 0,
    };
}

static vec2 cmsdf_edge_at(cmsdf_edge* e, double t) {
    uint8_t ty = cmsdf_edge_type(e);
    switch (ty) {
        case EDGE_TY_LINE:
            return vec2_add(e->start, vec2_mult(vec2_sub(e->end, e->start), splat2(t)));
        case EDGE_TY_CONIC: {
            vec2 p0 = vec2_mult(e->start, splat2((1 - t) * (1 - t)));
            vec2 p1 = vec2_mult(e->control1, splat2(2 * t * (1 - t)));
            vec2 p2 = vec2_mult(e->end, splat2(t * t));
            return vec2_add(p0, vec2_add(p1, p2));
        }
        case EDGE_TY_CUBIC: {
            double one_sub_t = 1 - t;
            double one_sub_t_sq = one_sub_t * one_sub_t;
            double one_sub_t_cub = one_sub_t_sq * one_sub_t;
            vec2 p0 = vec2_mult(e->start, splat2(one_sub_t_cub));
            vec2 p1 = vec2_mult(e->control1, splat2(3 * one_sub_t_sq * t));
            vec2 p2 = vec2_mult(e->control2, splat2(3 * one_sub_t * t * t));
            vec2 p3 = vec2_mult(e->end, splat2(t * t * t));
            return vec2_add(vec2_add(p0, p1), vec2_add(p2, p3));
        }
    }
    assert(0);
    return (vec2){};
}

static vec2 cmsdf_edge_dir(cmsdf_edge* e, double t) {
    uint8_t ty = cmsdf_edge_type(e);
    switch (ty) {
        case EDGE_TY_LINE:
            return vec2_sub(e->end, e->start);
        case EDGE_TY_CONIC: {
            vec2 d0 = vec2_mult(vec2_sub(e->control1, e->start), splat2(2 * (1 - t)));
            vec2 d1 = vec2_mult(vec2_sub(e->end, e->control1), splat2(2 * t));
            return vec2_add(d0, d1);
        }
        case EDGE_TY_CUBIC: {
            double one_sub_t = 1 - t;
            double one_sub_t_sq = one_sub_t * one_sub_t;
            vec2 d0 = vec2_mult(vec2_sub(e->control1, e->start), splat2(3 * one_sub_t_sq));
            vec2 d1 = vec2_mult(vec2_sub(e->control2, e->control1), splat2(6 * one_sub_t * t));
            vec2 d2 = vec2_mult(vec2_sub(e->end, e->control2), splat2(3 * t * t));
            return vec2_add(d0, vec2_add(d1, d2));
        }
    }
    assert(0);
    return (vec2){};
}

static int approx_eq_abs(double a, double b, double tolerance) {
    assert(tolerance >= 0);
    if (a == b) {
        return 1;
    }
    if (isnan(a) || isnan(b)) {
        return 0;
    }
    return fabs(a - b) <= tolerance;
}

static size_t solveCubic(double coeff[4], double* out) {
    double a = coeff[0];
    double b = coeff[1];
    double c = coeff[2];
    double d = coeff[3];
    assert(a != 0.0);

    // Depressed cubic transformation
    double p = (3.0 * a * c - b * b) / (3.0 * a * a);
    double q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);
    double discriminant = (q * q / 4.0) + (p * p * p / 27.0);
    double offset = b / (3.0 * a);

    if (approx_eq_abs(discriminant, 0, 1e-12)) {
        double u = cbrt(-q / 2.0);
        out[0] = 2 * u - offset;
        out[1] = -u - offset;
        return 2;
    }
    if (discriminant > 0) {
        double sqrt_discriminant = sqrt(discriminant);
        double u = cbrt(-q / 2.0 + sqrt_discriminant);
        double v = cbrt(-q / 2.0 - sqrt_discriminant);
        double t = u + v;
        out[0] = t - offset;
        return 1;
    }
    double r = sqrt(-p / 3.0);
    double theta = acos(-q / (2.0 * pow(r, 3.0)));
    double two_r = 2.0 * r;

    out[0] = two_r * cos(theta / 3.0) - offset;
    out[1] = two_r * cos((theta + 2.0 * M_PI) / 3.0) - offset;
    out[2] = two_r * cos((theta + 4.0 * M_PI) / 3.0) - offset;
    return 3;
}

static double min_dist_sq(cmsdf_edge* e, double t, vec2 point) {
    vec2 on_curve = cmsdf_edge_at(e, t);
    vec2 tangent = cmsdf_edge_dir(e, t);
    vec2 to_point = vec2_sub(on_curve, point);
    return 2 * vec2_dot(to_point, tangent);
}

static double cmsdf_edge_arg_min_dist(cmsdf_edge* e, vec2 point) {
    uint8_t ty = cmsdf_edge_type(e);
    switch (ty) {
        case EDGE_TY_LINE: {
            vec2 dir = vec2_sub(e->end, e->start);
            double numerator = vec2_dot(vec2_sub(point, e->start), dir);
            double denominator = vec2_dot(dir, dir);
            double t = numerator / denominator;
            if (t < 0) {
                return 0;
            }
            return t > 1 ? 1 : t;
        }
        case EDGE_TY_CONIC: {
            vec2 p0 = vec2_sub(point, e->start);
            vec2 p1 = vec2_sub(e->control1, e->start);
            vec2 p2 = vec2_sub(vec2_add(e->start, e->end), vec2_mult(e->control1, splat2(2)));

            double coeff[4];
            coeff[0] = vec2_dot(p2, p2);
            coeff[1] = 3.0 * vec2_dot(p1, p2);
            coeff[2] = vec2_dot(vec2_mult(splat2(2), p1), p1) - vec2_dot(p2, p0);
            coeff[3] = -vec2_dot(p1, p0);

            double ts[5];
            size_t ts_len = solveCubic(coeff, ts);
            ts[ts_len] = 0;
            ts[ts_len + 1] = 1;
            ts_len += 2;
            double min_dist = INFINITY;
            double min_arg = 1;
            for (size_t i = 0; i < ts_len; i++) {
                double t = ts[i];
                if (isnan(t) || t < 0.0 || t > 1.0) {
                    continue;
                }
                vec2 on_curve = cmsdf_edge_at(e, t);
                double dist = vec2_dist(on_curve, point);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_arg = t;
                }
            }
            return min_arg;
        }
        case EDGE_TY_CUBIC: {
            double t = 0;
            double min_dist = vec2_dist(e->start, point);
            double end_dist = vec2_dist(e->end, point);
            if (end_dist < min_dist) {
                min_dist = end_dist;
                t = 1;
            }
            for (size_t i = 0; i < 4; i++) {
                double x0 = 1.0 / 4.0 * (double)i;
                double x1 = 1.0 / 4.0 * (double)(i + 1);
                double epsilon = 1e-6;
                for (size_t j = 0; j < 16; j++) {  // secant method
                    double fx0 = min_dist_sq(e, x0, point);
                    double fx1 = min_dist_sq(e, x1, point);
                    if (fabs(fx1) < epsilon) {
                        break;
                    }
                    if (fx1 - fx0 == 0) {
                        continue;
                    }
                    double next = x1 - fx1 * ((x1 - x0) / (fx1 - fx0));
                    x0 = x1;
                    x1 = next;
                }
                if (x1 < 0 || x1 > 1) {
                    continue;
                }
                double d = vec2_dist(cmsdf_edge_at(e, x1), point);
                if (d < min_dist) {
                    min_dist = d;
                    t = x1;
                }
            }
            return t;
        }
    }
    assert(0);
    return NAN;
}

typedef struct {
    double dist;
    double orthogonality;
    double sign;
} edge_point_stats;

static bool edge_point_stats_eq(edge_point_stats a, edge_point_stats b) {
    return a.dist == b.dist && a.orthogonality == b.orthogonality;
}

static edge_point_stats edge_point_stats_min(edge_point_stats a, edge_point_stats b) {
    double diff_dist = a.dist - b.dist;
    double epsilon = 1e-9;
    if (diff_dist > epsilon) {
        return b;
    }
    if (diff_dist < -epsilon) {
        return a;
    }
    return a.orthogonality > b.orthogonality ? a : b;
}

static edge_point_stats cmsdf_edge_dist_ortho(cmsdf_edge* e, vec2 point) {
    double arg_min = cmsdf_edge_arg_min_dist(e, point);
    vec2 at = cmsdf_edge_at(e, arg_min);
    double dist = vec2_dist(at, point);

    vec2 dir = cmsdf_edge_dir(e, arg_min);
    vec2 diff = vec2_sub(point, at);
    double ortho = fabs(vec2_cross(vec2_normalize(dir), vec2_normalize(diff)));
    double sign = vec2_cross(dir, vec2_sub(at, point));

    return (edge_point_stats){.dist = dist, .orthogonality = ortho, .sign = sign};
}

#define EDGE_ARRAY_DEFAULT_CAP 32

int cmsdf_edge_array_new(cmsdf_edge_array* arr) {
    arr->data = malloc(sizeof(cmsdf_edge) * EDGE_ARRAY_DEFAULT_CAP);
    if (!arr->data) {
        return CMSDF_ERR_OOM;
    }
    arr->cap = EDGE_ARRAY_DEFAULT_CAP;
    arr->len = 0;
    return 0;
}

void cmsdf_edge_array_free(cmsdf_edge_array* arr) {
    free(arr->data);
    arr->data = NULL;
    arr->cap = 0;
    arr->len = 0;
}

static int cmsdf_edge_array_push(cmsdf_edge_array* arr, cmsdf_edge e) {
    if (arr->len == arr->cap - 1) {
        size_t new_cap = arr->cap * 3 / 2;
        cmsdf_edge* tmp = realloc(arr->data, sizeof(cmsdf_edge) * new_cap);
        if (!tmp) {
            free(arr->data);
            return CMSDF_ERR_OOM;
        }
        arr->data = tmp;
        arr->cap = new_cap;
    }
    arr->data[arr->len] = e;
    arr->len++;
    return 0;
}

void cmsdf_edge_array_print(const cmsdf_edge_array* arr) {
    for (size_t i = 0; i < arr->len; i++) {
        cmsdf_edge* current = arr->data + i;
        printf("start: (%g, %g) end: (%g, %g)\n", current->start.x, current->start.y, current->end.x, current->end.y);
    }
}

static void vec2_min_max(vec2 p, vec2* min, vec2* max) {
    min->x = p.x < min->x ? p.x : min->x;
    min->y = p.y < min->y ? p.y : min->y;
    max->x = p.x > max->x ? p.x : max->x;
    max->y = p.y > max->y ? p.y : max->y;
}

static void cmsdf_edge_array_fit_to_grid(cmsdf_edge_array edges, vec2 dim) {
    vec2 min = {.x = INFINITY, .y = INFINITY};
    vec2 max = {.x = -INFINITY, .y = -INFINITY};
    for (size_t i = 0; i < edges.len; i++) {
        vec2_min_max(edges.data[i].start, &min, &max);
        vec2_min_max(edges.data[i].end, &min, &max);
    }
    vec2 len = vec2_sub(max, min);
    vec2 scale = {.x = (dim.x - 2) / len.x, .y = (dim.y - 2) / len.y};
    for (size_t i = 0; i < edges.len; i++) {
        cmsdf_edge* e = edges.data + i;
        e->start = vec2_add(vec2_mult(vec2_sub(e->start, min), scale), splat2(1));
        e->end = vec2_add(vec2_mult(vec2_sub(e->end, min), scale), splat2(1));
        e->control1 = vec2_add(vec2_mult(vec2_sub(e->control1, min), scale), splat2(1));
        e->control2 = vec2_add(vec2_mult(vec2_sub(e->control2, min), scale), splat2(1));
    }
}

typedef struct {
    vec2 start;
    cmsdf_edge_array* edges;
    cmsdf_contour_index contour_idx;
} decompose_ctx;

static vec2 from_ft_vec(FT_Vector v) {
    return (vec2){.x = v.x / 64.0, .y = v.y / 64.0};
}

static int decompose_move_to(const FT_Vector* to, void* user) {
    decompose_ctx* ctx = (decompose_ctx*)user;
    ctx->start = from_ft_vec(*to);
    if (ctx->contour_idx.len == CMSDF_CONTOUR_INDEX_CAP - 1) {
        return CMSDF_ERR_OOM;
    }
    ctx->contour_idx.offsets[ctx->contour_idx.len] = ctx->edges->len;
    ctx->contour_idx.len++;
    return 0;
}

static int decompose_line_to(const FT_Vector* to, void* user) {
    decompose_ctx* ctx = (decompose_ctx*)user;
    int err = cmsdf_edge_array_push(ctx->edges, make_line(ctx->start, from_ft_vec(*to)));
    ctx->start = from_ft_vec(*to);
    return err;
}

static int decompose_conic_to(const FT_Vector* control, const FT_Vector* to, void* user) {
    decompose_ctx* ctx = (decompose_ctx*)user;
    int err = cmsdf_edge_array_push(ctx->edges, make_conic(ctx->start, from_ft_vec(*to), from_ft_vec(*control)));
    ctx->start = from_ft_vec(*to);
    return err;
}

static int decompose_cubic_to(const FT_Vector* control1, const FT_Vector* control2, const FT_Vector* to, void* user) {
    decompose_ctx* ctx = (decompose_ctx*)user;
    int err = cmsdf_edge_array_push(ctx->edges, make_cubic(ctx->start, from_ft_vec(*to), from_ft_vec(*control1), from_ft_vec(*control2)));
    ctx->start = from_ft_vec(*to);
    return err;
}

int cmsdf_decompose(const cmsdf_decompose_params* params, cmsdf_decompose_result* result) {
    if (!params) {
        return CMSDF_ERR_FACE_MISSING_GLYPH;
    }
    FT_UInt idx = FT_Get_Char_Index(params->face, params->character);
    if (!idx) {
        return CMSDF_ERR_FACE_MISSING_GLYPH;
    }
    FT_Error err = FT_Load_Glyph(params->face, idx, FT_LOAD_NO_SCALE);
    if (err) {
        return err;
    }
    if (params->face->glyph->format != FT_GLYPH_FORMAT_OUTLINE) {
        return CMSDF_ERR_FACE_NO_OUTLINE;
    }
    FT_Outline* outline = &params->face->glyph->outline;
    FT_Outline_Funcs funcs = {
        .move_to = decompose_move_to,
        .line_to = decompose_line_to,
        .conic_to = decompose_conic_to,
        .cubic_to = decompose_cubic_to,
        .delta = 0,
        .shift = 0,
    };
    decompose_ctx ctx;
    ctx.contour_idx.len = 0;
    cmsdf_edge_array tmp_edges;
    if (params->arr) {
        ctx.edges = params->arr;
    } else {
        err = cmsdf_edge_array_new(&tmp_edges);
        if (err) {
            return err;
        }
        ctx.edges = &tmp_edges;
    }
    err = FT_Outline_Decompose(outline, &funcs, &ctx);
    if (err) {
        goto cleanup;
    }
    result->edges = *ctx.edges;
    result->contour_idx = ctx.contour_idx;

    result->rec = (cmsdf_rec){.width = params->pixel_width, .height = params->pixel_height};

    cmsdf_edge_array_fit_to_grid(result->edges, (vec2){.x = params->pixel_width, .y = params->pixel_height});
    for (size_t i = 0; i < result->contour_idx.len; i++) {
        cmsdf_edge* start = result->edges.data + result->contour_idx.offsets[i];
        cmsdf_edge* end = NULL;
        if (i + 1 == result->contour_idx.len) {
            end = result->edges.data + result->edges.len;
        } else {
            end = result->edges.data + result->contour_idx.offsets[i + 1];
        }
        if (end - start == 1) {
            start->color = EDGE_COLOR_WHITE;
            continue;
        }
        uint32_t color = EDGE_COLOR_MAGENTA;
        const uint32_t check = EDGE_COLOR_YELLOW;
        for (cmsdf_edge* current = start; current != end; current++) {
            current->color = color;
            if (color == check) {
                color = EDGE_COLOR_CYAN;
            } else {
                color = check;
            }
        }
    }
cleanup:
    if (err) {
        cmsdf_edge_array_free(ctx.edges);
    }
    return err;
}

#define RASTER_COLOR_SCALE 16.0

static double clamp(double a, double min, double max) {
    if (a > max) {
        return max;
    }
    return a < min ? min : a;
}

size_t cmsdf_raster_edges(const cmsdf_edge_array* edges, const cmsdf_raster_params* params, uint32_t* pixels) {
    if (!params) {
        return 0;
    }
    cmsdf_rec rec = params->rec;
    if (!edges || !pixels) {
        return rec.width * rec.height * sizeof(uint32_t);
    }
    double max_dist = sqrt(rec.width * rec.width + rec.height * rec.height);
    for (size_t y = 0; y < rec.height; y++) {
        for (size_t x = 0; x < rec.width; x++) {
            edge_point_stats min_blue = {.dist = max_dist};
            edge_point_stats min_green = {.dist = max_dist};
            edge_point_stats min_red = {.dist = max_dist};
            vec2 origin = (vec2){.x = x + 0.5, .y = y + 0.5};
            for (size_t i = 0; i < edges->len; i++) {
                cmsdf_edge* current = edges->data + i;
                edge_point_stats dist = cmsdf_edge_dist_ortho(current, origin);
                if ((current->color & 255) != 0 && edge_point_stats_eq(dist, edge_point_stats_min(dist, min_blue))) {
                    min_blue = dist;
                }
                if ((current->color & 255 << 8) != 0 && edge_point_stats_eq(dist, edge_point_stats_min(dist, min_green))) {
                    min_green = dist;
                }
                if ((current->color & 255 << 16) != 0 && edge_point_stats_eq(dist, edge_point_stats_min(dist, min_red))) {
                    min_red = dist;
                }
            }
            double val_blue = clamp(min_blue.dist / max_dist * RASTER_COLOR_SCALE, 0, 1);
            double val_green = clamp(min_green.dist / max_dist * RASTER_COLOR_SCALE, 0, 1);
            double val_red = clamp(min_red.dist / max_dist * RASTER_COLOR_SCALE, 0, 1);
            if (min_blue.sign < 0) {
                val_blue *= -1;
            }
            if (min_green.sign < 0) {
                val_green *= -1;
            }
            if (min_red.sign < 0) {
                val_red *= -1;
            }
            uint32_t blue = (uint32_t)round((val_blue + 1) * 127.5);
            uint32_t green = (uint32_t)round((val_green + 1) * 127.5);
            uint32_t red = (uint32_t)round((val_red + 1) * 127.5);
            pixels[params->offset + x + y * params->stride] = blue | green << 8 | red << 16;
        }
    }
    return rec.width * rec.height * sizeof(uint32_t);
}

static uint32_t get_channel(uint32_t val, uint32_t shift) {
    return (val >> shift) & 0xff;
}

// based on sorting network
// should compile branchless with cmovs on
// gcc and clang with -O2
static uint8_t median3u(uint8_t a, uint8_t b, uint8_t c) {
    // clang-format off
    if (a > b) { uint8_t t = a; a = b; b = t; }
    if (b > c) { uint8_t t = b; b = c; c = t; }
    if (a > b) { uint8_t t = a; a = b; b = t; }
    // clang-format on
    return b;
}

// https://github.com/Chlumsky/msdfgen/issues/74#issuecomment-479573572
// a pair of pixels may cause defects if
// - 2 components share the same sign
// - the remaining component has an opposing sign
// - the opposing sign occurs in a different channel
// - the median sign flips (so the pixel encode an edge)
static bool causes_defect(uint32_t target, uint32_t neighbour) {
    uint32_t mask = 0x00808080;
    if ((target & mask) == 0 || (neighbour & mask) == 0) {  // all channels < 128
        return false;
    }
    if ((target & mask) == mask || (neighbour & mask) == mask) {  // all channels > 127
        return false;
    }
    if ((target & mask) == (neighbour & mask)) {  // signs are equal across channels
        return false;
    }
    uint8_t t_median = median3u(get_channel(target, 0), get_channel(target, 8), get_channel(target, 16));
    uint8_t n_median = median3u(get_channel(neighbour, 0), get_channel(neighbour, 8), get_channel(neighbour, 16));
    return (t_median > 127) == (n_median > 127);
}

// Basic error correction
// Each pixel is compared with it's neighbour above and to the right.
// If the pixel could causes a defect, all channels are set to the median channel.
// As the loop goes from the bottom-left to the top-right fixing a pixel does
// not interfere with further checks.
// Technically the top-right neighbour could also be checked, but from a few tests
// that did more harm than help.
void cmsdf_postprocess(const cmsdf_raster_params* params, uint32_t* pixels) {
    if (!params || !pixels) {
        return;
    }
    for (size_t y = 0; y < params->rec.height - 1; y++) {
        for (size_t x = 0; x < params->rec.width - 1; x++) {
            size_t check = params->offset + x + y * params->stride;
            size_t right = check + 1;
            size_t up = check + params->stride;
            if (causes_defect(pixels[check], pixels[right]) || causes_defect(pixels[check], pixels[up])) {
                uint8_t median = median3u(get_channel(pixels[check], 0), get_channel(pixels[check], 8), get_channel(pixels[check], 16));
                pixels[check] = median | median << 8 | median << 16;
            }
        }
    }
}

size_t cmsdf_draw_edges(const cmsdf_edge_array* edges, const cmsdf_raster_params* params, uint32_t* pixels) {
    if (!params) {
        return 0;
    }
    cmsdf_rec rec = params->rec;
    if (!pixels || !edges) {
        return rec.width * rec.width * sizeof(uint32_t);
    }
    for (size_t y = 0; y < rec.height; y++) {
        for (size_t x = 0; x < rec.width; x++) {
            cmsdf_edge* closet_edge;
            double min_dist = INFINITY;
            vec2 origin = (vec2){.x = x + 0.5, .y = y + 0.5};
            for (size_t i = 0; i < edges->len; i++) {
                cmsdf_edge* current = edges->data + i;
                edge_point_stats dist = cmsdf_edge_dist_ortho(current, origin);
                if (dist.dist < min_dist) {
                    min_dist = dist.dist;
                    closet_edge = current;
                }
            }
            size_t idx = params->offset + x + y * params->stride;
            if (min_dist > 1.5) {
                pixels[idx] = 0;
            } else {
                pixels[idx] = closet_edge->color;
            }
        }
    }
    return rec.width * rec.height * sizeof(uint32_t);
}

static uint32_t sample_bilinear(vec2 at, uint32_t* image, size_t width, size_t height, uint32_t shift) {
    double frac_x, int_x;
    frac_x = modf(at.x, &int_x);
    double frac_y, int_y;
    frac_y = modf(at.y, &int_y);
    uint32_t x = (uint32_t)int_x;
    uint32_t y = (uint32_t)int_y;

    frac_x = frac_x >= 0 ? frac_x : 1 - frac_x;
    frac_y = frac_y >= 0 ? frac_y : 1 - frac_y;
    uint32_t high_x = at.x < 0 ? int_x : int_x + 1;
    uint32_t high_y = at.y < 0 ? int_y : int_y + 1;
    high_x = high_x >= width ? width - 1 : high_x;
    high_y = high_y >= height ? height - 1 : high_y;

    double part00 = get_channel(image[x + y * width], shift) * (1 - frac_x) * (1 - frac_y);
    double part01 = get_channel(image[x + high_y * width], shift) * (1 - frac_x) * frac_y;
    double part10 = get_channel(image[high_x + y * width], shift) * frac_x * (1 - frac_y);
    double part11 = get_channel(image[high_x + high_y * width], shift) * frac_x * frac_y;
    return part00 + part01 + part10 + part11;
}

static double median3d(double a, double b, double c) {
    // clang-format off
    if (a > b) { uint8_t t = a; a = b; b = t; }
    if (b > c) { uint8_t t = b; b = c; c = t; }
    if (a > b) { uint8_t t = a; a = b; b = t; }
    // clang-format on
    return b;
}

size_t cmsdf_render(const cmsdf_render_params* params, uint32_t* pixels) {
    if (!params) {
        return 0;
    }
    if (!pixels) {
        return params->render_width * params->render_height * sizeof(uint32_t);
    }
    double x_mult = (double)params->msdf_width / params->render_width;
    double y_mult = (double)params->msdf_height / params->render_height;
    for (size_t y = 0; y < params->render_height; y++) {
        for (size_t x = 0; x < params->render_width; x++) {
            vec2 offset = {.x = x * x_mult - 0.5, .y = y * y_mult - 0.5};
            double blue = sample_bilinear(offset, params->msdf, params->msdf_width, params->msdf_height, 0);
            double green = sample_bilinear(offset, params->msdf, params->msdf_width, params->msdf_height, 8);
            double red = sample_bilinear(offset, params->msdf, params->msdf_width, params->msdf_height, 16);
            double median = median3d(blue, green, red);
            if (params->anti_aliasing) {
                double dist = median - 127.5;
                double color = (dist + 1.0) * 127.5;
                pixels[x + y * params->render_width] = clamp(color, 0.0, 255.0);
            } else {
                pixels[x + y * params->render_width] = median > 127.5 ? 255 : 0;
            }
        }
    }
    return params->render_width * params->render_height * sizeof(uint32_t);
}

static void tile_size(size_t count, size_t* tile_width, size_t* tile_height) {
    size_t min_dim = (size_t)ceil(sqrt(count));
    *tile_width = min_dim;
    *tile_height = min_dim;
    if (min_dim * min_dim - count >= min_dim) {
        (*tile_height)--;
    }
    return;
}

int cmsdf_gen_atlas(const cmsdf_gen_atlas_params* params, cmsdf_gen_atlas_result* result) {
    if (!params || !result) {
        return CMSDF_ERR_FACE_MISSING_GLYPH;
    }
    int err = 0;
    size_t tile_width, tile_height;
    tile_size(params->chars_len, &tile_width, &tile_height);
    cmsdf_raster_params raster_params = {
        .rec = (cmsdf_rec){.width = params->dim.width, .height = params->dim.height},
        .offset = 0,
        .stride = tile_width * params->dim.width,
    };
    cmsdf_edge_array arr;
    err = cmsdf_edge_array_new(&arr);
    if (err) {
        return err;
    }
    result->pixels = calloc(tile_width * tile_height * cmsdf_raster_edges(NULL, &raster_params, NULL), 1);
    if (!result->pixels) {
        return CMSDF_ERR_OOM;
    }
    for (size_t i = 0; i < params->chars_len; i++) {
        if (params->flags & CMSDF_GEN_ATLAS_VERBOSE) {
            printf("codepoint: %x\n", params->chars[i]);
        }
        cmsdf_decompose_params decompose_params = {
            .face = params->face,
            .character = params->chars[i],
            .pixel_height = params->dim.height,
            .pixel_width = params->dim.width,
            .arr = &arr,
        };
        cmsdf_decompose_result decompose_result;
        int err = cmsdf_decompose(&decompose_params, &decompose_result);
        if (err) {
            printf("failed to decompose freetype: %d\n", err);
            goto cleanup;
        }
        if (params->flags & CMSDF_GEN_ATLAS_VERBOSE) {
            printf("width: %zu, height: %zu\n", decompose_result.rec.width, decompose_result.rec.height);
            printf("contours: %zu, edge count: %zu\n", decompose_result.contour_idx.len, decompose_result.edges.len);
            cmsdf_edge_array_print(&decompose_result.edges);
        }
        size_t x = i % tile_width;
        size_t y = i / tile_width;
        raster_params.offset = x * decompose_result.rec.width + y * decompose_result.rec.height * tile_width * decompose_result.rec.width;
        if (params->flags & CMSDF_GEN_ATLAS_EDGES) {
            cmsdf_draw_edges(&decompose_result.edges, &raster_params, result->pixels);
        } else {
            cmsdf_raster_edges(&decompose_result.edges, &raster_params, result->pixels);
            cmsdf_postprocess(&raster_params, result->pixels);
        }
        arr.len = 0;
    }
    result->dim.width = tile_width * params->dim.width;
    result->dim.height = tile_height * params->dim.height;
    result->len = result->dim.width * result->dim.height;
cleanup:
    cmsdf_edge_array_free(&arr);
    if (err) {
        free(result->pixels);
    }
    return err;
}
