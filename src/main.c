#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freetype/freetype.h"
#include "freetype/ftglyph.h"
#include "freetype/ftimage.h"
#include "freetype/ftoutln.h"
#include "freetype/fttypes.h"

#define ERR_FACE_MISSING_GLYPH -1337
#define ERR_FACE_NO_OUTLINE -1338
#define ERR_OOM -1339
#define ERR_FILE_IO -1340

typedef struct {
    double x;
    double y;
} vec2;

vec2 splat2(double v) {
    return (vec2){.x = v, .y = v};
}

vec2 vec2_add(vec2 a, vec2 b) {
    return (vec2){.x = a.x + b.x, .y = a.y + b.y};
}

vec2 vec2_sub(vec2 a, vec2 b) {
    return (vec2){.x = a.x - b.x, .y = a.y - b.y};
}

vec2 vec2_mult(vec2 a, vec2 b) {
    return (vec2){.x = a.x * b.x, .y = a.y * b.y};
}

double vec2_dot(vec2 a, vec2 b) {
    return a.x * b.x + a.y * b.y;
}

double vec2_cross(vec2 a, vec2 b) {
    return a.x * b.y - a.y * b.x;
}

double vec2_dist(vec2 a, vec2 b) {
    vec2 diff = vec2_sub(a, b);
    return sqrt(diff.x * diff.x + diff.y * diff.y);
}

#define EDGE_TY_LINE 1
#define EDGE_TY_CONIC 2
#define EDGE_TY_CUBIC 3

#define EDGE_COLOR_CYAN 1
#define EDGE_COLOR_YELLOW 2
#define EDGE_COLOR_MAGENTA 3

typedef struct {
    vec2 start;
    vec2 end;
    vec2 control1;
    vec2 control2;
    int color;
} edge;

int edge_type(edge* e) {
    if (!isnan(e->control2.x)) {
        return EDGE_TY_CUBIC;
    }
    if (!isnan(e->control1.x)) {
        return EDGE_TY_CONIC;
    }
    return EDGE_TY_LINE;
}

edge make_line(vec2 start, vec2 end) {
    return (edge){
        .start = start,
        .end = end,
        .control1 = (vec2){.x = NAN, .y = NAN},
        .control2 = (vec2){.x = NAN, .y = NAN},
        .color = 0,
    };
}

edge make_conic(vec2 start, vec2 end, vec2 control1) {
    return (edge){
        .start = start,
        .end = end,
        .control1 = control1,
        .control2 = (vec2){.x = NAN, .y = NAN},
        .color = 0,
    };
}

edge make_cubic(vec2 start, vec2 end, vec2 control1, vec2 control2) {
    return (edge){
        .start = start,
        .end = end,
        .control1 = control1,
        .control2 = control2,
        .color = 0,
    };
}

vec2 edge_at(edge* e, double t) {
    int ty = edge_type(e);
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

vec2 edge_dir(edge* e, double t) {
    int ty = edge_type(e);
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

int approx_eq_abs(double a, double b, double tolerance) {
    assert(tolerance >= 0);
    if (a == b) {
        return 1;
    }
    if (isnan(a) || isnan(b)) {
        return 0;
    }
    return fabs(a - b) <= tolerance;
}

size_t solveCubic(double coeff[4], double* out) {
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

double min_dist_sq(edge* e, double t, vec2 point) {
    vec2 on_curve = edge_at(e, t);
    vec2 tangent = edge_dir(e, t);
    vec2 to_point = vec2_sub(on_curve, point);
    return 2 * vec2_dot(to_point, tangent);
}

double edge_arg_min_dist(edge* e, vec2 point) {
    int ty = edge_type(e);
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
                vec2 on_curve = edge_at(e, t);
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
                double d = vec2_dist(edge_at(e, x1), point);
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

double edge_dist(edge* e, vec2 point) {
    double arg_min = edge_arg_min_dist(e, point);
    vec2 at = edge_at(e, arg_min);
    return vec2_dist(at, point);
}

size_t edge_intersect_ray(edge* e, vec2 origin, vec2 dir, double* out) {
    int ty = edge_type(e);
    switch (ty) {
        case EDGE_TY_LINE: {
            vec2 edge_dir = vec2_sub(e->end, e->start);
            double cross_dir = vec2_cross(edge_dir, dir);
            if (cross_dir == 0.0) {
                return 0;
            }
            vec2 shifted = vec2_sub(e->start, origin);
            double t = vec2_cross(dir, shifted) / cross_dir;
            double s = vec2_cross(edge_dir, shifted) / cross_dir;
            if (t < 0 || t > 1 || s < 0) {
                return 0;
            }
            out[0] = t;
            return 1;
        }
        case EDGE_TY_CONIC: {  // doesn't support vertical rays
            assert(dir.x != 0);
            vec2 a = vec2_sub(vec2_add(e->start, e->end), vec2_mult(e->control1, splat2(2)));
            vec2 b = vec2_mult(vec2_sub(e->control1, e->start), splat2(2));
            vec2 c = vec2_sub(e->start, origin);
            double slope = dir.y / dir.x;

            double denominator = a.y - slope * a.x;
            double p = (b.y - slope * b.x) / denominator;
            double q = (c.y - slope * c.x) / denominator;
            double ts[2] = {-p / 2 + sqrt((p * p / 4) - q), -p / 2 - sqrt((p * p / 4) - q)};
            size_t count = 0;
            for (size_t i = 0; i < 2; i++) {
                if (ts[i] < 0 || ts[i] > 1) {
                    continue;
                }
                double s = a.x * ts[i] * ts[i] + b.x * ts[i] + c.x;
                if (s >= 0) {
                    out[count] = ts[i];
                    count++;
                }
            }
            if (count == 2 && out[0] == out[1]) {
                return 1;
            }
            return count;
        }
        case EDGE_TY_CUBIC: {
            vec2 a = vec2_add(vec2_sub(vec2_mult(splat2(3), e->control1), e->start), vec2_sub(e->end, vec2_mult(splat2(3), e->control2)));
            vec2 b = vec2_add(vec2_sub(vec2_mult(splat2(3), e->start), vec2_mult(splat2(6), e->control1)), vec2_mult(splat2(3), e->control2));
            vec2 c = vec2_add(vec2_mult(splat2(-3), e->start), vec2_mult(splat2(3), e->control1));
            vec2 d = e->start;

            double coeff[4] = {vec2_cross(a, dir), vec2_cross(b, dir), vec2_cross(c, dir), vec2_cross(d, dir) - vec2_cross(origin, dir)};
            double roots[3];
            size_t root_count = solveCubic(coeff, roots);
            size_t count = 0;
            for (size_t i = 0; i < root_count; i++) {
                if (roots[i] < 0 || roots[i] > 1) {
                    continue;
                }
                vec2 at = edge_at(e, roots[i]);
                double dot = vec2_dot(vec2_sub(at, origin), dir);
                double s = dot / (dir.x * dir.x + dir.y * dir.y);
                if (s > 0) {
                    out[count] = roots[i];
                    count++;
                }
            }
            return count;
        }
    }
    assert(0);
    return 0;
}

double edge_sgn_dist(edge* e, vec2 origin) {
    double arg_min = edge_arg_min_dist(e, origin);
    vec2 at = edge_at(e, arg_min);
    vec2 dir = edge_dir(e, arg_min);
    double dist = vec2_dist(at, origin);
    double cross = vec2_cross(dir, vec2_sub(at, origin));
    if (cross > 0) {
        return dist;
    }
    if (cross < 0) {
        return -dist;
    }
    // fully parallel to an edge
    // assuming a line min_arg between 0 and 1
    // means origin is on the line => inside
    // else outside
    if (arg_min >= 0 && arg_min <= 1) {
        return dist;
    }
    return -dist;
}

typedef struct {
    edge* data;
    size_t len;
    size_t cap;
} edge_array;

#define EDGE_ARRAY_DEFAULT_CAP 16

int edge_array_new(edge_array* arr) {
    arr->data = malloc(sizeof(edge) * EDGE_ARRAY_DEFAULT_CAP);
    if (!arr->data) {
        return ERR_OOM;
    }
    arr->cap = EDGE_ARRAY_DEFAULT_CAP;
    arr->len = 0;
    return 0;
}

int edge_array_push(edge_array* arr, edge e) {
    if (arr->len == arr->cap - 1) {
        size_t new_cap = arr->cap * 3 / 2;
        edge* tmp = realloc(arr->data, sizeof(edge) * new_cap);
        if (!tmp) {
            free(arr->data);
            return ERR_OOM;
        }
        arr->data = tmp;
        arr->cap = new_cap;
    }
    arr->data[arr->len] = e;
    arr->len++;
    return 0;
}

int edge_array_inside(edge_array edges, vec2 origin) {
    size_t acc = 0;
    double intersections[3];
    for (size_t i = 0; i < edges.len; i++) {
        acc += edge_intersect_ray(edges.data + i, origin, (vec2){.x = 1, .y = 0}, intersections);
    }
    return acc % 2 == 0;
}

#define CONTOUR_INDEX_CAP 15

typedef struct {
    size_t offsets[CONTOUR_INDEX_CAP];
    size_t len;
} contour_index;

typedef struct {
    vec2 start;
    edge_array edges;
    contour_index contour_idx;
} decompose_ctx;

vec2 from_ft_vec(FT_Vector v) {
    return (vec2){.x = v.x / 64.0, .y = v.y / 64.0};
}

int decompose_move_to(const FT_Vector* to, void* user) {
    decompose_ctx* ctx = (decompose_ctx*)user;
    ctx->start = from_ft_vec(*to);
    if (ctx->contour_idx.len == CONTOUR_INDEX_CAP - 1) {
        return ERR_OOM;
    }
    ctx->contour_idx.offsets[ctx->contour_idx.len] = ctx->edges.len;
    ctx->contour_idx.len++;
    return 0;
}

int decompose_line_to(const FT_Vector* to, void* user) {
    decompose_ctx* ctx = (decompose_ctx*)user;
    int err = edge_array_push(&ctx->edges, make_line(ctx->start, from_ft_vec(*to)));
    ctx->start = from_ft_vec(*to);
    return err;
}

int decompose_conic_to(const FT_Vector* control, const FT_Vector* to, void* user) {
    decompose_ctx* ctx = (decompose_ctx*)user;
    int err = edge_array_push(&ctx->edges, make_conic(ctx->start, from_ft_vec(*to), from_ft_vec(*control)));
    ctx->start = from_ft_vec(*to);
    return err;
}

int decompose_cubic_to(const FT_Vector* control1, const FT_Vector* control2, const FT_Vector* to, void* user) {
    decompose_ctx* ctx = (decompose_ctx*)user;
    int err = edge_array_push(&ctx->edges, make_cubic(ctx->start, from_ft_vec(*to), from_ft_vec(*control1), from_ft_vec(*control2)));
    ctx->start = from_ft_vec(*to);
    return err;
}

typedef struct {
    edge_array edges;
    contour_index contour_idx;
} decompose_result;

typedef struct {
    ssize_t width;
    ssize_t height;
    ssize_t offX;
    ssize_t offY;
} raster_rec;

int decompose(FT_Library ft_library, const char* fontpath, decompose_result* result, raster_rec* rec) {
    FT_Face face;
    FT_Error err = FT_New_Face(ft_library, fontpath, 0, &face);
    if (err) {
        return err;
    }
    err = FT_Set_Pixel_Sizes(face, 64, 0);
    if (err) {
        goto cleanup;
    }
    FT_UInt idx = FT_Get_Char_Index(face, '&');
    if (!idx) {
        err = ERR_FACE_MISSING_GLYPH;
        goto cleanup;
    }
    err = FT_Load_Glyph(face, idx, 0);
    if (err) {
        goto cleanup;
    }
    FT_Glyph_Metrics* metrics = &face->glyph->metrics;
    if (face->glyph->format != FT_GLYPH_FORMAT_OUTLINE) {
        err = ERR_FACE_NO_OUTLINE;
        goto cleanup;
    }
    FT_Outline* outline = &face->glyph->outline;
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
    err = edge_array_new(&ctx.edges);
    if (err) {
        goto cleanup;
    }
    err = FT_Outline_Decompose(outline, &funcs, &ctx);
    if (err) {
        goto cleanup_edges;
    }
    result->edges = ctx.edges;
    result->contour_idx = ctx.contour_idx;

    FT_Glyph glyph;
    err = FT_Get_Glyph(face->glyph, &glyph);
    if (err) {
        goto cleanup_edges;
    }
    FT_BBox cbox;
    FT_Glyph_Get_CBox(glyph, FT_GLYPH_BBOX_PIXELS, &cbox);
    FT_Done_Glyph(glyph);
    *rec = (raster_rec){
        .width = (cbox.xMax - cbox.xMin),
        .height = (cbox.yMax - cbox.yMin),
        .offX = cbox.xMin,
        .offY = cbox.yMin,
    };
cleanup_edges:
    if (err) {
        free(ctx.edges.data);
    }
cleanup:
    FT_Done_Face(face);
    return err;
}

int raster_edges(edge_array edges, raster_rec rec, uint32_t** out, size_t* out_size) {
    *out_size = rec.width * rec.height * sizeof(uint32_t);
    uint32_t* pixels = malloc(*out_size);
    if (!pixels) {
        return ERR_OOM;
    }
    double max_dist = sqrt(rec.width * rec.width + rec.height * rec.height);
    for (ssize_t y = 0; y < rec.height; y++) {
        for (ssize_t x = 0; x < rec.width; x++) {
            double min_dist = max_dist;
            // at y + 0.5 it is quite likely to pass through a vertex, which confuses the ray
            // casting algorithm, so a small offset is added here, which should make a collision
            // sufficiently unlikely.
            vec2 origin = (vec2){.x = (x + rec.offX) + 0.5, .y = (y + rec.offY) + 0.5 + 1e-9};
            for (size_t i = 0; i < edges.len; i++) {
                double dist = edge_dist(edges.data + i, origin);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            int inside = edge_array_inside(edges, origin);
            double val = min_dist / max_dist * 3;
            if (!inside) {
                val *= -1;
            }
            uint32_t color = (uint32_t)((val + 1) * 127.5);
            size_t idx = x + y * rec.width;
            pixels[idx] = color | color << 8 | color << 16;
        }
    }
    *out = pixels;
    return 0;
}

#define BMP_HEADER_SIZE 54

size_t bmp_file_header(uint8_t* data, uint32_t size, uint32_t pixel_offset) {
    data[0] = 0x42;
    data[1] = 0x4D;
    memcpy(data + 2, &size, 4);
    memset(data + 6, 0, 4);
    memcpy(data + 10, &pixel_offset, 4);
    return 14;
}

size_t bmp_bitmap_info_header(uint8_t* data, int32_t width, int32_t height) {
    data[0] = 40;
    memset(data + 1, 0, 3);
    memcpy(data + 4, &width, 4);
    memcpy(data + 8, &height, 4);
    data[12] = 1;
    data[13] = 0;
    data[14] = 32;
    data[15] = 0;
    memset(data + 16, 0, 24);
    return 40;
}

int write_file(const char* filename, void* buf, size_t buf_size) {
    int err = 0;
    FILE* f = fopen(filename, "wb");
    if (!f) {
        errno = 0;
        return ERR_FILE_IO;
    }
    unsigned long written = fwrite(buf, sizeof(uint8_t), buf_size, f);
    if (written != buf_size) {
        err = ERR_FILE_IO;
    }
cleanup:
    fclose(f);
    return err;
}

int main() {
    FT_Library ft_library;
    FT_Error err = FT_Init_FreeType(&ft_library);
    if (err) {
        printf("failed to init freetype: %d\n", err);
        return EXIT_FAILURE;
    }
    decompose_result result;
    raster_rec rec;
    err = decompose(ft_library, "/usr/share/fonts/adobe-source-code-pro/SourceCodePro-Regular.otf", &result, &rec);
    if (err) {
        printf("failed to decompose freetype: %d\n", err);
        goto cleanup;
    }
    printf("width: %zu, height: %zu\n", rec.width, rec.height);
    printf("contours: %zu, edge count: %zu\n", result.contour_idx.len, result.edges.len);
    size_t raster_size;
    uint32_t* raster;
    err = raster_edges(result.edges, rec, &raster, &raster_size);
    if (err) {
        printf("failed to raster edges: %d\n", err);
        goto cleanup_raster;
    }
    uint8_t* buf = malloc(raster_size + BMP_HEADER_SIZE);
    size_t file_header_size = bmp_file_header(buf, raster_size + BMP_HEADER_SIZE, BMP_HEADER_SIZE);
    size_t bitmap_info_size = bmp_bitmap_info_header(buf + file_header_size, rec.width, rec.height);
    memcpy(buf + BMP_HEADER_SIZE, raster, raster_size);
    err = write_file("out.bmp", buf, raster_size + BMP_HEADER_SIZE);
    if (err) {
        printf("failed to write file: %d\n", err);
    }
    free(buf);
cleanup_raster:
    free(raster);
cleanup_edges:
    free(result.edges.data);
cleanup:
    if (FT_Done_FreeType(ft_library)) {
        printf("failed to deinit freetype: %d\n", err);
    }
    return err ? EXIT_FAILURE : EXIT_SUCCESS;
}
