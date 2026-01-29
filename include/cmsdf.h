// SPDX-License-Identifier: BSD-3-Clause

#ifndef CMSDF_H
#define CMSDF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "freetype/freetype.h"

#define CMSDF_ERR_FACE_MISSING_GLYPH -1337
#define CMSDF_ERR_FACE_NO_OUTLINE -1338
#define CMSDF_ERR_OOM -1339

typedef struct cmsdf_edge cmsdf_edge;

typedef struct {
    cmsdf_edge* data;
    size_t len;
    size_t cap;
} cmsdf_edge_array;

void cmsdf_edge_array_print(cmsdf_edge_array arr);

typedef struct {
    FT_Face face;
    FT_ULong character;
    FT_UInt pixel_width;
    FT_UInt pixel_height;
} cmsdf_decompose_params;

#define CMSDF_CONTOUR_INDEX_CAP 15

typedef struct {
    size_t offsets[CMSDF_CONTOUR_INDEX_CAP];
    size_t len;
} cmsdf_contour_index;

typedef struct {
    size_t width;
    size_t height;
} cmsdf_rec;

typedef struct {
    cmsdf_edge_array edges;
    cmsdf_rec rec;
    cmsdf_contour_index contour_idx;
} cmsdf_decompose_result;

int cmsdf_decompose(cmsdf_decompose_params params, cmsdf_decompose_result* result);
size_t cmsdf_raster_edges(cmsdf_edge_array edges, cmsdf_rec rec, uint32_t* pixels);
void cmsdf_postprocess(uint32_t* pixels, size_t width, size_t height);
size_t cmsdf_draw_edges(cmsdf_edge_array edges, cmsdf_rec rec, uint32_t* pixels);

typedef struct {
    uint32_t* msdf;
    size_t msdf_width;
    size_t msdf_height;
    size_t render_width;
    size_t render_height;
    bool anti_aliasing;
} cmsdf_render_params;

size_t cmsdf_render(cmsdf_render_params params, uint32_t* pixels);

#endif
