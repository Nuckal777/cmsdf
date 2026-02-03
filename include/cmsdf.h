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

int cmsdf_edge_array_new(cmsdf_edge_array* arr);
void cmsdf_edge_array_free(cmsdf_edge_array* arr);
void cmsdf_edge_array_print(const cmsdf_edge_array* arr);

typedef struct {
    FT_Face face;
    FT_ULong character;
    FT_UInt pixel_width;
    FT_UInt pixel_height;
    cmsdf_edge_array* arr;
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

int cmsdf_decompose(const cmsdf_decompose_params* params, cmsdf_decompose_result* result);

typedef struct {
    cmsdf_rec rec;
    size_t offset;
    size_t stride;
} cmsdf_raster_params;

size_t cmsdf_raster_edges(const cmsdf_edge_array* edges, const cmsdf_raster_params* params, uint32_t* pixels);

void cmsdf_postprocess(const cmsdf_raster_params* params, uint32_t* pixels);

size_t cmsdf_draw_edges(const cmsdf_edge_array* edges, const cmsdf_raster_params* params, uint32_t* pixels);

typedef struct {
    uint32_t* msdf;
    size_t msdf_width;
    size_t msdf_height;
    size_t render_width;
    size_t render_height;
    bool anti_aliasing;
} cmsdf_render_params;

size_t cmsdf_render(const cmsdf_render_params* params, uint32_t* pixels);

#define CMSDF_GEN_ATLAS_VERBOSE 1
#define CMSDF_GEN_ATLAS_EDGES 2

typedef struct {
    FT_Face face;
    cmsdf_rec dim;
    uint32_t* chars;
    size_t chars_len;
    uint32_t flags;
} cmsdf_gen_atlas_params;

typedef struct {
    uint32_t* pixels;
    size_t len;
    cmsdf_rec dim;
} cmsdf_gen_atlas_result;

int cmsdf_gen_atlas(const cmsdf_gen_atlas_params* params, cmsdf_gen_atlas_result* result);

#endif
