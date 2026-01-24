#ifndef CMSDF_H
#define CMSDF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "freetype/freetype.h"

#define ERR_FACE_MISSING_GLYPH -1337
#define ERR_FACE_NO_OUTLINE -1338
#define ERR_OOM -1339
#define ERR_FILE_IO -1340
#define ERR_INVALID_ARGS -1341
#define ERR_INVALID_BMP -1342
#define ERR_INVALID_UTF8 -1343

typedef struct edge edge;

typedef struct {
    edge* data;
    size_t len;
    size_t cap;
} edge_array;

void edge_array_print(edge_array arr);

typedef struct {
    const char* fontpath;
    FT_ULong character;
    FT_UInt pixel_width;
    FT_UInt pixel_height;
} decompose_params;

#define CONTOUR_INDEX_CAP 15

typedef struct {
    size_t offsets[CONTOUR_INDEX_CAP];
    size_t len;
} contour_index;

typedef struct {
    edge_array edges;
    contour_index contour_idx;
} decompose_result;

typedef struct {
    size_t width;
    size_t height;
} raster_rec;

int decompose(FT_Face ft_face, decompose_params params, decompose_result* result, raster_rec* rec);
size_t raster_edges(edge_array edges, raster_rec rec, uint32_t* pixels);
void postprocess(uint32_t* pixels, size_t width, size_t height);
size_t draw_edges(edge_array edges, raster_rec rec, uint32_t* pixels);

typedef struct {
    uint32_t* msdf;
    size_t msdf_width;
    size_t msdf_height;
    size_t render_width;
    size_t render_height;
    bool anti_aliasing;
} render_params;

size_t render(render_params params, uint32_t* pixels);

#endif
