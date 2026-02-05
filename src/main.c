// SPDX-License-Identifier: BSD-3-Clause

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cmsdf.h"

#define CMSDF_ERR_FILE_IO -1340
#define CMSDF_ERR_INVALID_ARGS -1341
#define CMSDF_ERR_INVALID_BMP -1342
#define CMSDF_ERR_INVALID_UTF8 -1343

#define BMP_HEADER_SIZE 54

typedef struct {
    uint8_t* data;
    int32_t width;
    int32_t height;
} bmp_params;

size_t bmp_write_header(bmp_params params, uint8_t* buf) {
    uint32_t data_size = params.width * params.height * sizeof(uint32_t);
    uint32_t total_size = BMP_HEADER_SIZE + data_size;
    if (buf == NULL) {
        return total_size;
    }
    // file header
    buf[0] = 0x42;
    buf[1] = 0x4D;
    memcpy(buf + 2, &total_size, 4);
    memset(buf + 6, 0, 4);
    uint32_t pixel_offset = BMP_HEADER_SIZE;
    memcpy(buf + 10, &pixel_offset, 4);
    // info header
    uint8_t* info_header = buf + 14;
    info_header[0] = 40;
    memset(info_header + 1, 0, 3);
    memcpy(info_header + 4, &params.width, 4);
    memcpy(info_header + 8, &params.height, 4);
    info_header[12] = 1;
    info_header[13] = 0;
    info_header[14] = 32;
    info_header[15] = 0;
    memset(info_header + 16, 0, 24);
    return BMP_HEADER_SIZE;
}

int read_bmp(uint8_t* buf, size_t buf_size, bmp_params* out) {
    uint32_t pixel_offset;
    memcpy(&pixel_offset, buf + 10, sizeof(pixel_offset));
    if (pixel_offset != BMP_HEADER_SIZE) {
        return CMSDF_ERR_INVALID_BMP;
    }
    uint8_t* info_header = buf + 14;
    uint32_t info_header_size;
    memcpy(&info_header_size, info_header, sizeof(info_header_size));
    if (info_header_size != 40) {
        return CMSDF_ERR_INVALID_BMP;
    }
    memcpy(&out->width, info_header + 4, 4);
    memcpy(&out->height, info_header + 8, 4);
    if (out->width < 0 || out->height < 0) {  // Technically negative values are allowed to flip direction
        return CMSDF_ERR_INVALID_BMP;
    }
    if (buf_size < BMP_HEADER_SIZE + (size_t)out->width * (size_t)out->height) {
        return CMSDF_ERR_INVALID_BMP;
    }
    uint16_t val;
    memcpy(&val, info_header + 12, sizeof(val));
    if (val != 1) {
        return CMSDF_ERR_INVALID_BMP;
    }
    memcpy(&val, info_header + 14, sizeof(val));
    if (val != 32) {
        return CMSDF_ERR_INVALID_BMP;
    }
    out->data = buf + BMP_HEADER_SIZE;
    return 0;
}

int write_file(const char* filename, void* buf, size_t buf_size) {
    int err = 0;
    FILE* f = fopen(filename, "wb");
    if (!f) {
        errno = 0;
        return CMSDF_ERR_FILE_IO;
    }
    unsigned long written = fwrite(buf, sizeof(uint8_t), buf_size, f);
    if (written != buf_size) {
        err = CMSDF_ERR_FILE_IO;
    }
    fclose(f);
    return err;
}

int read_file(const char* filename, void** buf, size_t* buf_size) {
    int err = 0;
    FILE* f = fopen(filename, "rb");
    if (!f) {
        errno = 0;
        return CMSDF_ERR_FILE_IO;
    }
    err = fseek(f, 0, SEEK_END);
    if (err) {
        errno = 0;
        err = CMSDF_ERR_FILE_IO;
        goto cleanup;
    }
    long size = ftell(f);
    if (size < 0) {
        errno = 0;
        err = CMSDF_ERR_FILE_IO;
        goto cleanup;
    }
    rewind(f);
    *buf = malloc(size);
    if (!buf) {
        err = CMSDF_ERR_OOM;
        goto cleanup;
    }
    unsigned long read = fread(*buf, 1, size, f);
    if (read != (unsigned long)size) {
        errno = 0;
        err = CMSDF_ERR_FILE_IO;
        goto cleanup_buf;
    }
    *buf_size = size;
cleanup_buf:
    if (err) {
        free(buf);
    }
cleanup:
    fclose(f);
    return err;
}

typedef struct {
    FT_Face face;
    uint32_t* chars;
    size_t chars_len;
    FT_UInt pixel_width;
    FT_UInt pixel_height;
} generate_params;

static int prog_generate(const cmsdf_gen_atlas_params* params, const char* basename) {
    cmsdf_gen_atlas_result result;
    int err = cmsdf_gen_atlas(params, &result, NULL);
    if (err) {
        return err;
    }
    uint8_t* buf = calloc(BMP_HEADER_SIZE + result.len, sizeof(uint8_t));
    err = cmsdf_gen_atlas(params, &result, buf + BMP_HEADER_SIZE);
    if (err) {
        goto cleanup;
    }
    bmp_params raster_bmp_params = {
        .data = buf,
        .width = result.dim.width,
        .height = result.dim.height,
    };
    if (!buf) {
        err = CMSDF_ERR_OOM;
        goto cleanup;
    }
    bmp_write_header(raster_bmp_params, buf);
    char fname[512];
    snprintf(fname, sizeof(fname), "%s.bmp", basename);
    if (params->flags & CMSDF_GEN_ATLAS_EDGES) {
        err = write_file(fname, buf, BMP_HEADER_SIZE + result.len);
    } else {
        err = write_file(fname, buf, BMP_HEADER_SIZE + result.len);
    }
    if (err) {
        printf("failed to write file: %d\n", err);
    }
cleanup:
    free(buf);
    return err;
}

static int prog_render(size_t render_width, size_t render_height) {
    uint8_t* bmp_data;
    size_t bmp_data_size;
    int err = read_file("out.bmp", (void**)&bmp_data, &bmp_data_size);
    if (err) {
        printf("failed to read file: %d\n", err);
        return err;
    }
    bmp_params raster_params;
    err = read_bmp(bmp_data, bmp_data_size, &raster_params);
    if (err) {
        printf("failed to parse bmp: %d\n", err);
        goto cleanup;
    }
    cmsdf_render_params render_params = {
        .msdf = raster_params.data,
        .msdf_height = raster_params.height,
        .msdf_width = raster_params.width,
        .render_width = render_width,
        .render_height = render_height,
        .anti_aliasing = true,
    };
    size_t rendered_size = BMP_HEADER_SIZE + cmsdf_render(&render_params, NULL);
    uint8_t* rendered = malloc(rendered_size);
    if (!rendered) {
        printf("failed to render: %d\n", err);
        goto cleanup;
    }
    cmsdf_render(&render_params, rendered + BMP_HEADER_SIZE);

    bmp_params render_bmp_params = {.data = rendered, .width = render_width, .height = render_height};
    bmp_write_header(render_bmp_params, (uint8_t*)rendered);
    err = write_file("render.bmp", rendered, rendered_size);
    if (err) {
        printf("failed to write file: %d\n", err);
    }
    free(rendered);
cleanup:
    free(bmp_data);
    return err;
}

typedef struct {
    const char* mode;
    const char* font;
    long width;
    long height;
    const char* character;
    const char* output;
    bool verbose;
} prog_args;

static int parse_args(int argc, char* argv[], prog_args* args) {
    args->mode = NULL;
    args->width = 32;
    args->height = 32;
    args->character = "A";
    args->font = "/usr/share/fonts/liberation/LiberationMono-Regular.ttf";
    args->output = "out";
    args->verbose = false;
    if (argc < 2) {
        return CMSDF_ERR_INVALID_ARGS;
    }
    args->mode = argv[1];
    for (int i = 2; i < argc; i++) {
        const char* current = argv[i];
        if (strcmp(current, "-w") == 0) {
            if (i >= argc) {
                return CMSDF_ERR_INVALID_ARGS;
            }
            args->width = strtol(argv[i + 1], NULL, 10);
            if (errno == ERANGE) {
                errno = 0;
                return CMSDF_ERR_INVALID_ARGS;
            }
            i++;
        } else if (strcmp(current, "-h") == 0) {
            if (i >= argc) {
                return CMSDF_ERR_INVALID_ARGS;
            }
            args->height = strtol(argv[i + 1], NULL, 10);
            if (errno == ERANGE) {
                errno = 0;
                return CMSDF_ERR_INVALID_ARGS;
            }
            i++;
        } else if (strcmp(current, "-c") == 0) {
            if (i >= argc) {
                return CMSDF_ERR_INVALID_ARGS;
            }
            args->character = argv[i + 1];
            i++;
        } else if (strcmp(current, "-f") == 0) {
            if (i >= argc) {
                return CMSDF_ERR_INVALID_ARGS;
            }
            args->font = argv[i + 1];
            i++;
        } else if (strcmp(current, "-o") == 0) {
            if (i >= argc) {
                return CMSDF_ERR_INVALID_ARGS;
            }
            args->output = argv[i + 1];
            i++;
        } else if (strcmp(current, "-v") == 0) {
            args->verbose = true;
        } else {
            return CMSDF_ERR_INVALID_ARGS;
        }
    }
    return 0;
}

typedef struct {
    uint32_t* data;
    size_t len;
    size_t cap;
} u32buf;

#define U32_BUF_EMPTY ((u32buf){.data = NULL, .cap = 0, .len = 0})

int u32buf_append(u32buf* buf, uint32_t v) {
    if (!buf->data) {
        buf->cap = 32;
        buf->data = malloc(buf->cap * sizeof(uint32_t));
        if (!buf->data) {
            return CMSDF_ERR_OOM;
        }
        buf->data[0] = v;
        buf->len = 1;
        return 0;
    }
    if (buf->len >= buf->cap) {
        size_t new_cap = buf->cap * 3 / 2;
        uint32_t* tmp = realloc(buf->data, new_cap * sizeof(uint32_t));
        if (!tmp) {
            return CMSDF_ERR_OOM;
        }
        buf->cap = new_cap;
        buf->data = tmp;
    }
    buf->data[buf->len] = v;
    buf->len++;
    return 0;
}

int codepoints_from_utf8(uint32_t** out, size_t* out_len, const char* in, size_t in_len) {
    int err = 0;
    *out = NULL;
    *out_len = 0;
    u32buf buf = U32_BUF_EMPTY;
    for (size_t i = 0; i < in_len; i++) {
        unsigned char current = in[i];
        if (0xf0 == (0xf8 & current)) {  // 4 byte
            if (i + 3 >= in_len) {
                err = CMSDF_ERR_INVALID_UTF8;
                goto cleanup;
            }
            unsigned char n1 = in[++i];
            unsigned char n2 = in[++i];
            unsigned char n3 = in[++i];
            if (0x80 != (0xc0 & n1) || 0x80 != (0xc0 & n2) || 0x80 != (0xc0 & n3)) {  // not continuation bytes
                err = CMSDF_ERR_INVALID_UTF8;
                goto cleanup;
            }
            uint32_t cp = ((current & 0x07) << 18) | ((n1 & 0x3f) << 12) | ((n2 & 0x3f) << 6) | (n3 & 0x3f);
            if (cp < 0x10000 || cp > 0x10ffff) {  // overlong encoding or too large
                err = CMSDF_ERR_INVALID_UTF8;
                goto cleanup;
            }
            err = u32buf_append(&buf, cp);
            if (err) {
                goto cleanup;
            }
            continue;
        }
        if (0xe0 == (0xf0 & current)) {  // 3 byte
            if (i + 2 >= in_len) {
                err = CMSDF_ERR_INVALID_UTF8;
                goto cleanup;
            }
            unsigned char n1 = in[++i];
            unsigned char n2 = in[++i];
            if (0x80 != (0xc0 & n1) || 0x80 != (0xc0 & n2)) {  // not continuation bytes
                err = CMSDF_ERR_INVALID_UTF8;
                goto cleanup;
            }
            uint32_t cp = ((current & 0x0f) << 12) | ((n1 & 0x3f) << 6) | (n2 & 0x3f);
            if (cp < 0x800) {  // overlong encoding
                err = CMSDF_ERR_INVALID_UTF8;
                goto cleanup;
            }
            if (0xd800 <= cp && cp <= 0xdfff) {  // surrogate pair
                err = CMSDF_ERR_INVALID_UTF8;
                goto cleanup;
            }
            err = u32buf_append(&buf, cp);
            if (err) {
                goto cleanup;
            }
            continue;
        }
        if (0xc0 == (0xe0 & current)) {  // 2 byte
            if (i + 1 >= in_len) {
                err = CMSDF_ERR_INVALID_UTF8;
                goto cleanup;
            }
            unsigned char next = in[++i];
            if (0x80 != (0xc0 & next)) {  // not continuation byte
                err = CMSDF_ERR_INVALID_UTF8;
                goto cleanup;
            }
            uint32_t cp = ((current & 0x1f) << 6) | (next & 0x3f);
            if (cp < 0x80) {  // overlong encoding
                err = CMSDF_ERR_INVALID_UTF8;
                goto cleanup;
            }
            err = u32buf_append(&buf, cp);
            if (err) {
                goto cleanup;
            }
            continue;
        }
        if (0x00 == (0x80 & current)) {  // 1 byte
            err = u32buf_append(&buf, current);
            if (err) {
                goto cleanup;
            }
            continue;
        }
        err = CMSDF_ERR_INVALID_UTF8;
        goto cleanup;
    }
    *out = buf.data;
    *out_len = buf.len;
cleanup:
    if (err) {
        free(buf.data);
    }
    return err;
}

int main_internal(int argc, char* argv[]) {
    prog_args args;
    int err = parse_args(argc, argv, &args);
    if (err) {
        fprintf(stderr, "usage: cmsdf [generate|edges|render]\n");
        return EXIT_FAILURE;
    }
    if (strcmp(args.mode, "render") == 0) {
        err = prog_render(args.width, args.height);
        return err ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    bool is_generate = strcmp(args.mode, "generate") == 0;
    bool is_edges = strcmp(args.mode, "edges") == 0;
    if (!is_generate && !is_edges) {
        fprintf(stderr, "unknown mode\n");
        return EXIT_FAILURE;
    }

    FT_Library ft_library;
    if (FT_Init_FreeType(&ft_library)) {
        printf("failed to init freetype: %d\n", err);
        return EXIT_FAILURE;
    }

    uint32_t* out = NULL;
    size_t out_len;
    err = codepoints_from_utf8(&out, &out_len, args.character, strlen(args.character));
    if (err) {
        fprintf(stderr, "failed to decode utf-8\n");
        goto cleanup;
    }
    if (out_len == 0) {
        fprintf(stderr, "no characters provided\n");
        err = CMSDF_ERR_INVALID_UTF8;
        goto cleanup;
    }
    FT_Face ft_face;
    err = FT_New_Face(ft_library, args.font, 0, &ft_face);
    if (err) {
        fprintf(stderr, "failed to load font face from %s\n", args.font);
        err = CMSDF_ERR_FILE_IO;
        goto cleanup;
    }
    cmsdf_gen_atlas_params params = {
        .face = ft_face,
        .chars = out,
        .chars_len = out_len,
        .dim = (cmsdf_rec){.width = args.width, .height = args.height},
        .flags = args.verbose ? CMSDF_GEN_ATLAS_VERBOSE : 0,
    };
    if (is_edges) {
        params.flags |= CMSDF_GEN_ATLAS_EDGES;
    }
    err = prog_generate(&params, args.output);
    FT_Done_Face(ft_face);
cleanup:
    free(out);
    if (FT_Done_FreeType(ft_library)) {
        printf("failed to deinit freetype: %d\n", err);
    }
    return err ? EXIT_FAILURE : EXIT_SUCCESS;
}

#if defined(_WIN32) || defined(_WIN64)

#include <windows.h>

static char* wide_to_utf8(wchar_t* wstr) {
    if (!wstr) {
        return NULL;
    }
    int out_size = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
    if (out_size <= 0) {
        return NULL;
    }
    char* out = malloc(out_size);
    if (!out) {
        return NULL;
    }
    int written = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, out, out_size, NULL, NULL);
    if (written <= 0) {
        free(out);
        return NULL;
    }
    return out;
}

int wmain(int wargc, wchar_t* wargv[]) {
    char** argv = malloc(wargc * sizeof(char*));
    if (!argv) {
        return EXIT_FAILURE;
    }
    int result = EXIT_FAILURE;
    int argc = 0;
    for (int i = 0; i < wargc; i++) {
        argv[i] = wide_to_utf8(wargv[i]);
        if (!argv[i]) {
            goto cleanup;
        }
        argc++;
    }
    result = main_internal(argc, argv);
cleanup:
    for (int i = 0; i < argc; i++) {
        free(argv[i]);
    }
    free(argv);
    return result;
}

#else

int main(int argc, char* argv[]) {
    return main_internal(argc, argv);
}

#endif
