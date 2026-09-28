/*
 * Copyright 2026 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_XFB_H
#define PANVK_XFB_H

#include <stdint.h>

/* Transform feedback on JM (Bifrost) is emulated: for each draw, the vertex
 * shader XFB variant runs as a compute job with one invocation per captured
 * vertex (primitive decomposed to a list), and stores the captured outputs
 * with global stores. Shared between the driver, the XFB shader variant and
 * the libpan helpers.
 */

#define PANVK_XFB_MAX_BUFFERS 4

/* Bytes written to each bound buffer, relative to the binding offset. Lives
 * in GPU memory for the duration of a Begin/EndTransformFeedback pair, so
 * GPU jobs can advance it and the counter buffers can be loaded/stored. */
struct panvk_xfb_offsets {
   uint32_t bytes[PANVK_XFB_MAX_BUFFERS];
};

/* Per-draw parameters, in GPU memory. */
struct panvk_xfb_params {
   /* Start of the bound range of each buffer (buffer address + offset). */
   uint64_t buffer_addr[PANVK_XFB_MAX_BUFFERS];

   /* struct panvk_xfb_offsets */
   uint64_t offsets;

   /* Index buffer, 0 for a non-indexed draw. */
   uint64_t index_buffer;

   /* With primitive restart: the indices of the captured vertices, one per
    * vertex of each output primitive, restarts resolved (written by
    * panlib_jm_xfb_patch). 0 without primitive restart. */
   uint64_t vertex_list;

   /* Size of the bound range of each buffer. */
   uint32_t buffer_size[PANVK_XFB_MAX_BUFFERS];

   /* Stride of each buffer in bytes, 0 if the shader does not write it. */
   uint32_t stride[PANVK_XFB_MAX_BUFFERS];

   /* Vertex (non-indexed) or index (indexed) count, per instance. */
   uint32_t vertex_count;
   uint32_t instance_count;

   /* firstVertex (non-indexed) or firstIndex (indexed). */
   uint32_t first;

   /* vertexOffset of an indexed draw. */
   int32_t vertex_offset;

   /* Index size in bytes, 0 for a non-indexed draw. */
   uint32_t index_size;
   uint32_t index_buffer_size_el;
   uint32_t vertex_list_size_el;

   /* enum mesa_prim of the draw, and the output primitives. */
   uint32_t prim;
   uint32_t verts_per_prim;
   uint32_t prims_per_instance;

   /* Vertex ID offset of the draw's vertex job (DRAW::offset_start, and the
    * raw_vertex_offset sysval): the XFB job uses the same attribute
    * descriptors, so it passes vertex IDs relative to it. */
   int32_t raw_offset;

   /* gl_BaseVertex and gl_BaseInstance (firstInstance). The XFB variant uses
    * these instead of the sysvals, which are only patched in the draw's own
    * push uniforms for indirect draws. */
   int32_t base_vertex;
   uint32_t base_instance;

   uint32_t pad;
};

#endif
