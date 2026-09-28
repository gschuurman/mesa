/*
 * Copyright 2025 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "compiler/libcl/libcl.h"
#include "compiler/libcl/libcl_vk.h"
#include "compiler/shader_enums.h"
#include "genxml/gen_macros.h"
#include "lib/pan_encoder.h"
#include "poly/cl/restart.h"
#include "draw_helper.h"
#include "vulkan/panvk_xfb.h"

#if PAN_ARCH >= 10

/* Subgroup size is always 16 on v10+ */
#define SUBGROUP_SIZE 16

KERNEL(SUBGROUP_SIZE * SUBGROUP_SIZE)
panlib_update_prims_generated_query_restart(
   global atomic_uint *prims_generated, uint64_t index_buffer,
   uint32_t index_buffer_size_el, uint32_t cmd_stride,
   constant VkDrawIndexedIndirectCommand *cmd, uint32_t view_count,
   uint32_t compact_prim__11, uint32_t index_bytes_log2__3)
{
   assert(get_sub_group_size() == SUBGROUP_SIZE);

   enum mesa_prim prim = poly_uncompact_prim(compact_prim__11);
   uint32_t index_bytes = 1 << index_bytes_log2__3;
   uint32_t restart_index = (1 << (index_bytes * 8)) - 1;

   uint32_t tid = cl_local_id.x;
   uint32_t draw_id = cl_group_id.x;

   cmd = (constant VkDrawIndexedIndirectCommand *)
      ((uintptr_t) cmd + cmd_stride * draw_id);

   POLY_DECL_UNROLL_RESTART_SCRATCH(scratch, SUBGROUP_SIZE * SUBGROUP_SIZE);
   uint32_t prims_per_instance = poly_count_restart_prims(
      (constant uint32_t *) cmd, index_buffer, index_buffer_size_el,
      index_bytes, restart_index, prim, scratch);

   if (tid == 0)
      atomic_fetch_add(prims_generated,
                       prims_per_instance * cmd->instanceCount * view_count);
}

KERNEL(1)
panlib_update_prims_generated_query_indirect(
   global uint32_t *prims_generated, global uint32_t *draw_count_buffer,
   uint32_t max_draw_count, uint32_t cmd_stride, constant uint32_t *cmd,
   uint32_t view_count, uint32_t compact_prim__11)
{
   enum mesa_prim prim = poly_uncompact_prim(compact_prim__11);
   uint32_t draw_count = draw_count_buffer ?
      min(*draw_count_buffer, max_draw_count) : max_draw_count;

   for (uint32_t draw_id = 0; draw_id < draw_count; draw_id++) {
      /* cmd may be either VkDrawnIndirectCommand or
       * VkDrawIndexedIndirectCommand. In both cases the vertex/index count is
       * the first field, and the instance count is the second */
      uint32_t vertex_count = cmd[0];
      uint32_t instance_count = cmd[1];

      uint32_t prims_per_instance =
         u_decomposed_prims_for_vertices(prim, vertex_count);
      *prims_generated += prims_per_instance * instance_count;

      cmd = (constant uint32_t *) ((uintptr_t) cmd + cmd_stride);
   }
}
#endif

#if (PAN_ARCH == 6 || PAN_ARCH == 7)
struct panlib_draw_info {
   struct {
      uint32_t size;
      uint32_t offset;
   } index;

   struct {
      int32_t raw_offset;
      int32_t base;
      uint32_t count;
   } vertex;

   struct {
      int32_t base;
      uint32_t count;
   } instance;

   struct {
      global struct mali_attribute_buffer_packed *descs;
      global struct libpan_draw_helper_varying_buf_info *info;
   } varying_bufs;

   struct {
      global struct mali_attribute_buffer_packed *descs;
      global struct libpan_draw_helper_attrib_buf_info *infos;
      uint32_t valid;
   } attrib_bufs;

   struct {
      global struct mali_attribute_packed *descs;
      global struct libpan_draw_helper_attrib_info *infos;
      uint32_t valid;
   } attribs;

   global uint8_t *idvs_job;
   global uint8_t *vertex_job;
   global uint8_t *tiler_job;

   uint32_t vertex_range;
   uint32_t padded_vertex_count;
   uint8_t primitive_vertex_count;
   global uint8_t *indices;

   uint64_t position;
   uint64_t psiz;
};

static void
panlib_patch_draw_vertex_dcd(struct panlib_draw_info *draw)
{
   global struct mali_invocation_packed *vertex_invocation;
   global struct mali_draw_packed *vertex_dcd;

   if (draw->idvs_job != NULL) {
      vertex_invocation =
         (global struct mali_invocation_packed *)(draw->idvs_job +
                                                  pan_section_offset(
                                                     INDEXED_VERTEX_JOB,
                                                     INVOCATION));
      vertex_dcd = (global struct mali_draw_packed *)(draw->idvs_job +
                                                      pan_section_offset(
                                                         INDEXED_VERTEX_JOB,
                                                         VERTEX_DRAW));
   } else {
      vertex_invocation =
         (global struct mali_invocation_packed *)(draw->vertex_job +
                                                  pan_section_offset(
                                                     COMPUTE_JOB, INVOCATION));
      vertex_dcd = (global struct mali_draw_packed *)(draw->vertex_job +
                                                      pan_section_offset(
                                                         COMPUTE_JOB, DRAW));
   }

   /* Patch the number of invocations on the vertex job */
   pan_pack_work_groups_compute(vertex_invocation, 1, draw->vertex_range,
                                draw->instance.count, 1, 1, 1, true, false);

   /* Patch the draw descriptor */
   pan_unpack(vertex_dcd, DRAW, unpacked_vertex_dcd)
      ;
   pan_pack(vertex_dcd, DRAW, cfg) {
      memcpy(&cfg, &unpacked_vertex_dcd, sizeof(cfg));
      cfg.offset_start = draw->vertex.raw_offset;
      cfg.instance_size =
         draw->instance.count > 1 ? draw->padded_vertex_count : 1;
   }
}

static void
panlib_patch_draw_tiler_dcd(struct panlib_draw_info *draw)
{
   /* No tiler job, nothing to do here */
   if (draw->idvs_job == NULL && draw->tiler_job == NULL)
      return;

   global struct mali_draw_packed *tiler_dcd;
   global struct mali_primitive_packed *tiler_primitive;
   global struct mali_primitive_size_packed *tiler_primsz;

   if (draw->idvs_job != NULL) {
      tiler_dcd = (global struct mali_draw_packed *)(draw->idvs_job +
                                                     pan_section_offset(
                                                        INDEXED_VERTEX_JOB,
                                                        FRAGMENT_DRAW));
      tiler_primitive =
         (global struct mali_primitive_packed *)(draw->idvs_job +
                                                 pan_section_offset(
                                                    INDEXED_VERTEX_JOB,
                                                    PRIMITIVE));
      tiler_primsz =
         (global struct mali_primitive_size_packed *)(draw->idvs_job +
                                                      pan_section_offset(
                                                         INDEXED_VERTEX_JOB,
                                                         PRIMITIVE_SIZE));
   } else {
      global struct mali_invocation_packed *vertex_invocation =
         (global struct mali_invocation_packed *)(draw->vertex_job +
                                                  pan_section_offset(
                                                     COMPUTE_JOB, INVOCATION));
      global struct mali_invocation_packed *tiler_invocation =
         (global struct mali_invocation_packed *)(draw->tiler_job +
                                                  pan_section_offset(
                                                     TILER_JOB, INVOCATION));
      /* Patch the tiler job invocation if present (identical to the vertex
       * invocations) */
      memcpy(tiler_invocation, vertex_invocation, pan_size(INVOCATION));

      tiler_dcd = (global struct mali_draw_packed *)(draw->tiler_job +
                                                     pan_section_offset(
                                                        TILER_JOB, DRAW));
      tiler_primitive =
         (global struct mali_primitive_packed *)(draw->tiler_job +
                                                 pan_section_offset(TILER_JOB,
                                                                    PRIMITIVE));
      tiler_primsz =
         (global struct mali_primitive_size_packed *)(draw->tiler_job +
                                                      pan_section_offset(
                                                         TILER_JOB,
                                                         PRIMITIVE_SIZE));
   }

   /* Patch the attribute start offset */
   pan_unpack(tiler_dcd, DRAW, unpacked_tiler_dcd)
      ;
   pan_pack(tiler_dcd, DRAW, cfg) {
      memcpy(&cfg, &unpacked_tiler_dcd, sizeof(cfg));
      cfg.position = draw->position;
      cfg.offset_start = draw->vertex.raw_offset;
      cfg.instance_size =
         draw->instance.count > 1 ? draw->padded_vertex_count : 1;
      uint32_t primitives_per_instance =
         DIV_ROUND_UP(draw->padded_vertex_count, draw->primitive_vertex_count);
      cfg.instance_primitive_size =
         pan_padded_vertex_count(primitives_per_instance);
   }

   /* Patch the primitive vertex offset to take offset_start into account in
    * case of indexed draw */
   pan_unpack(tiler_primitive, PRIMITIVE, unpacked_tiler_primitive)
      ;

   pan_pack(tiler_primitive, PRIMITIVE, cfg) {
      memcpy(&cfg, &unpacked_tiler_primitive, sizeof(cfg));
      cfg.index_count = draw->vertex.count;

      if (draw->index.size) {
         cfg.indices =
            (uint64_t)draw->indices + draw->index.offset * draw->index.size;
         cfg.base_vertex_offset =
            (int64_t)draw->vertex.base - draw->vertex.raw_offset;
      }
   }

   /* In case we have a point size, we need to patch the descriptor there */
   if (unpacked_tiler_primitive.point_size_array_format !=
       MALI_POINT_SIZE_ARRAY_FORMAT_NONE) {
      pan_pack(tiler_primsz, PRIMITIVE_SIZE, cfg) {
         cfg.size_array = draw->psiz;
      }
   }
}

static void
panlib_patch_varying_bufs(struct panlib_draw_info *draw)
{
   uint32_t vertex_count = draw->padded_vertex_count * draw->instance.count;

   for (uint32_t i = 0; i < PANLIB_VARY_BUF_MAX; i++) {
      global struct libpan_draw_helper_varying_buf_info *info =
         &draw->varying_bufs.info[0];

      global struct mali_attribute_buffer_packed *desc =
         &draw->varying_bufs.descs[i];
      pan_unpack(desc, ATTRIBUTE_BUFFER, unpacked_desc)
         ;

      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         memcpy(&cfg, &unpacked_desc, sizeof(cfg));
         cfg.size = vertex_count * cfg.stride;

         /* Ensure we are aligned properly */
         uint32_t offset_add = ALIGN_POT(cfg.size, 64);

         /* Allocate some space for ourself */
         uint32_t current_offset = atomic_fetch_add_explicit(
            &info->offset, offset_add, memory_order_relaxed);

         /* Sanity check that we don't overrun the buffer */
         assert(current_offset + cfg.size <= info->size);

         cfg.pointer = info->address + current_offset;

         if (i == PANLIB_VARY_BUF_POSITION)
            draw->position = cfg.pointer;
         else if (i == PANLIB_VARY_BUF_PSIZ)
            draw->psiz = cfg.pointer;
      }
   }
}

static void
panlib_patch_attrib_buf(struct panlib_draw_info *draw,
                        global struct mali_attribute_buffer_packed *desc,
                        struct libpan_draw_helper_attrib_buf_info info)
{
   uint32_t divisor = draw->padded_vertex_count * info.divisor;
   global struct mali_attribute_buffer_continuation_npot_packed *desc_ext =
      (global struct mali_attribute_buffer_continuation_npot_packed *)&desc[1];

   /* We are effectively recreating the attribute buffer descriptors. The only
    * trustworthy infos in the descriptor are the pointer and size */
   pan_unpack(desc, ATTRIBUTE_BUFFER, unpacked_desc)
      ;

   if (draw->instance.count <= 1) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.stride = info.per_instance ? 0 : info.stride;
         cfg.pointer = unpacked_desc.pointer;
         cfg.size = unpacked_desc.size;
      }
   } else if (!info.per_instance) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_MODULUS;
         cfg.divisor = draw->padded_vertex_count;
         cfg.stride = info.stride;
         cfg.pointer = unpacked_desc.pointer;
         cfg.size = unpacked_desc.size;
      }
   } else if (!divisor) {
      /* instance_divisor == 0 means all instances share the same value.
       * Make it a 1D array with a zero stride.
       */
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.stride = 0;
         cfg.pointer = unpacked_desc.pointer;
         cfg.size = unpacked_desc.size;
      }
   } else if (util_is_power_of_two_or_zero(divisor)) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_POT_DIVISOR;
         cfg.stride = info.stride;
         cfg.pointer = unpacked_desc.pointer;
         cfg.size = unpacked_desc.size;
         cfg.divisor_r = __builtin_ctz(divisor);
      }
   } else {
      uint32_t divisor_r = 0, divisor_e = 0;
      uint32_t divisor_d =
         pan_compute_npot_divisor(divisor, &divisor_r, &divisor_e);
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_NPOT_DIVISOR;
         cfg.stride = info.stride;
         cfg.pointer = unpacked_desc.pointer;
         cfg.size = unpacked_desc.size;
         cfg.divisor_r = divisor_r;
         cfg.divisor_e = divisor_e;
      }

      pan_pack(desc_ext, ATTRIBUTE_BUFFER_CONTINUATION_NPOT, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_CONTINUATION;
         cfg.divisor_numerator = divisor_d;
         cfg.divisor = info.divisor;
      }

      return;
   }

   /* If the buffer extension wasn't used, memset(0) */
   for (uint32_t i = 0; i < pan_size(ATTRIBUTE_BUFFER) / 4; i++)
      desc_ext->opaque[i] = 0;
}

static void
panlib_patch_attrib(struct panlib_draw_info *draw,
                    global struct mali_attribute_packed *desc,
                    struct libpan_draw_helper_attrib_info info)
{
   /* We are effectively recreating the attribute descriptors. The only
    * trustworthy infos in the descriptor are the buffer index and format */
   pan_unpack(desc, ATTRIBUTE, unpacked_desc)
      ;

   /* info.stride is only set for per-instance attributes. For instanced
    * draws, the hardware also offsets per-instance data by offset_start (the
    * raw vertex offset), compensate. */
   int32_t start_offset =
      draw->instance.count > 1 ? draw->vertex.raw_offset * info.stride : 0;

   pan_pack(desc, ATTRIBUTE, cfg) {
      cfg.buffer_index = unpacked_desc.buffer_index;
      cfg.offset =
         info.base_offset + draw->instance.base * info.stride - start_offset;
      cfg.offset_enable = true;
      cfg.format = unpacked_desc.format;
   }
}

static void
panlib_patch_job_type_header(global struct mali_job_header_packed *desc,
                             enum mali_job_type job_type)
{
   if (desc == NULL)
      return;

   pan_unpack(desc, JOB_HEADER, unpacked_desc)
      ;

   pan_pack(desc, JOB_HEADER, cfg) {
      memcpy(&cfg, &unpacked_desc, sizeof(cfg));
      cfg.type = job_type;
   }
}

static void
panlib_patch_draw(struct panlib_draw_info *draw)
{
   /* First of all, we need to ensure each jobs have the proper header */
   bool is_null_job = draw->vertex_range == 0 || draw->instance.count == 0;

   if (draw->idvs_job != NULL) {
      panlib_patch_job_type_header(
         (global struct mali_job_header_packed *)draw->idvs_job,
         is_null_job ? MALI_JOB_TYPE_NULL : MALI_JOB_TYPE_INDEXED_VERTEX);
   } else {
      panlib_patch_job_type_header(
         (global struct mali_job_header_packed *)draw->vertex_job,
         is_null_job ? MALI_JOB_TYPE_NULL : MALI_JOB_TYPE_VERTEX);
      panlib_patch_job_type_header(
         (global struct mali_job_header_packed *)draw->tiler_job,
         is_null_job ? MALI_JOB_TYPE_NULL : MALI_JOB_TYPE_TILER);
   }

   /* In case there is nothing to draw, let's just bail out. */
   if (is_null_job)
      return;

   /* Next we patch varying buffer descriptors and collect position and point
    * size varying addresses */
   panlib_patch_varying_bufs(draw);

   /* Patch the vertex draw and invocation descriptors */
   panlib_patch_draw_vertex_dcd(draw);

   /* Patch the tiler descriptors if present */
   panlib_patch_draw_tiler_dcd(draw);

   /* Finally, patch attribute descriptors */
   uint32_t num_vbs = util_last_bit(draw->attrib_bufs.valid);
   for (uint32_t i = 0; i < num_vbs; i++) {
      if (draw->attrib_bufs.valid & BITFIELD_BIT(i))
         panlib_patch_attrib_buf(draw, &draw->attrib_bufs.descs[i * 2],
                                 draw->attrib_bufs.infos[i]);
   }

   uint32_t num_vs_attribs = util_last_bit(draw->attribs.valid);
   for (uint32_t i = 0; i < num_vs_attribs; i++) {
      if (draw->attribs.valid & BITFIELD_BIT(i))
         panlib_patch_attrib(draw, &draw->attribs.descs[i],
                             draw->attribs.infos[i]);
   }
}

static uint32_t
padded_vertex_count(uint32_t vertex_count, uint32_t instance_count, bool idvs)
{
   if (instance_count == 1)
      return vertex_count;

   /* Index-Driven Vertex Shading requires different instances to
    * have different cache lines for position results. Each vertex
    * position is 16 bytes and the Mali cache line is 64 bytes, so
    * the instance count must be aligned to 4 vertices.
    */
   if (idvs)
      vertex_count = ALIGN_POT(vertex_count, 4);

   return pan_padded_vertex_count(vertex_count);
}

/* Argument ordering choosen to avoid any padding */

KERNEL(1)
panlib_draw_indirect_helper(
   global VkDrawIndirectCommand *cmd,
   global struct mali_attribute_buffer_packed *varying_bufs_descs,
   global struct libpan_draw_helper_varying_buf_info *varying_bufs_info,
   global struct mali_attribute_buffer_packed *attrib_bufs_descs,
   global struct libpan_draw_helper_attrib_buf_info *attrib_bufs_infos,
   uint32_t attrib_bufs_valid, uint32_t attribs_valid,
   global struct mali_attribute_packed *attribs_descs,
   global struct libpan_draw_helper_attrib_info *attribs_infos,
   global uint32_t *first_vertex_sysval, global uint32_t *first_instance_sysval,
   global uint32_t *raw_vertex_offset_sysval, global uint8_t *idvs_job,
   global uint8_t *vertex_job, global uint8_t *tiler_job,
   uint32_t primitive_vertex_count)
{
   const uint32_t vertex_count = cmd->vertexCount;
   const uint32_t instance_count = cmd->instanceCount;
   const uint32_t first_vertex = cmd->firstVertex;
   const uint32_t first_instance = cmd->firstInstance;

   struct panlib_draw_info draw = {
      .idvs_job = idvs_job,
      .vertex_job = vertex_job,
      .tiler_job = tiler_job,

      .vertex.base = first_vertex,
      .vertex.raw_offset = first_vertex,
      .vertex.count = vertex_count,
      .instance.base = first_instance,
      .instance.count = instance_count,

      .varying_bufs.descs = varying_bufs_descs,
      .varying_bufs.info = varying_bufs_info,

      .attrib_bufs.descs = attrib_bufs_descs,
      .attrib_bufs.infos = attrib_bufs_infos,
      .attrib_bufs.valid = attrib_bufs_valid,

      .attribs.descs = attribs_descs,
      .attribs.infos = attribs_infos,
      .attribs.valid = attribs_valid,

      .vertex_range = vertex_count,
      .padded_vertex_count =
         padded_vertex_count(vertex_count, instance_count, idvs_job != NULL),
      .primitive_vertex_count = primitive_vertex_count,
   };

   panlib_patch_draw(&draw);

   *first_vertex_sysval = draw.vertex.base;
   *first_instance_sysval = draw.instance.base;
   *raw_vertex_offset_sysval = draw.vertex.raw_offset;
}

KERNEL(1)
panlib_draw_indexed_indirect_helper(
   global VkDrawIndexedIndirectCommand *cmd, global uint8_t *index_buffer_ptr,
   global struct libpan_draw_helper_index_min_max_result *index_min_max_res,
   uint32_t index_size, uint32_t primitive_vertex_count,
   uint32_t attrib_bufs_valid, uint32_t attribs_valid,
   global struct mali_attribute_buffer_packed *varying_bufs_descs,
   global struct libpan_draw_helper_varying_buf_info *varying_bufs_info,
   global struct mali_attribute_buffer_packed *attrib_bufs_descs,
   global struct libpan_draw_helper_attrib_buf_info *attrib_bufs_infos,
   global struct mali_attribute_packed *attribs_descs,
   global struct libpan_draw_helper_attrib_info *attribs_infos,
   global uint32_t *first_vertex_sysval, global uint32_t *first_instance_sysval,
   global uint32_t *raw_vertex_offset_sysval, global uint8_t *idvs_job,
   global uint8_t *vertex_job, global uint8_t *tiler_job)
{
   const uint32_t index_count = cmd->indexCount;
   const uint32_t first_index = cmd->firstIndex;
   const uint32_t first_instance = cmd->firstInstance;
   const uint32_t instance_count = cmd->instanceCount;
   const int32_t vertex_offset = cmd->vertexOffset;
   const uint32_t min_vertex = index_min_max_res->min;
   const uint32_t max_vertex = index_min_max_res->max;
   const uint32_t vertex_range = max_vertex - min_vertex + 1;

   struct panlib_draw_info draw = {
      .idvs_job = idvs_job,
      .vertex_job = vertex_job,
      .tiler_job = tiler_job,

      .index.size = index_size,
      .index.offset = first_index,
      .vertex.base = vertex_offset,
      .vertex.raw_offset = min_vertex + vertex_offset,
      .vertex.count = index_count,
      .instance.base = first_instance,
      .instance.count = instance_count,

      .varying_bufs.descs = varying_bufs_descs,
      .varying_bufs.info = varying_bufs_info,

      .attrib_bufs.descs = attrib_bufs_descs,
      .attrib_bufs.infos = attrib_bufs_infos,
      .attrib_bufs.valid = attrib_bufs_valid,

      .attribs.descs = attribs_descs,
      .attribs.infos = attribs_infos,
      .attribs.valid = attribs_valid,

      .vertex_range = vertex_range,
      .padded_vertex_count =
         padded_vertex_count(vertex_range, instance_count, idvs_job != NULL),
      .primitive_vertex_count = primitive_vertex_count,
      .indices = index_buffer_ptr,
   };

   panlib_patch_draw(&draw);

   *first_vertex_sysval = draw.vertex.base;
   *first_instance_sysval = draw.instance.base;
   *raw_vertex_offset_sysval = draw.vertex.raw_offset;
}

KERNEL(64)
panlib_draw_index_minmax_search_helper(global uint8_t *index_buffer_ptr,
                                       global VkDrawIndexedIndirectCommand *cmd,
                                       global atomic_uint *min_ptr,
                                       global atomic_uint *max_ptr,
                                       uint32_t index_bytes_log2__3,
                                       uint8_t primitive_restart__2)
{
   /* Max count of values to process per thread */
   const uint32_t max_count_per_thread = 1024;

   const uint32_t index_bit_size = (1 << index_bytes_log2__3) * 8;
   const uint32_t start = cmd->firstIndex;
   const uint32_t index_count = cmd->indexCount;

   uint32_t base_idx = cl_global_id.x * max_count_per_thread;

   /* If the thread is out of range, bail out */
   if (base_idx >= index_count)
      return;

   /* Compute expected max iteration to do in this thread */
   uint32_t count = MIN2(max_count_per_thread, index_count - base_idx);

   /* Sanity check so nothing weird will happen */
   assert(base_idx + count <= index_count);

   uint32_t local_min = ((uint64_t)1 << index_bit_size) - 1;
   uint32_t local_max = 0;

   switch (index_bit_size) {
#define MINMAX_SEARCH_CASE(sz)                                                 \
   case sz: {                                                                  \
      global uint##sz##_t *indices = (global uint##sz##_t *)index_buffer_ptr;  \
      for (uint32_t i = 0; i < count; i++) {                                   \
         uint32_t val = (uint32_t)indices[start + base_idx + i];               \
         if (primitive_restart__2 && val == UINT##sz##_MAX)                    \
            continue;                                                          \
         local_min = min(local_min, val);                                      \
         local_max = max(local_max, val);                                      \
      }                                                                        \
      break;                                                                   \
   }
      MINMAX_SEARCH_CASE(32)
      MINMAX_SEARCH_CASE(16)
      MINMAX_SEARCH_CASE(8)
#undef MINMAX_SEARCH_CASE
   default:
      assert(0 && "Invalid index size");
   }

   atomic_fetch_min(min_ptr, local_min);
   atomic_fetch_max(max_ptr, local_max);
}

static uint32_t
count_restart_prims_serial(global uint8_t *index_buffer,
                           uint32_t index_buffer_size_el, uint32_t index_bytes,
                           uint32_t first, uint32_t count, enum mesa_prim prim)
{
   uint32_t restart_index = ((uint64_t)1 << (index_bytes * 8)) - 1;
   uint32_t prims = 0, run = 0;

   for (uint32_t i = 0; i < count; i++) {
      uint32_t el = first + i;
      /* Out-of-bounds indices read as zero (robustness) */
      uint32_t index = 0;

      if (el < index_buffer_size_el) {
         if (index_bytes == 1)
            index = index_buffer[el];
         else if (index_bytes == 2)
            index = ((global uint16_t *)index_buffer)[el];
         else
            index = ((global uint32_t *)index_buffer)[el];
      }

      if (index == restart_index) {
         prims += u_decomposed_prims_for_vertices(prim, run);
         run = 0;
      } else {
         run++;
      }
   }

   return prims + u_decomposed_prims_for_vertices(prim, run);
}

/* Adds the primitives generated by one draw to a primitives generated query.
 * JM has no multi-draw indirect, so there is at most one draw. cmd is a
 * Vk[Indexed]DrawIndirectCommand (the vertex/index count and the instance
 * count are the first two fields), or 0 to only add cpu_count. With a non-zero
 * index_buffer, the indices are scanned for primitive restarts.
 *
 * cpu_count holds the primitives of direct draws counted on the CPU.
 */
KERNEL(1)
panlib_jm_update_prims_generated_query(global atomic_uint *prims_generated,
                                       constant uint32_t *cmd,
                                       global uint8_t *index_buffer,
                                       uint32_t index_buffer_size_el,
                                       uint32_t index_bytes, uint32_t prim,
                                       uint32_t view_count, uint32_t cpu_count)
{
   uint32_t count = cpu_count;

   if (cmd) {
      uint32_t vertex_count = cmd[0];
      uint32_t instance_count = cmd[1];
      uint32_t prims_per_instance =
         index_buffer
            ? count_restart_prims_serial(index_buffer, index_buffer_size_el,
                                         index_bytes, cmd[2], vertex_count,
                                         (enum mesa_prim)prim)
            : u_decomposed_prims_for_vertices((enum mesa_prim)prim,
                                              vertex_count);

      count += prims_per_instance * instance_count * view_count;
   }

   atomic_fetch_add(prims_generated, count);
}

/* Transform feedback (see vulkan/panvk_xfb.h). The counter buffers hold the
 * bytes written to each buffer; 0 means no counter buffer. */
KERNEL(1)
panlib_jm_xfb_begin(global struct panvk_xfb_offsets *offsets,
                    uint64_t counter0, uint64_t counter1, uint64_t counter2,
                    uint64_t counter3)
{
   uint64_t counters[PANVK_XFB_MAX_BUFFERS] = {counter0, counter1, counter2,
                                               counter3};

   for (unsigned i = 0; i < PANVK_XFB_MAX_BUFFERS; i++) {
      uint32_t bytes = 0;

      if (counters[i])
         bytes = *(global uint32_t *)counters[i];

      offsets->bytes[i] = bytes;
   }
}

KERNEL(1)
panlib_jm_xfb_end(global struct panvk_xfb_offsets *offsets,
                  uint64_t counter0, uint64_t counter1, uint64_t counter2,
                  uint64_t counter3)
{
   uint64_t counters[PANVK_XFB_MAX_BUFFERS] = {counter0, counter1, counter2,
                                               counter3};

   for (unsigned i = 0; i < PANVK_XFB_MAX_BUFFERS; i++) {
      if (counters[i])
         *(global uint32_t *)counters[i] = offsets->bytes[i];
   }
}

/* Vertex of the draw (0 = first) for vertex k of primitive p. Same as
 * xfb_decompose() in panvk_vX_shader.c. */
static uint32_t
xfb_decompose(enum mesa_prim prim, uint32_t p, uint32_t k)
{
   uint32_t strip_k = (p & 1) && k < 2 ? 1 - k : k;

   switch (prim) {
   case MESA_PRIM_LINES:
      return 2 * p + k;
   case MESA_PRIM_LINE_STRIP:
      return p + k;
   case MESA_PRIM_LINES_ADJACENCY:
      return 4 * p + k + 1;
   case MESA_PRIM_LINE_STRIP_ADJACENCY:
      return p + k + 1;
   case MESA_PRIM_TRIANGLES:
      return 3 * p + k;
   case MESA_PRIM_TRIANGLE_STRIP:
      return p + strip_k;
   case MESA_PRIM_TRIANGLE_FAN:
      return k == 2 ? 0 : p + k + 1;
   case MESA_PRIM_TRIANGLES_ADJACENCY:
      return 6 * p + 2 * k;
   case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY:
      return 2 * p + 2 * strip_k;
   default:
      return p;
   }
}

static uint32_t
xfb_load_index(global struct panvk_xfb_params *params, uint32_t el)
{
   if (el >= params->index_buffer_size_el)
      return 0;

   switch (params->index_size) {
   case 1:
      return ((global uint8_t *)params->index_buffer)[el];
   case 2:
      return ((global uint16_t *)params->index_buffer)[el];
   default:
      return ((global uint32_t *)params->index_buffer)[el];
   }
}

/* Primitive restart: writes the index of each captured vertex to the vertex
 * list, one restart-free segment at a time, and returns the primitive count.
 * Primitives that don't fit in the list are dropped. */
static uint32_t
xfb_expand_restart(global struct panvk_xfb_params *params, uint32_t first,
                   uint32_t count)
{
   global uint32_t *list = (global uint32_t *)params->vertex_list;
   enum mesa_prim prim = (enum mesa_prim)params->prim;
   uint32_t vpp = params->verts_per_prim;
   uint32_t restart_index =
      params->index_size == 4 ? UINT32_MAX
                              : (1u << (params->index_size * 8)) - 1;
   uint32_t prims = 0, out = 0, start = 0;

   for (uint32_t i = 0; i <= count; i++) {
      if (i < count && xfb_load_index(params, first + i) != restart_index)
         continue;

      uint32_t seg_prims = u_decomposed_prims_for_vertices(prim, i - start);
      for (uint32_t p = 0; p < seg_prims; p++) {
         if (out + vpp > params->vertex_list_size_el)
            return prims;

         for (uint32_t k = 0; k < vpp; k++) {
            list[out++] = xfb_load_index(
               params, first + start + xfb_decompose(prim, p, k));
         }
         prims++;
      }

      start = i + 1;
   }

   return prims;
}

/* For an indirect or indexed draw: fills the parameters only known on the
 * GPU, and patches the XFB job like panlib_patch_draw_vertex_dcd() patches
 * the draw's vertex job (the XFB job uses the same attribute descriptors).
 * cmd is a Vk[Indexed]DrawIndirectCommand; index_min_max is only set for
 * indexed draws. Runs after the draw helper.
 *
 * When the draw has no vertex job (the vertex shader only feeds transform
 * feedback), the draw helper did not run, so the attribute descriptors are
 * patched here (attrib_bufs_valid/attribs_valid are 0 otherwise). */
KERNEL(1)
panlib_jm_xfb_patch(
   global struct panvk_xfb_params *params, constant uint32_t *cmd,
   constant struct libpan_draw_helper_index_min_max_result *index_min_max,
   global uint8_t *xfb_job,
   global struct mali_attribute_buffer_packed *attrib_bufs_descs,
   global struct libpan_draw_helper_attrib_buf_info *attrib_bufs_infos,
   global struct mali_attribute_packed *attribs_descs,
   global struct libpan_draw_helper_attrib_info *attribs_infos, uint32_t idvs,
   uint32_t attrib_bufs_valid, uint32_t attribs_valid)
{
   uint32_t count = cmd[0];
   uint32_t instance_count = cmd[1];
   uint32_t first = cmd[2];
   int32_t vertex_offset = 0;
   int32_t raw_offset = first;
   uint32_t vertex_range = count;

   /* VkDrawIndirectCommand::firstInstance, or the indexed one's */
   params->base_instance = cmd[3];
   params->base_vertex = first;

   if (params->index_size) {
      vertex_offset = (int32_t)cmd[3];
      raw_offset = index_min_max->min + vertex_offset;
      vertex_range = index_min_max->max - index_min_max->min + 1;
      params->base_instance = cmd[4];
      params->base_vertex = vertex_offset;
   }

   uint32_t prims =
      params->vertex_list
         ? xfb_expand_restart(params, first, count)
         : u_decomposed_prims_for_vertices((enum mesa_prim)params->prim, count);

   params->vertex_count = count;
   params->instance_count = instance_count;
   params->first = first;
   params->vertex_offset = vertex_offset;
   params->raw_offset = raw_offset;
   params->prims_per_instance = prims;

   bool is_null_job = prims == 0 || instance_count == 0 ||
                      index_min_max && index_min_max->min > index_min_max->max;
   panlib_patch_job_type_header((global struct mali_job_header_packed *)xfb_job,
                                is_null_job ? MALI_JOB_TYPE_NULL
                                            : MALI_JOB_TYPE_VERTEX);
   if (is_null_job) {
      params->prims_per_instance = 0;
      return;
   }

   global struct mali_invocation_packed *invocation =
      (global struct mali_invocation_packed *)(xfb_job +
                                               pan_section_offset(COMPUTE_JOB,
                                                                  INVOCATION));
   global struct mali_draw_packed *dcd =
      (global struct mali_draw_packed *)(xfb_job +
                                         pan_section_offset(COMPUTE_JOB, DRAW));

   uint32_t padded =
      padded_vertex_count(vertex_range, instance_count, idvs);

   pan_pack_work_groups_compute(invocation, 1, prims * params->verts_per_prim,
                                instance_count, 1, 1, 1, true, false);

   pan_unpack(dcd, DRAW, unpacked_dcd)
      ;
   pan_pack(dcd, DRAW, cfg) {
      memcpy(&cfg, &unpacked_dcd, sizeof(cfg));
      cfg.offset_start = raw_offset;
      cfg.instance_size = instance_count > 1 ? padded : 1;
   }

   struct panlib_draw_info draw = {
      .vertex.raw_offset = raw_offset,
      .instance.base = params->base_instance,
      .instance.count = instance_count,
      .padded_vertex_count = padded,
   };

   uint32_t num_vbs = util_last_bit(attrib_bufs_valid);
   for (uint32_t i = 0; i < num_vbs; i++) {
      if (attrib_bufs_valid & BITFIELD_BIT(i))
         panlib_patch_attrib_buf(&draw, &attrib_bufs_descs[i * 2],
                                 attrib_bufs_infos[i]);
   }

   uint32_t num_attribs = util_last_bit(attribs_valid);
   for (uint32_t i = 0; i < num_attribs; i++) {
      if (attribs_valid & BITFIELD_BIT(i))
         panlib_patch_attrib(&draw, &attribs_descs[i], attribs_infos[i]);
   }
}

/* After the XFB job of a draw: advance the offsets by the primitives that
 * were written, i.e. the ones that fit in all the buffers (see the XFB
 * variant in panvk_vX_shader.c). */
KERNEL(1)
panlib_jm_xfb_advance(constant struct panvk_xfb_params *params)
{
   global struct panvk_xfb_offsets *offsets =
      (global struct panvk_xfb_offsets *)params->offsets;
   uint32_t vpp = params->verts_per_prim;
   uint32_t prims = params->prims_per_instance * params->instance_count;

   for (unsigned i = 0; i < PANVK_XFB_MAX_BUFFERS; i++) {
      uint32_t stride = params->stride[i];
      if (!stride)
         continue;

      uint32_t written = offsets->bytes[i];
      uint32_t size = params->buffer_size[i];
      uint32_t room = written < size ? size - written : 0;
      prims = min(prims, room / (stride * vpp));
   }

   for (unsigned i = 0; i < PANVK_XFB_MAX_BUFFERS; i++) {
      if (params->stride[i])
         offsets->bytes[i] += prims * vpp * params->stride[i];
   }
}

#endif
