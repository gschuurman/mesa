/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#include "util/os_time.h"

#include "libpan_dgc.h"
#include "nir_builder.h"

#include "vk_log.h"
#include "vk_meta.h"
#include "vk_pipeline.h"

#include "genxml/gen_macros.h"

#include "panvk_buffer.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_meta.h"
#include "panvk_cmd_precomp.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_macros.h"
#include "panvk_query_pool.h"

#include "libpan.h"

static nir_def *
panvk_nir_query_report_dev_addr(nir_builder *b, nir_def *pool_addr,
                                nir_def *query_stride, nir_def *query)
{
   return nir_iadd(b, pool_addr, nir_umul_2x32_64(b, query, query_stride));
}

static nir_def *
panvk_nir_available_dev_addr(nir_builder *b, nir_def *available_addr,
                             nir_def *query)
{
   nir_def *offset = nir_imul_imm(b, query, sizeof(uint32_t));
   return nir_iadd(b, available_addr, nir_u2u64(b, offset));
}

static void
panvk_emit_write_job(struct panvk_cmd_buffer *cmd, struct panvk_batch *batch,
                     enum mali_write_value_type type, uint64_t addr,
                     uint64_t value)
{
   struct pan_ptr job =
      pan_pool_alloc_desc(&cmd->desc_pool.base, WRITE_VALUE_JOB);

   pan_section_pack(job.cpu, WRITE_VALUE_JOB, PAYLOAD, payload) {
      payload.type = type;
      payload.address = addr;
      payload.immediate_value = value;
   };

   pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_WRITE_VALUE, true, false, 0, 0,
                  &job, false);
   util_dynarray_append(&batch->jobs, job.cpu);
}

static struct panvk_batch *
open_batch(struct panvk_cmd_buffer *cmd, bool *had_batch)
{
   bool res = cmd->cur_batch != NULL;

   if (!res)
      panvk_per_arch(cmd_open_batch)(cmd);

   *had_batch = res;

   return cmd->cur_batch;
}

static void
close_batch(struct panvk_cmd_buffer *cmd, bool had_batch)
{
   if (!had_batch)
      panvk_per_arch(cmd_close_batch)(cmd);
}

static void
panvk_emit_clear_queries(struct panvk_cmd_buffer *cmd,
                         struct panvk_query_pool *pool, bool availaible,
                         uint32_t first_query, uint32_t query_count)
{
   const struct panlib_clear_query_result_args push = {
      .pool_addr = panvk_priv_mem_dev_addr(pool->mem),
      .available_addr = panvk_priv_mem_dev_addr(pool->available_mem),
      .query_stride = pool->query_stride,
      .first_query = first_query,
      .query_count = query_count,
      .report_count = pool->reports_per_query,
      .availaible_value = availaible,
   };

   bool had_batch;
   open_batch(cmd, &had_batch);
   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmd);
   panlib_clear_query_result_struct(&precomp_ctx, panlib_1d(query_count),
                                    PANLIB_BARRIER_NONE, push);
   close_batch(cmd, had_batch);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdResetQueryPool)(VkCommandBuffer commandBuffer,
                                  VkQueryPool queryPool, uint32_t firstQuery,
                                  uint32_t queryCount)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(panvk_query_pool, pool, queryPool);

   if (queryCount == 0)
      return;

   panvk_emit_clear_queries(cmd, pool, false, firstQuery, queryCount);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdWriteTimestamp2)(VkCommandBuffer commandBuffer,
                                   VkPipelineStageFlags2 stage,
                                   VkQueryPool queryPool, uint32_t query)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(panvk_query_pool, pool, queryPool);

   /* JM has no finer-grained stage tracking: close the current batch so the
    * timestamp is taken after all previous work, whatever the stage, and
    * write it at the start of a new batch. Like CmdEndQuery, keep that batch
    * open if we were recording into one (e.g. inside a render pass).
    */
   bool end_sync = cmd->cur_batch != NULL;
   if (end_sync)
      panvk_per_arch(cmd_close_batch)(cmd);

   bool had_batch;
   struct panvk_batch *batch = open_batch(cmd, &had_batch);
   had_batch |= end_sync;

   /* With multiview, the timestamp uses one query per view. */
   const uint32_t n_views =
      MAX2(1, util_bitcount(cmd->state.gfx.render.view_mask));

   batch->needs_cycle_count = true;
   for (uint32_t q = query; q < query + n_views; q++) {
      panvk_emit_write_job(cmd, batch, MALI_WRITE_VALUE_TYPE_SYSTEM_TIMESTAMP,
                           panvk_query_report_dev_addr(pool, q), 0);
      panvk_emit_write_job(cmd, batch, MALI_WRITE_VALUE_TYPE_IMMEDIATE_32,
                           panvk_query_available_dev_addr(pool, q), 1);
   }

   close_batch(cmd, had_batch);
}

/* Adds the primitives of one draw (draw_cmd: a Vk[Indexed]DrawIndirectCommand
 * in GPU memory, or 0 for none) plus cpu_count to the active primitives
 * generated query, with a compute job. A non-zero index_buffer means primitive
 * restart is enabled, so the indices are scanned. Must be called with a batch
 * open.
 */
void
panvk_per_arch(cmd_update_prims_generated_query)(
   struct panvk_cmd_buffer *cmd, uint64_t draw_cmd, uint64_t index_buffer,
   uint32_t index_buffer_size_el, uint32_t index_size, unsigned prim,
   uint32_t cpu_count)
{
   struct panvk_prims_generated_query_state *state =
      &cmd->state.gfx.prims_generated_query;

   assert(state->ptr && cmd->cur_batch);

   const struct panlib_jm_update_prims_generated_query_args args = {
      .prims_generated = state->ptr,
      .cmd = draw_cmd,
      .index_buffer = index_buffer,
      .index_buffer_size_el = index_buffer_size_el,
      .index_bytes = index_size,
      .prim = prim,
      .view_count = MAX2(1, util_bitcount(cmd->state.gfx.render.view_mask)),
      .cpu_count = cpu_count,
   };

   /* The barrier orders the job after the report reset done when the query
    * began, and after any earlier job writing the draw parameters. */
   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmd);
   panlib_jm_update_prims_generated_query_struct(
      &precomp_ctx, panlib_1d(1), PANLIB_BARRIER_JM_BARRIER, args);

   state->gpu_count = true;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBeginQueryIndexedEXT)(VkCommandBuffer commandBuffer,
                                        VkQueryPool queryPool, uint32_t query,
                                        VkQueryControlFlags flags,
                                        uint32_t index)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(panvk_query_pool, pool, queryPool);

   /* TODO: transform feedback */
   assert(index == 0);

   bool had_batch;
   struct panvk_batch *batch = open_batch(cmd, &had_batch);
   uint64_t report_addr = panvk_query_report_dev_addr(pool, query);

   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_OCCLUSION: {
      cmd->state.gfx.occlusion_query.ptr = report_addr;
      cmd->state.gfx.occlusion_query.mode = flags & VK_QUERY_CONTROL_PRECISE_BIT
                                               ? MALI_OCCLUSION_MODE_COUNTER
                                               : MALI_OCCLUSION_MODE_PREDICATE;
      gfx_state_set_dirty(cmd, OQ);

      /* From the Vulkan spec:
       *
       *   "When an occlusion query begins, the count of passing samples
       *    always starts at zero."
       *
       * With multiview, the query uses one query per view: all views count
       * into the first one, the others stay zero.
       */
      const uint32_t n_views =
         MAX2(1, util_bitcount(cmd->state.gfx.render.view_mask));
      for (uint32_t q = query; q < query + n_views; q++) {
         uint64_t addr = panvk_query_report_dev_addr(pool, q);

         for (unsigned i = 0; i < pool->reports_per_query; i++) {
            panvk_emit_write_job(
               cmd, batch, MALI_WRITE_VALUE_TYPE_IMMEDIATE_64,
               addr + i * sizeof(struct panvk_query_report), 0);
         }
      }
      break;
   }
   case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT: {
      struct panvk_prims_generated_query_state *state =
         &cmd->state.gfx.prims_generated_query;

      state->ptr = report_addr;
      state->cpu_count = 0;
      state->gpu_count = false;

      /* From the Vulkan spec:
       *
       *   "When a primitives generated query begins, the count of primitives
       *    generated starts from zero."
       *
       * With multiview, all views count into the first query, the others
       * stay zero.
       */
      const uint32_t n_views =
         MAX2(1, util_bitcount(cmd->state.gfx.render.view_mask));
      for (uint32_t q = query; q < query + n_views; q++) {
         panvk_emit_write_job(cmd, batch, MALI_WRITE_VALUE_TYPE_IMMEDIATE_64,
                              panvk_query_report_dev_addr(pool, q), 0);
      }
      break;
   }
   default:
      UNREACHABLE("Unsupported query type");
   }

   close_batch(cmd, had_batch);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdEndQueryIndexedEXT)(VkCommandBuffer commandBuffer,
                                      VkQueryPool queryPool, uint32_t query,
                                      uint32_t index)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(panvk_query_pool, pool, queryPool);

   /* TODO: transform feedback */
   assert(index == 0);

   bool end_sync = cmd->cur_batch != NULL;

   /* Close to ensure we are sync and flush caches */
   if (end_sync)
      panvk_per_arch(cmd_close_batch)(cmd);

   bool had_batch;
   struct panvk_batch *batch = open_batch(cmd, &had_batch);
   had_batch |= end_sync;

   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_OCCLUSION: {
      cmd->state.gfx.occlusion_query.ptr = 0;
      cmd->state.gfx.occlusion_query.mode = MALI_OCCLUSION_MODE_DISABLED;
      gfx_state_set_dirty(cmd, OQ);
      break;
   }
   case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT: {
      struct panvk_prims_generated_query_state *state =
         &cmd->state.gfx.prims_generated_query;

      /* The draws of the query are in previous batches, so they are done.
       * Add the primitives counted on the CPU: a plain write is enough if no
       * GPU job added to the report. */
      if (state->cpu_count) {
         if (state->gpu_count) {
            panvk_per_arch(cmd_update_prims_generated_query)(
               cmd, 0, 0, 0, 0, 0, state->cpu_count);
         } else {
            panvk_emit_write_job(cmd, batch,
                                 MALI_WRITE_VALUE_TYPE_IMMEDIATE_64,
                                 state->ptr, state->cpu_count);
         }

         /* The GPU caches are not coherent with the CPU and may write lines
          * back in any order: flush the report, by ending the batch, before
          * marking the query available. */
         panvk_per_arch(cmd_close_batch)(cmd);
         batch = panvk_per_arch(cmd_open_batch)(cmd);
      }

      state->ptr = 0;
      state->cpu_count = 0;
      state->gpu_count = false;
      break;
   }
   default:
      UNREACHABLE("Unsupported query type");
   }

   /* With multiview, every per-view query becomes available. */
   const uint32_t n_views =
      MAX2(1, util_bitcount(cmd->state.gfx.render.view_mask));
   for (uint32_t q = query; q < query + n_views; q++) {
      panvk_emit_write_job(cmd, batch, MALI_WRITE_VALUE_TYPE_IMMEDIATE_32,
                           panvk_query_available_dev_addr(pool, q), 1);
   }

   close_batch(cmd, had_batch);
}

static void
panvk_meta_copy_query_pool_results(struct panvk_cmd_buffer *cmd,
                                   struct panvk_query_pool *pool,
                                   uint32_t first_query, uint32_t query_count,
                                   uint64_t dst_addr, uint64_t dst_stride,
                                   VkQueryResultFlags flags)
{
   const struct panlib_copy_query_result_args push = {
      .pool_addr = panvk_priv_mem_dev_addr(pool->mem),
      .available_addr = panvk_priv_mem_dev_addr(pool->available_mem),
      .query_stride = pool->query_stride,
      .first_query = first_query,
      .query_count = query_count,
      .dst_addr = dst_addr,
      .dst_stride = dst_stride,
      .query_type = pool->vk.query_type,
      .flags = flags,
      .report_count = pool->reports_per_query,
   };

   bool had_batch;
   open_batch(cmd, &had_batch);
   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmd);
   panlib_copy_query_result_struct(&precomp_ctx, panlib_1d(query_count),
                                   PANLIB_BARRIER_NONE, push);
   close_batch(cmd, had_batch);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdCopyQueryPoolResults)(
   VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery,
   uint32_t queryCount, VkBuffer dstBuffer, VkDeviceSize dstOffset,
   VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(panvk_query_pool, pool, queryPool);
   VK_FROM_HANDLE(panvk_buffer, dst_buffer, dstBuffer);

   /* XXX: Do we really need that barrier when EndQuery already handle it? */
   if ((flags & VK_QUERY_RESULT_WAIT_BIT) && cmd->cur_batch != NULL) {
      close_batch(cmd, true);
   }

   uint64_t dst_addr = panvk_buffer_gpu_ptr(dst_buffer, dstOffset);
   panvk_meta_copy_query_pool_results(cmd, pool, firstQuery, queryCount,
                                      dst_addr, stride, flags);
}
