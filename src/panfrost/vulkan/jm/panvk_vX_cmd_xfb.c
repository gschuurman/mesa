/*
 * Copyright 2026 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

/* VK_EXT_transform_feedback on JM: state and counters. The capture itself is
 * done per draw by the XFB variant of the vertex shader, see panvk_xfb.h and
 * panvk_draw_emit_xfb() in panvk_vX_cmd_draw.c.
 */

#include "genxml/gen_macros.h"

#include "panvk_buffer.h"
#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_precomp.h"
#include "panvk_entrypoints.h"
#include "panvk_macros.h"
#include "panvk_xfb.h"

#include "libpan_dgc.h"
#include "libpan.h"

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBindTransformFeedbackBuffersEXT)(
   VkCommandBuffer commandBuffer, uint32_t firstBinding, uint32_t bindingCount,
   const VkBuffer *pBuffers, const VkDeviceSize *pOffsets,
   const VkDeviceSize *pSizes)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(panvk_buffer, buffer, pBuffers[i]);
      uint32_t idx = firstBinding + i;
      VkDeviceSize size =
         pSizes && pSizes[i] != VK_WHOLE_SIZE ? pSizes[i]
                                              : buffer->vk.size - pOffsets[i];

      assert(idx < PANVK_XFB_MAX_BUFFERS);
      cmdbuf->state.gfx.xfb.buffers[idx].addr =
         panvk_buffer_gpu_ptr(buffer, pOffsets[i]);
      cmdbuf->state.gfx.xfb.buffers[idx].size = MIN2(size, UINT32_MAX);
   }
}

/* Addresses of the counter buffers, 0 for none. */
static void
get_counters(uint32_t firstCounterBuffer, uint32_t counterBufferCount,
             const VkBuffer *pCounterBuffers,
             const VkDeviceSize *pCounterBufferOffsets,
             uint64_t counters[PANVK_XFB_MAX_BUFFERS])
{
   for (uint32_t i = 0; i < PANVK_XFB_MAX_BUFFERS; i++)
      counters[i] = 0;

   for (uint32_t i = 0; i < counterBufferCount; i++) {
      if (!pCounterBuffers || pCounterBuffers[i] == VK_NULL_HANDLE)
         continue;

      VK_FROM_HANDLE(panvk_buffer, buffer, pCounterBuffers[i]);
      uint64_t offset = pCounterBufferOffsets ? pCounterBufferOffsets[i] : 0;
      counters[firstCounterBuffer + i] = panvk_buffer_gpu_ptr(buffer, offset);
   }
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBeginTransformFeedbackEXT)(
   VkCommandBuffer commandBuffer, uint32_t firstCounterBuffer,
   uint32_t counterBufferCount, const VkBuffer *pCounterBuffers,
   const VkDeviceSize *pCounterBufferOffsets)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   struct pan_ptr offsets = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, sizeof(struct panvk_xfb_offsets), 8);
   if (!offsets.gpu)
      return;

   uint64_t counters[PANVK_XFB_MAX_BUFFERS];
   get_counters(firstCounterBuffer, counterBufferCount, pCounterBuffers,
                pCounterBufferOffsets, counters);

   bool had_batch = cmdbuf->cur_batch != NULL;
   if (!had_batch)
      panvk_per_arch(cmd_open_batch)(cmdbuf);

   /* Resume from the counter buffers (written by the GPU), or start at 0. */
   const struct panlib_jm_xfb_begin_args args = {
      .offsets = offsets.gpu,
      .counter0 = counters[0],
      .counter1 = counters[1],
      .counter2 = counters[2],
      .counter3 = counters[3],
   };
   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   panlib_jm_xfb_begin_struct(&precomp_ctx, panlib_1d(1),
                              PANLIB_BARRIER_JM_BARRIER, args);

   if (!had_batch)
      panvk_per_arch(cmd_close_batch)(cmdbuf);

   cmdbuf->state.gfx.xfb.offsets = offsets.gpu;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdEndTransformFeedbackEXT)(
   VkCommandBuffer commandBuffer, uint32_t firstCounterBuffer,
   uint32_t counterBufferCount, const VkBuffer *pCounterBuffers,
   const VkDeviceSize *pCounterBufferOffsets)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   uint64_t offsets = cmdbuf->state.gfx.xfb.offsets;

   cmdbuf->state.gfx.xfb.offsets = 0;
   if (!offsets)
      return;

   uint64_t counters[PANVK_XFB_MAX_BUFFERS];
   get_counters(firstCounterBuffer, counterBufferCount, pCounterBuffers,
                pCounterBufferOffsets, counters);
   if (!(counters[0] | counters[1] | counters[2] | counters[3]))
      return;

   bool had_batch = cmdbuf->cur_batch != NULL;
   if (!had_batch)
      panvk_per_arch(cmd_open_batch)(cmdbuf);

   /* The barrier orders this after the XFB jobs of the draws. */
   const struct panlib_jm_xfb_end_args args = {
      .offsets = offsets,
      .counter0 = counters[0],
      .counter1 = counters[1],
      .counter2 = counters[2],
      .counter3 = counters[3],
   };
   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   panlib_jm_xfb_end_struct(&precomp_ctx, panlib_1d(1),
                            PANLIB_BARRIER_JM_BARRIER, args);

   if (!had_batch)
      panvk_per_arch(cmd_close_batch)(cmdbuf);
}

/* transformFeedbackDraw is not supported. */
VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndirectByteCountEXT)(
   VkCommandBuffer commandBuffer, uint32_t instanceCount,
   uint32_t firstInstance, VkBuffer counterBuffer,
   VkDeviceSize counterBufferOffset, uint32_t counterOffset,
   uint32_t vertexStride)
{
   UNREACHABLE("transformFeedbackDraw is not supported");
}
