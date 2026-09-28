/*
 * Copyright © 2024 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_CMD_QUERY_H
#define PANVK_CMD_QUERY_H

#ifndef PAN_ARCH
#error "PAN_ARCH must be defined"
#endif

#include "genxml/gen_macros.h"
#include "panvk_macros.h"

struct panvk_occlusion_query_state {
#if PAN_ARCH >= 10
   uint64_t syncobj;
#endif
   uint64_t ptr;
   enum mali_occlusion_mode mode;
};

struct panvk_prims_generated_query_state {
#if PAN_ARCH >= 10
   uint64_t syncobj;
#endif
   uint64_t ptr;
#if PAN_ARCH < 10
   /* Primitives of direct draws, counted on the CPU and added to the report
    * when the query ends. */
   uint32_t cpu_count;
   /* Whether a GPU job added to the report, see the JM EndQuery. */
   bool gpu_count;
#endif
};

#if PAN_ARCH < 10
struct panvk_cmd_buffer;

void panvk_per_arch(cmd_update_prims_generated_query)(
   struct panvk_cmd_buffer *cmd, uint64_t draw_cmd, uint64_t index_buffer,
   uint32_t index_buffer_size_el, uint32_t index_size, unsigned prim,
   uint32_t cpu_count);
#endif

#endif
