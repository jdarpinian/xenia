/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_XBOXKRNL_ORDINALS_H_
#define XENIA_KERNEL_XBOXKRNL_ORDINALS_H_

#include <xenia/common.h>
#include <xenia/core.h>
#include <xenia/export_resolver.h>

// Build an ordinal enum to make it easy to lookup ordinals.
#include <xenia/kernel/util/ordinal_table_pre.inc>
namespace ordinals {
enum {
#include <xenia/kernel/xboxkrnl_table.inc>
};
}  // namespace ordinals
#include <xenia/kernel/util/ordinal_table_post.inc>

#endif  // XENIA_KERNEL_XBOXKRNL_ORDINALS_H_
