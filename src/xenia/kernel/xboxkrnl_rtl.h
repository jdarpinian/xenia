/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_XBOXKRNL_RTL_H_
#define XENIA_KERNEL_XBOXKRNL_RTL_H_

#include <xenia/common.h>
#include <xenia/core.h>
#include <xenia/xbox.h>

namespace xe {
namespace kernel {

struct X_RTL_CRITICAL_SECTION;

void xeRtlInitializeCriticalSection(X_RTL_CRITICAL_SECTION* cs);
X_STATUS xeRtlInitializeCriticalSectionAndSpinCount(X_RTL_CRITICAL_SECTION* cs,
                                                    uint32_t spin_count);
void xeRtlEnterCriticalSection(X_RTL_CRITICAL_SECTION* cs, uint32_t thread_id);
uint32_t xeRtlTryEnterCriticalSection(X_RTL_CRITICAL_SECTION* cs,
                                      uint32_t thread_id);
void xeRtlLeaveCriticalSection(X_RTL_CRITICAL_SECTION* cs);

}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_XBOXKRNL_RTL_H_
