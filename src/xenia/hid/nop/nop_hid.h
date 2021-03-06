/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_HID_NOP_NOP_HID_H_
#define XENIA_HID_NOP_NOP_HID_H_

#include <xenia/core.h>

XEDECLARECLASS2(xe, hid, InputDriver);
XEDECLARECLASS2(xe, hid, InputSystem);

namespace xe {
namespace hid {
namespace nop {

InputDriver* Create(InputSystem* input_system);

}  // namespace nop
}  // namespace hid
}  // namespace xe

#endif  // XENIA_HID_NOP_NOP_HID_H_
