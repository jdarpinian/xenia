/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <xenia/kernel/fs/devices/host_path_device.h>

#include <xenia/kernel/fs/devices/host_path_entry.h>
#include <xenia/kernel/objects/xfile.h>

namespace xe {
namespace kernel {
namespace fs {

HostPathDevice::HostPathDevice(const std::string& path,
                               const std::wstring& local_path)
    : Device(path), local_path_(local_path) {}

HostPathDevice::~HostPathDevice() {}

Entry* HostPathDevice::ResolvePath(const char* path) {
  // The filesystem will have stripped our prefix off already, so the path will
  // be in the form:
  // some\PATH.foo

  XELOGFS("HostPathDevice::ResolvePath(%s)", path);

  auto rel_path = poly::to_wstring(path);
  auto full_path = poly::join_paths(local_path_, rel_path);
  full_path = poly::fix_path_separators(full_path);

  // TODO(benvanik): get file info
  // TODO(benvanik): fail if does not exit
  // TODO(benvanik): switch based on type

  auto type = Entry::Type::FILE;
  HostPathEntry* entry = new HostPathEntry(type, this, path, full_path);
  return entry;
}

// TODO(gibbed): call into HostPathDevice?
X_STATUS HostPathDevice::QueryVolume(XVolumeInfo* out_info, size_t length) {
  assert_not_null(out_info);
  const char* name = "test";  // TODO(gibbed): actual value

  auto end = (uint8_t*)out_info + length;
  size_t name_length = strlen(name);
  if (((uint8_t*)&out_info->label[0]) + name_length > end) {
    return X_STATUS_BUFFER_OVERFLOW;
  }

  out_info->creation_time = 0;
  out_info->serial_number = 12345678;
  out_info->supports_objects = 0;
  out_info->label_length = (uint32_t)name_length;
  memcpy(out_info->label, name, name_length);
  return X_STATUS_SUCCESS;
}

// TODO(gibbed): call into HostPathDevice?
X_STATUS HostPathDevice::QueryFileSystemAttributes(
    XFileSystemAttributeInfo* out_info, size_t length) {
  assert_not_null(out_info);
  const char* name = "test";  // TODO(gibbed): actual value

  auto end = (uint8_t*)out_info + length;
  size_t name_length = strlen(name);
  if (((uint8_t*)&out_info->fs_name[0]) + name_length > end) {
    return X_STATUS_BUFFER_OVERFLOW;
  }

  out_info->attributes = 0;
  out_info->maximum_component_name_length = 255;  // TODO(gibbed): actual value
  out_info->fs_name_length = (uint32_t)name_length;
  memcpy(out_info->fs_name, name, name_length);
  return X_STATUS_SUCCESS;
}

}  // namespace fs
}  // namespace kernel
}  // namespace xe
