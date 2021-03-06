/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_KERNEL_STATE_H_
#define XENIA_KERNEL_KERNEL_STATE_H_

#include <memory>
#include <mutex>

#include <xenia/common.h>
#include <xenia/core.h>

#include <xenia/export_resolver.h>
#include <xenia/xbox.h>
#include <xenia/kernel/app.h>
#include <xenia/kernel/object_table.h>
#include <xenia/kernel/user_profile.h>
#include <xenia/kernel/fs/filesystem.h>

namespace xe {
class Emulator;
namespace cpu {
class Processor;
}  // namespace cpu
}  // namespace xe

namespace xe {
namespace kernel {

class Dispatcher;
class XModule;
class XNotifyListener;
class XThread;
class XUserModule;

class KernelState {
 public:
  KernelState(Emulator* emulator);
  ~KernelState();

  static KernelState* shared();

  Emulator* emulator() const { return emulator_; }
  Memory* memory() const { return memory_; }
  cpu::Processor* processor() const { return processor_; }
  fs::FileSystem* file_system() const { return file_system_; }

  Dispatcher* dispatcher() const { return dispatcher_; }

  XAppManager* app_manager() const { return app_manager_.get(); }
  UserProfile* user_profile() const { return user_profile_.get(); }

  ObjectTable* object_table() const { return object_table_; }
  std::mutex& object_mutex() { return object_mutex_; }

  void RegisterModule(XModule* module);
  void UnregisterModule(XModule* module);
  XModule* GetModule(const char* name);
  XUserModule* GetExecutableModule();
  void SetExecutableModule(XUserModule* module);

  void RegisterThread(XThread* thread);
  void UnregisterThread(XThread* thread);
  XThread* GetThreadByID(uint32_t thread_id);

  void RegisterNotifyListener(XNotifyListener* listener);
  void UnregisterNotifyListener(XNotifyListener* listener);
  void BroadcastNotification(XNotificationID id, uint32_t data);

  void CompleteOverlapped(uint32_t overlapped_ptr, X_RESULT result,
                          uint32_t length = 0);
  void CompleteOverlappedImmediate(uint32_t overlapped_ptr, X_RESULT result,
                                   uint32_t length = 0);

 private:
  Emulator* emulator_;
  Memory* memory_;
  cpu::Processor* processor_;
  fs::FileSystem* file_system_;

  Dispatcher* dispatcher_;

  std::unique_ptr<XAppManager> app_manager_;
  std::unique_ptr<UserProfile> user_profile_;

  ObjectTable* object_table_;
  std::mutex object_mutex_;
  std::unordered_map<uint32_t, XThread*> threads_by_id_;
  std::vector<XNotifyListener*> notify_listeners_;

  XUserModule* executable_module_;

  friend class XObject;
};

}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_KERNEL_STATE_H_
