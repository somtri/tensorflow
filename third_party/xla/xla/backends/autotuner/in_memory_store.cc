/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/autotuner/in_memory_store.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/autotuner/autotuning.pb.h"

namespace xla {

// Define the process-wide static storage.
absl::Mutex InMemoryStore::mutex_;
absl::flat_hash_map<TargetKey, std::vector<autotuner::AutotuneEntry>>
    InMemoryStore::entries_;

namespace {

TargetKey ToTargetKey(const autotuner::AutotuneTargetKey& proto) {
  return TargetKey{proto.device(), proto.explicit_version(),
                   proto.hlo_fingerprint()};
}

// Returns true if the two full AutotuneKeys refer to the same cache slot (same
// target and environment).
bool SameKey(const autotuner::AutotuneKey& a, const autotuner::AutotuneKey& b) {
  return a.target().device() == b.target().device() &&
         a.target().explicit_version() == b.target().explicit_version() &&
         a.target().hlo_fingerprint() == b.target().hlo_fingerprint() &&
         a.environment().codegen_version() ==
             b.environment().codegen_version() &&
         a.environment().codegen_options_fingerprint() ==
             b.environment().codegen_options_fingerprint();
}

}  // namespace

absl::StatusOr<std::vector<autotuner::AutotuneEntry>> InMemoryStore::Read(
    const autotuner::AutotuneTargetKey& target_key) {
  absl::MutexLock lock(mutex_);
  auto it = entries_.find(ToTargetKey(target_key));
  if (it == entries_.end()) {
    return std::vector<autotuner::AutotuneEntry>{};
  }
  return it->second;
}

absl::Status InMemoryStore::Write(const autotuner::AutotuneEntry& entry) {
  absl::MutexLock lock(mutex_);
  std::vector<autotuner::AutotuneEntry>& bucket =
      entries_[ToTargetKey(entry.key().target())];
  for (autotuner::AutotuneEntry& existing : bucket) {
    if (SameKey(existing.key(), entry.key())) {
      *existing.mutable_value() = entry.value();
      return absl::OkStatus();
    }
  }
  bucket.push_back(entry);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<autotuner::AutotuneEntry>> InMemoryStore::ReadAll() {
  absl::MutexLock lock(mutex_);
  std::vector<autotuner::AutotuneEntry> all;
  for (const auto& [key, bucket] : entries_) {
    all.insert(all.end(), bucket.begin(), bucket.end());
  }
  return all;
}

void InMemoryStore::Clear() {
  absl::MutexLock lock(mutex_);
  entries_.clear();
}

}  // namespace xla
