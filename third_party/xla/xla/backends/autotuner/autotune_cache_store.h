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

#ifndef XLA_BACKENDS_AUTOTUNER_AUTOTUNE_CACHE_STORE_H_
#define XLA_BACKENDS_AUTOTUNER_AUTOTUNE_CACHE_STORE_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/autotuning.pb.h"

namespace xla {

// A proto-keyed key-value store for autotune entries. Implementations only
// store and retrieve protos.
class AutotuneCacheStore {
 public:
  using CacheMode = xla::CacheMode;

  virtual ~AutotuneCacheStore() = default;

  // Returns all entries whose target matches (device, explicit_version,
  // hlo_fingerprint). The caller performs strict/loose selection.
  virtual absl::StatusOr<std::vector<autotuner::AutotuneEntry>> Read(
      const autotuner::AutotuneTargetKey& target_key) = 0;

  // Inserts or updates an entry (dedup on full AutotuneKey).
  virtual absl::Status Write(const autotuner::AutotuneEntry& entry) = 0;

  // Enumerates all entries in the store.
  virtual absl::StatusOr<std::vector<autotuner::AutotuneEntry>> ReadAll() = 0;

  virtual CacheMode GetMode() const = 0;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_AUTOTUNE_CACHE_STORE_H_
