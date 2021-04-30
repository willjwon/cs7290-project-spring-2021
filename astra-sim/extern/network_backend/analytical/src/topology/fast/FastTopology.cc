/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "FastTopology.hh"

using namespace Analytical;

FastTopology::FastTopology(TopologyConfigs& configs) noexcept
    : Topology(configs) {}

FastTopology::~FastTopology() noexcept = default;

FastTopology::Latency FastTopology::linkLatency(int dimension, int hops_count)
    const noexcept {
  return hops_count * configs[dimension].getLinkLatency();
}
