/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_map>
#include <vector>

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

namespace
{

enum NodeState
{
    NODE_UNVISITED,
    NODE_ACTIVE,
    NODE_VISITED
};

template <class Container>
[[nodiscard]] bool get_post_order(size_t node_idx, Container const& nodes,
    std::unordered_map<std::string, size_t> const& node_map,
    std::unordered_map<size_t, std::vector<std::string>> const& extra_deps, std::vector<NodeState>* node_states,
    std::vector<size_t>* order)
{
    NodeState& node_state = node_states->at(node_idx);
    if (node_state == NODE_ACTIVE)
    {
        // Cycle detected!
        cerr << "ERROR: Graph contains a cycle" << endl;
        return false;
    }
    else if (node_state == NODE_VISITED)
    {
        return true;
    }
    else
    {
        node_state = NODE_ACTIVE;
        for (auto const& input : nodes[node_idx].input())
        {
            auto const inputIt = node_map.find(input);
            if (inputIt == node_map.end())
            {
                // Input node not found in graph — skip missing input edges.
                continue;
            }
            size_t input_node_idx = inputIt->second;
            if (!get_post_order(input_node_idx, nodes, node_map, extra_deps, node_states, order))
            {
                return false;
            }
        }
        // Follow extra dependencies (e.g. outer-scope tensor references from subgraph attributes).
        auto it = extra_deps.find(node_idx);
        if (it != extra_deps.end())
        {
            for (auto const& dep : it->second)
            {
                auto const depIt = node_map.find(dep);
                if (depIt == node_map.end())
                {
                    continue;
                }
                size_t dep_node_idx = depIt->second;
                if (!get_post_order(dep_node_idx, nodes, node_map, extra_deps, node_states, order))
                {
                    return false;
                }
            }
        }
        node_state = NODE_VISITED;
        order->push_back(node_idx);
    }
    return true;
}

} // anonymous namespace

//! Topologically sort \p nodes, storing the resulting node index order in \p order.
//! \p extra_deps maps node indices to additional tensor names they implicitly depend on
//! (e.g. outer-scope references inside subgraph attributes like If/Loop/Scan).
//! \return false if a cycle is detected, true otherwise.
template <class Container>
[[nodiscard]] bool toposort(Container const& nodes, std::vector<size_t>* order,
    std::unordered_map<size_t, std::vector<std::string>> const& extra_deps = {})
{
    std::unordered_map<std::string, size_t> node_map;
    for (size_t i = 0; i < (size_t) nodes.size(); ++i)
    {
        for (auto const& output : nodes[i].output())
        {
            // Empty output strings mean null outputs, do not register them.
            if (output.empty())
            {
                continue;
            }
            if (!node_map.emplace(output, i).second)
            {
                // Output name appears more than once in graph!
                cerr << "ERROR: Output name is not unique: " << output << endl;
                return false;
            }
        }
    }
    order->reserve(nodes.size());
    std::vector<NodeState> node_states(nodes.size(), NODE_UNVISITED);
    for (size_t i = 0; i < (size_t) nodes.size(); ++i)
    {
        if (!get_post_order(i, nodes, node_map, extra_deps, &node_states, order))
        {
            return false;
        }
    }
    return true;
}
