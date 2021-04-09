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
bool get_post_order(size_t node_idx, Container const& nodes, std::unordered_map<std::string, size_t> const& node_map,
    std::vector<NodeState>* node_states, std::vector<size_t>* order)
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
        // TODO: This .Get().input() is highly specific to protobuf, should
        //       generalise it somehow.
        for (auto const& input : nodes.Get(node_idx).input())
        {
            if (!node_map.count(input))
            {
                // Input node not found in graph!
                // cerr << "ERROR: Input node not found in graph: "
                //     << input << endl;
                // return false;
                continue; // Skip missing input edges
            }
            size_t input_node_idx = node_map.at(input);
            if (!get_post_order(input_node_idx, nodes, node_map, node_states, order))
            {
                return false;
            }
        }
        node_state = NODE_VISITED;
        order->push_back(node_idx);
    }
    return true;
}

} // anonymous namespace

template <class Container>
bool toposort(Container const& nodes, std::vector<size_t>* order)
{
    std::unordered_map<std::string, size_t> node_map;
    for (size_t i = 0; i < (size_t) nodes.size(); ++i)
    {
        // TODO: This .Get().input() is highly specific to protobuf, should
        //       generalise it somehow.
        for (auto const& output : nodes.Get(i).output())
        {
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
        if (!get_post_order(i, nodes, node_map, &node_states, order))
        {
            return false;
        }
    }
    return true;
}
