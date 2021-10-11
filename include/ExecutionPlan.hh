#pragma once

#include <utility>
#include <vector>
#include <set>
#include "VertexSet.h"
#include "cmap.h"

struct uidType {
    static const int64_t MIN = -1;
    static const int64_t MAX = -2;

    int64_t id;
    uidType(size_t uid) : id(uid) {}
    uidType(int64_t uid) : id(uid) {}
    bool is_bounded() const {
        return id >= 0;
    }
    bool is_min() const {
        return id == MIN;
    }
    bool is_max() const {
        return id == MAX;
    }
};

// parameters in cmap update policy
enum UPDATE_RANGE { ALL, FILTERED };        // which bucket to update: all or only filtered vertices
                                            // TODO: other cases e.g. LOWER, UPPER
enum UPDATE_OP { COMP_SET, AND_OR, NO_OP }; // when a bucket will be updated:
                                            //  COMP(field, cond_value) == true or AND(field, cond_value) != 0
                                            //  - COMP: the bucket will be set a new value
                                            //  - AND: the bucket will be OR'd a new value

template <typename VT>
struct UpdatePolicy {
    bool update_cmap;
    UPDATE_RANGE range;
    UPDATE_OP opcode;
    VT cond_value;
    VT upd_value;

    inline bool update_cond(VT bucket_value) const {
        switch (opcode) {
            case COMP_SET:
                return bucket_value == cond_value;
            case AND_OR:
                return static_cast<bool>(bucket_value & cond_value);
            default:
                return false;
        }
    }

    inline VT update_op(VT bucket_value) const {
        switch (opcode) {
            case COMP_SET:
                return upd_value;
            case AND_OR:
                return bucket_value | upd_value;
            default:
                return bucket_value;
        }
    }

    inline bool restore_cond(VT bucket_value) const {
        switch (opcode) {
            case COMP_SET:
                return bucket_value == upd_value;
            case AND_OR:
                return true;
            default:
                return false;
        }
    }

    inline VT restore_op(VT bucket_value) const {
        switch (opcode) {
            case COMP_SET:
                return cond_value;
            case AND_OR:
                return bucket_value & (~upd_value);
            default:
                return bucket_value;
        }
    }

};

struct Rule {
    // partial order
    std::pair<uidType, uidType> bound;
    // connection info
    std::set<uidType> connected;
    std::set<uidType> disconnected;
};

template <typename idT>
struct EncodedRule {
    idT adj_id;     // the base vertex id to extend from
    idT upd_id;     // the vertex id used to update cmap
    idT lower_id;
    idT upper_id;
    UpdatePolicy<uint8_t> policy;
};

template <>
struct EncodedRule<uidType> {
    uidType src_id;
    uidType lower_id;
    uidType upper_id;
    UpdatePolicy<uint8_t> policy;

    EncodedRule<vidType> inst_with(const std::vector<vidType> &history) {
        return { history.at(src_id),
                 lower_id.is_bounded() ? MIN_VID : history.at(lower_id),
                 upper_id.is_bounded() ? MAX_VID : history.at(upper_id),
                 policy
        };
    }
};

using EncodedRuleAbst = EncodedRule<uidType>;
using EncodedRuleInst = EncodedRule<vidType>;

using embidType = size_t;
using EmbRelation = std::pair<embidType, embidType>;
using EmbExtension = std::pair<embidType, Rule>;

// an execution plan for matching a specific pattern
// will be passed to a graph pattern mining template hardware / software
class ExecutionPlan {
private:
    // RI:
    // rules.size() + 1 equals the pattern size
    // rules[i].src_id <= i for 0 <= i < rules.size()
    // uid in rules[i].connected and rules[i].disconnected <= i
    std::vector<EmbRelation> relations;
    std::vector<EncodedRuleAbst> extend_rules;

public:

    ExecutionPlan(const std::vector<EmbExtension> &extensions) {
        // TODO transformation optimization
    }

    size_t pattern_size() {
        return extend_rules.size() + 1;
    }

    // @param level     >=1, the level of rule
    const EncodedRuleAbst &rule_at(size_t level) {
        return extend_rules.at(level-1);
    }
}

