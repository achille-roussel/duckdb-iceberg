#pragma once
#include "metadata/iceberg_transform.hpp"
#include "metadata/iceberg_predicate_stats.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

struct IcebergPredicate {
public:
	IcebergPredicate() = delete;

public:
	//! Match a TableFilter against file/column bounds
	static bool MatchBounds(ClientContext &context, const TableFilter &filter, const IcebergPredicateStats &stats,
	                        const IcebergTransform &transform);

	//! Match an Expression directly against file/column bounds (for complex expressions
	//! that FilterCombiner drops, such as complex OR patterns, NOT BETWEEN, etc.)
	static bool MatchBoundsFromExpression(ClientContext &context, const Expression &expr,
	                                      const IcebergPredicateStats &stats, const IcebergTransform &transform);
};

} // namespace duckdb
