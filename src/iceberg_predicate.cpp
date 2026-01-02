#include "iceberg_predicate.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/null_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"

#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/execution/expression_executor.hpp"

namespace duckdb {

namespace {

struct BoundExpressionReplacer : public LogicalOperatorVisitor {
public:
	BoundExpressionReplacer(const Value &val) : val(val) {
	}

public:
	unique_ptr<Expression> VisitReplace(BoundReferenceExpression &expr, unique_ptr<Expression> *expr_ptr) override {
		if (expr.index != 0) {
			return nullptr;
		}
		return make_uniq<BoundConstantExpression>(val);
	}

public:
	const Value &val;
};

} // namespace

template <class TRANSFORM>
bool MatchBoundsTemplated(ClientContext &context, const TableFilter &filter, const IcebergPredicateStats &stats,
                          const IcebergTransform &transform);

template <class TRANSFORM>
static bool MatchBoundsConstant(const Value &constant, ExpressionType comparison_type,
                                const IcebergPredicateStats &stats, const IcebergTransform &transform) {
	auto constant_value = TRANSFORM::ApplyTransform(constant, transform);

	if (stats.BoundsAreNull()) {
		// bounds are actually null, expression is not a null comparison expression
		// those are handled in MatchBoundsTemplated
		// So we can return false since no remaining expression type will match a null value
		D_ASSERT(comparison_type != ExpressionType::OPERATOR_IS_NOT_NULL);
		D_ASSERT(comparison_type != ExpressionType::OPERATOR_IS_NULL);
		D_ASSERT(comparison_type != ExpressionType::COMPARE_DISTINCT_FROM);
		D_ASSERT(comparison_type != ExpressionType::COMPARE_NOT_DISTINCT_FROM);
		return false;
	}

	if (!stats.has_upper_bounds || !stats.has_lower_bounds) {
		// we do not have upper or lower bounds, assume the file matches.
		return true;
	}

	switch (comparison_type) {
	case ExpressionType::COMPARE_EQUAL:
		return TRANSFORM::CompareEqual(constant_value, stats);
	case ExpressionType::COMPARE_GREATERTHAN:
		return TRANSFORM::CompareGreaterThan(constant_value, stats);
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return TRANSFORM::CompareGreaterThanOrEqual(constant_value, stats);
	case ExpressionType::COMPARE_LESSTHAN:
		return TRANSFORM::CompareLessThan(constant_value, stats);
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return TRANSFORM::CompareLessThanOrEqual(constant_value, stats);
	case ExpressionType::COMPARE_NOTEQUAL:
		//! Can only prune if entire file contains exactly this value (lower == upper == value)
		return !(constant_value == stats.lower_bound && constant_value == stats.upper_bound);
	default:
		//! Conservative approach: we don't know, so we just say it's not filtered out
		return true;
	}
}

template <class TRANSFORM>
static bool MatchBoundsConstantFilter(const ConstantFilter &constant_filter, const IcebergPredicateStats &stats,
                                      const IcebergTransform &transform) {
	return MatchBoundsConstant<TRANSFORM>(constant_filter.constant, constant_filter.comparison_type, stats, transform);
}

template <class TRANSFORM>
static bool MatchBoundsIsNullFilter(const IcebergPredicateStats &stats, const IcebergTransform &transform) {
	return stats.has_null == true;
}

template <class TRANSFORM>
static bool MatchBoundsIsNotNullFilter(const IcebergPredicateStats &stats, const IcebergTransform &transform) {
	return stats.has_not_null == true;
}

template <class TRANSFORM>
static bool MatchBoundsConjunctionAndFilter(ClientContext &context, const ConjunctionAndFilter &conjunction_and,
                                            const IcebergPredicateStats &stats, const IcebergTransform &transform) {
	for (auto &child : conjunction_and.child_filters) {
		if (!MatchBoundsTemplated<TRANSFORM>(context, *child, stats, transform)) {
			return false;
		}
	}
	return true;
}

template <class TRANSFORM>
bool MatchTransformedBounds(ClientContext &context, ExpressionType comparison_type, const Expression &left,
                            const Expression &right, const IcebergPredicateStats &stats,
                            const IcebergTransform &transform) {
	BoundExpressionReplacer lower_replacer(stats.lower_bound);
	BoundExpressionReplacer upper_replacer(stats.upper_bound);
	auto lower_copy = left.Copy();
	auto upper_copy = left.Copy();
	lower_replacer.VisitExpression(&lower_copy);
	upper_replacer.VisitExpression(&upper_copy);

	Value right_constant;
	if (!ExpressionExecutor::TryEvaluateScalar(context, right, right_constant)) {
		return true;
	}

	Value transformed_lower_bound;
	Value transformed_upper_bound;
	if (!ExpressionExecutor::TryEvaluateScalar(context, *lower_copy, transformed_lower_bound)) {
		return true;
	}
	if (!ExpressionExecutor::TryEvaluateScalar(context, *upper_copy, transformed_upper_bound)) {
		return true;
	}
	IcebergPredicateStats transformed_stats(stats);
	transformed_stats.lower_bound = transformed_lower_bound;
	transformed_stats.upper_bound = transformed_upper_bound;

	return MatchBoundsConstant<TRANSFORM>(right_constant, comparison_type, transformed_stats, transform);
}

template <class TRANSFORM>
bool MatchBoundsTemplated(ClientContext &context, const TableFilter &filter, const IcebergPredicateStats &stats,
                          const IcebergTransform &transform) {
	//! TODO: support more filter types
	switch (filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = filter.Cast<ConstantFilter>();
		return MatchBoundsConstantFilter<TRANSFORM>(constant_filter, stats, transform);
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &conjunction_and_filter = filter.Cast<ConjunctionAndFilter>();
		return MatchBoundsConjunctionAndFilter<TRANSFORM>(context, conjunction_and_filter, stats, transform);
	}
	case TableFilterType::CONJUNCTION_OR: {
		auto &conjunction_or_filter = filter.Cast<ConjunctionOrFilter>();
		//! For OR: file matches if ANY child filter matches
		for (auto &child : conjunction_or_filter.child_filters) {
			if (MatchBoundsTemplated<TRANSFORM>(context, *child, stats, transform)) {
				return true;
			}
		}
		return false;
	}
	case TableFilterType::IS_NULL: {
		//! FIXME: these are never hit, because it goes through ExpressionFilter instead?
		return MatchBoundsIsNullFilter<TRANSFORM>(stats, transform);
	}
	case TableFilterType::IS_NOT_NULL: {
		//! FIXME: these are never hit, because it goes through ExpressionFilter instead?
		return MatchBoundsIsNotNullFilter<TRANSFORM>(stats, transform);
	}
	case TableFilterType::OPTIONAL_FILTER: {
		auto &optional_filter = filter.Cast<OptionalFilter>();
		if (optional_filter.child_filter) {
			return MatchBoundsTemplated<TRANSFORM>(context, *optional_filter.child_filter, stats, transform);
		}
		//! child filter wasn't populated (yet?) for some reason, just be conservative
		return true;
	}
	case TableFilterType::IN_FILTER: {
		auto &in_filter = filter.Cast<InFilter>();
		D_ASSERT(!in_filter.values.empty());
		for (auto &value : in_filter.values) {
			if (MatchBoundsConstant<TRANSFORM>(value, ExpressionType::COMPARE_EQUAL, stats, transform)) {
				return true;
			}
		}
		return false;
	}
	case TableFilterType::EXPRESSION_FILTER: {
		//! Expressions can be arbitrarily complex, and we currently only support IS NULL/IS NOT NULL checks against the
		//! column itself, i.e. where the expression is a BOUND_OPERATOR with type OPERATOR_IS_NULL/_IS_NOT_NULL with a
		//! single child expression of type BOUND_REF.
		//!
		//! See duckdb/duckdb-iceberg#464
		auto &expression_filter = filter.Cast<ExpressionFilter>();
		auto &expr = *expression_filter.expr;

		switch (expr.type) {
		case ExpressionType::OPERATOR_IS_NULL:
		case ExpressionType::OPERATOR_IS_NOT_NULL: {
			D_ASSERT(expr.GetExpressionClass() == ExpressionClass::BOUND_OPERATOR);
			auto &bound_operator_expr = expr.Cast<BoundOperatorExpression>();

			D_ASSERT(bound_operator_expr.children.size() == 1);
			auto &child_expr = bound_operator_expr.children[0];
			if (child_expr->type != ExpressionType::BOUND_REF) {
				//! We can't evaluate expressions that aren't direct column references
				return true;
			}

			if (expr.type == ExpressionType::OPERATOR_IS_NULL) {
				return MatchBoundsIsNullFilter<TRANSFORM>(stats, transform);
			}
			D_ASSERT(expr.type == ExpressionType::OPERATOR_IS_NOT_NULL);
			return MatchBoundsIsNotNullFilter<TRANSFORM>(stats, transform);
		}
		case ExpressionType::COMPARE_BETWEEN: {
			D_ASSERT(expr.GetExpressionClass() == ExpressionClass::BOUND_BETWEEN);
			auto &between_expr = expr.Cast<BoundBetweenExpression>();

			//! Input must be a column reference
			if (between_expr.input->type != ExpressionType::BOUND_REF) {
				return true;
			}

			//! Both bounds must be foldable (constants)
			if (!between_expr.lower->IsFoldable() || !between_expr.upper->IsFoldable()) {
				return true;
			}

			Value lower_val, upper_val;
			if (!ExpressionExecutor::TryEvaluateScalar(context, *between_expr.lower, lower_val) ||
			    !ExpressionExecutor::TryEvaluateScalar(context, *between_expr.upper, upper_val)) {
				return true;
			}

			//! File matches if ranges overlap: file.lower <= query.upper AND file.upper >= query.lower
			auto lower_cmp = between_expr.lower_inclusive ? ExpressionType::COMPARE_GREATERTHANOREQUALTO
			                                              : ExpressionType::COMPARE_GREATERTHAN;
			auto upper_cmp = between_expr.upper_inclusive ? ExpressionType::COMPARE_LESSTHANOREQUALTO
			                                              : ExpressionType::COMPARE_LESSTHAN;

			return MatchBoundsConstant<TRANSFORM>(lower_val, lower_cmp, stats, transform) &&
			       MatchBoundsConstant<TRANSFORM>(upper_val, upper_cmp, stats, transform);
		}
		case ExpressionType::COMPARE_GREATERTHAN:
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		case ExpressionType::COMPARE_LESSTHAN:
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		case ExpressionType::COMPARE_EQUAL:
		case ExpressionType::COMPARE_NOTEQUAL: {
			D_ASSERT(expr.GetExpressionClass() == ExpressionClass::BOUND_COMPARISON);
			auto &compare_expr = expr.Cast<BoundComparisonExpression>();
			if (transform.Type() == IcebergTransformType::IDENTITY) {
				//! No further processing has been done on the stats (lower/upper bounds)
				auto &left = *compare_expr.left;
				auto &right = *compare_expr.right;

				bool left_foldable = left.IsFoldable();
				bool right_foldable = right.IsFoldable();
				if (!left_foldable && !right_foldable) {
					//! Both are not foldable, can't evaluate at all
					return true;
				}

				if (left_foldable) {
					return MatchTransformedBounds<TRANSFORM>(context, expr.type, right, left, stats, transform);
				} else {
					return MatchTransformedBounds<TRANSFORM>(context, expr.type, left, right, stats, transform);
				}
				return true;
			}
		}
		case ExpressionType::BOUND_FUNCTION: {
			//! Handle prefix/starts_with functions for LIKE 'prefix%' optimization
			auto &func_expr = expr.Cast<BoundFunctionExpression>();

			if (func_expr.function.name == "prefix" || func_expr.function.name == "starts_with") {
				if (func_expr.children.size() != 2) {
					return true;
				}

				//! First child should be column reference
				if (func_expr.children[0]->type != ExpressionType::BOUND_REF) {
					return true;
				}

				//! Second child should be constant prefix string
				if (!func_expr.children[1]->IsFoldable()) {
					return true;
				}

				Value prefix_val;
				if (!ExpressionExecutor::TryEvaluateScalar(context, *func_expr.children[1], prefix_val)) {
					return true;
				}

				if (prefix_val.IsNull() || stats.BoundsAreNull() || !stats.has_upper_bounds ||
				    !stats.has_lower_bounds) {
					return true;
				}

				auto prefix_str = prefix_val.ToString();
				if (prefix_str.empty()) {
					return true;
				}

				auto lower_str = stats.lower_bound.ToString();
				auto upper_str = stats.upper_bound.ToString();

				//! Prune if upper_bound < prefix (no strings can start with prefix)
				if (upper_str < prefix_str) {
					return false;
				}

				//! Calculate next_prefix (increment last char)
				string next_prefix = prefix_str;
				next_prefix.back()++;

				//! Prune if lower_bound >= next_prefix
				if (lower_str >= next_prefix) {
					return false;
				}

				return true;
			}
			return true;
		}
		case ExpressionType::COMPARE_NOT_IN: {
			//! NOT IN can only prune if file has single value that's in the exclusion list
			D_ASSERT(expr.GetExpressionClass() == ExpressionClass::BOUND_OPERATOR);
			auto &operator_expr = expr.Cast<BoundOperatorExpression>();

			if (operator_expr.children.empty()) {
				return true;
			}

			//! First child should be column reference
			if (operator_expr.children[0]->type != ExpressionType::BOUND_REF) {
				return true;
			}

			//! Can only prune if file has single value (lower == upper)
			if (stats.lower_bound != stats.upper_bound) {
				return true;
			}

			//! Check if the single value is in the exclusion list
			for (size_t i = 1; i < operator_expr.children.size(); i++) {
				Value child_val;
				if (!ExpressionExecutor::TryEvaluateScalar(context, *operator_expr.children[i], child_val)) {
					return true;
				}
				if (stats.lower_bound == child_val) {
					return false; //! Can prune: only value in file is excluded
				}
			}
			return true;
		}
		default:
			return true;
		}
	}
	default:
		//! Conservative approach: we don't know what this is, just say it doesn't filter anything
		return true;
	}
}

//! =============================================================================
//! Expression-based matching (for complex expressions that FilterCombiner drops)
//! =============================================================================

//! Forward declaration
template <class TRANSFORM>
static bool MatchBoundsFromExpressionTemplated(ClientContext &context, const Expression &expr,
                                               const IcebergPredicateStats &stats, const IcebergTransform &transform);

//! Helper: Match comparison expression (col op constant)
template <class TRANSFORM>
static bool MatchComparisonExpression(ClientContext &context, const Expression &expr,
                                      const IcebergPredicateStats &stats, const IcebergTransform &transform) {
	D_ASSERT(expr.GetExpressionClass() == ExpressionClass::BOUND_COMPARISON);
	auto &compare_expr = expr.Cast<BoundComparisonExpression>();

	auto &left = *compare_expr.left;
	auto &right = *compare_expr.right;

	bool left_foldable = left.IsFoldable();
	bool right_foldable = right.IsFoldable();
	if (!left_foldable && !right_foldable) {
		//! Both are not foldable, can't evaluate at all
		return true;
	}

	if (left_foldable) {
		Value left_val;
		if (!ExpressionExecutor::TryEvaluateScalar(context, left, left_val)) {
			return true;
		}
		//! Flip comparison type for constant on left side
		ExpressionType flipped_type;
		switch (expr.type) {
		case ExpressionType::COMPARE_LESSTHAN:
			flipped_type = ExpressionType::COMPARE_GREATERTHAN;
			break;
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			flipped_type = ExpressionType::COMPARE_GREATERTHANOREQUALTO;
			break;
		case ExpressionType::COMPARE_GREATERTHAN:
			flipped_type = ExpressionType::COMPARE_LESSTHAN;
			break;
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			flipped_type = ExpressionType::COMPARE_LESSTHANOREQUALTO;
			break;
		default:
			flipped_type = expr.type;
			break;
		}
		return MatchBoundsConstant<TRANSFORM>(left_val, flipped_type, stats, transform);
	} else {
		Value right_val;
		if (!ExpressionExecutor::TryEvaluateScalar(context, right, right_val)) {
			return true;
		}
		return MatchBoundsConstant<TRANSFORM>(right_val, expr.type, stats, transform);
	}
}

//! Helper: Match BETWEEN expression
template <class TRANSFORM>
static bool MatchBetweenExpression(ClientContext &context, const Expression &expr, const IcebergPredicateStats &stats,
                                   const IcebergTransform &transform) {
	D_ASSERT(expr.GetExpressionClass() == ExpressionClass::BOUND_BETWEEN);
	auto &between_expr = expr.Cast<BoundBetweenExpression>();

	//! Input must be a column reference
	if (between_expr.input->type != ExpressionType::BOUND_REF) {
		return true;
	}

	//! Both bounds must be foldable (constants)
	if (!between_expr.lower->IsFoldable() || !between_expr.upper->IsFoldable()) {
		return true;
	}

	Value lower_val, upper_val;
	if (!ExpressionExecutor::TryEvaluateScalar(context, *between_expr.lower, lower_val) ||
	    !ExpressionExecutor::TryEvaluateScalar(context, *between_expr.upper, upper_val)) {
		return true;
	}

	//! File matches if ranges overlap: file.lower <= query.upper AND file.upper >= query.lower
	auto lower_cmp = between_expr.lower_inclusive ? ExpressionType::COMPARE_GREATERTHANOREQUALTO
	                                              : ExpressionType::COMPARE_GREATERTHAN;
	auto upper_cmp =
	    between_expr.upper_inclusive ? ExpressionType::COMPARE_LESSTHANOREQUALTO : ExpressionType::COMPARE_LESSTHAN;

	return MatchBoundsConstant<TRANSFORM>(lower_val, lower_cmp, stats, transform) &&
	       MatchBoundsConstant<TRANSFORM>(upper_val, upper_cmp, stats, transform);
}

//! Helper: Match NOT BETWEEN expression
template <class TRANSFORM>
static bool MatchNotBetweenExpression(ClientContext &context, const Expression &expr,
                                      const IcebergPredicateStats &stats, const IcebergTransform &transform) {
	D_ASSERT(expr.GetExpressionClass() == ExpressionClass::BOUND_BETWEEN);
	auto &between_expr = expr.Cast<BoundBetweenExpression>();

	//! Input must be a column reference
	if (between_expr.input->type != ExpressionType::BOUND_REF) {
		return true;
	}

	//! Both bounds must be foldable (constants)
	if (!between_expr.lower->IsFoldable() || !between_expr.upper->IsFoldable()) {
		return true;
	}

	Value lower_val, upper_val;
	if (!ExpressionExecutor::TryEvaluateScalar(context, *between_expr.lower, lower_val) ||
	    !ExpressionExecutor::TryEvaluateScalar(context, *between_expr.upper, upper_val)) {
		return true;
	}

	if (stats.BoundsAreNull() || !stats.has_upper_bounds || !stats.has_lower_bounds) {
		return true;
	}

	auto file_lower = TRANSFORM::ApplyTransform(stats.lower_bound, transform);
	auto file_upper = TRANSFORM::ApplyTransform(stats.upper_bound, transform);

	//! Can prune if entire file is within the excluded range
	//! For inclusive BETWEEN: file.lower >= query.lower AND file.upper <= query.upper
	bool lower_inside = between_expr.lower_inclusive ? (file_lower >= lower_val) : (file_lower > lower_val);
	bool upper_inside = between_expr.upper_inclusive ? (file_upper <= upper_val) : (file_upper < upper_val);

	if (lower_inside && upper_inside) {
		return false; //! Can prune: entire file is within excluded range
	}

	return true;
}

//! Helper: Match starts_with/prefix function
template <class TRANSFORM>
static bool MatchStartsWithExpression(ClientContext &context, const BoundFunctionExpression &func_expr,
                                      const IcebergPredicateStats &stats, const IcebergTransform &transform) {
	if (func_expr.children.size() != 2) {
		return true;
	}

	//! First child should be column reference
	if (func_expr.children[0]->type != ExpressionType::BOUND_REF) {
		return true;
	}

	//! Second child should be constant prefix string
	if (!func_expr.children[1]->IsFoldable()) {
		return true;
	}

	Value prefix_val;
	if (!ExpressionExecutor::TryEvaluateScalar(context, *func_expr.children[1], prefix_val)) {
		return true;
	}

	if (prefix_val.IsNull() || stats.BoundsAreNull() || !stats.has_upper_bounds || !stats.has_lower_bounds) {
		return true;
	}

	auto prefix_str = prefix_val.ToString();
	if (prefix_str.empty()) {
		return true;
	}

	auto lower_str = stats.lower_bound.ToString();
	auto upper_str = stats.upper_bound.ToString();

	//! Prune if upper_bound < prefix (no strings can start with prefix)
	if (upper_str < prefix_str) {
		return false;
	}

	//! Calculate next_prefix (increment last char)
	string next_prefix = prefix_str;
	next_prefix.back()++;

	//! Prune if lower_bound >= next_prefix
	if (lower_str >= next_prefix) {
		return false;
	}

	return true;
}

//! Helper: Match ends_with/suffix function
template <class TRANSFORM>
static bool MatchEndsWithExpression(ClientContext &context, const BoundFunctionExpression &func_expr,
                                    const IcebergPredicateStats &stats, const IcebergTransform &transform) {
	if (func_expr.children.size() != 2) {
		return true;
	}

	//! First child should be column reference
	if (func_expr.children[0]->type != ExpressionType::BOUND_REF) {
		return true;
	}

	//! Second child should be constant suffix string
	if (!func_expr.children[1]->IsFoldable()) {
		return true;
	}

	Value suffix_val;
	if (!ExpressionExecutor::TryEvaluateScalar(context, *func_expr.children[1], suffix_val)) {
		return true;
	}

	if (suffix_val.IsNull() || stats.BoundsAreNull() || !stats.has_upper_bounds || !stats.has_lower_bounds) {
		return true;
	}

	auto suffix_str = suffix_val.ToString();
	if (suffix_str.empty()) {
		return true;
	}

	auto lower_str = stats.lower_bound.ToString();
	auto upper_str = stats.upper_bound.ToString();
	auto suffix_len = suffix_str.length();

	//! Can only prune if both bounds are long enough and their suffixes don't overlap with target
	if (lower_str.length() >= suffix_len && upper_str.length() >= suffix_len) {
		auto lower_suffix = lower_str.substr(lower_str.length() - suffix_len);
		auto upper_suffix = upper_str.substr(upper_str.length() - suffix_len);

		//! If both suffixes are greater than target, or both less than target, can prune
		if (lower_suffix > suffix_str && upper_suffix > suffix_str) {
			return false;
		}
		if (lower_suffix < suffix_str && upper_suffix < suffix_str) {
			return false;
		}
	}

	return true;
}

//! Helper: Match NOT IN expression
template <class TRANSFORM>
static bool MatchNotInExpression(ClientContext &context, const Expression &expr, const IcebergPredicateStats &stats,
                                 const IcebergTransform &transform) {
	D_ASSERT(expr.GetExpressionClass() == ExpressionClass::BOUND_OPERATOR);
	auto &operator_expr = expr.Cast<BoundOperatorExpression>();

	if (operator_expr.children.empty()) {
		return true;
	}

	//! First child should be column reference
	if (operator_expr.children[0]->type != ExpressionType::BOUND_REF) {
		return true;
	}

	//! Can only prune if file has single value (lower == upper)
	if (stats.lower_bound != stats.upper_bound) {
		return true;
	}

	//! Check if the single value is in the exclusion list
	for (size_t i = 1; i < operator_expr.children.size(); i++) {
		Value child_val;
		if (!ExpressionExecutor::TryEvaluateScalar(context, *operator_expr.children[i], child_val)) {
			return true;
		}
		if (stats.lower_bound == child_val) {
			return false; //! Can prune: only value in file is excluded
		}
	}
	return true;
}

//! Main expression-based matching function
template <class TRANSFORM>
static bool MatchBoundsFromExpressionTemplated(ClientContext &context, const Expression &expr,
                                               const IcebergPredicateStats &stats, const IcebergTransform &transform) {
	switch (expr.type) {
	//! === LOGICAL OPERATORS ===
	case ExpressionType::CONJUNCTION_OR: {
		auto &conj = expr.Cast<BoundConjunctionExpression>();
		for (auto &child : conj.children) {
			if (MatchBoundsFromExpressionTemplated<TRANSFORM>(context, *child, stats, transform)) {
				return true; //! Any child matches = file might have data
			}
		}
		return false; //! All children failed = can prune
	}
	case ExpressionType::CONJUNCTION_AND: {
		auto &conj = expr.Cast<BoundConjunctionExpression>();
		for (auto &child : conj.children) {
			if (!MatchBoundsFromExpressionTemplated<TRANSFORM>(context, *child, stats, transform)) {
				return false; //! Any child fails = can prune
			}
		}
		return true;
	}
	case ExpressionType::OPERATOR_NOT: {
		//! NOT expressions cannot be pruned with simple inversion.
		//! MatchBounds returns true="might match some rows" or false="definitely no rows match".
		//! For NOT expr:
		//!   - If child returns true (some rows might match expr), file might also have rows NOT matching expr
		//!   - If child returns false (no rows match expr), ALL rows match NOT expr
		//! In both cases we cannot prune, so always return true (conservative).
		return true;
	}

	//! === COMPARISON OPERATORS ===
	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_GREATERTHAN:
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
	case ExpressionType::COMPARE_EQUAL:
	case ExpressionType::COMPARE_NOTEQUAL:
		return MatchComparisonExpression<TRANSFORM>(context, expr, stats, transform);

	//! === RANGE OPERATORS ===
	case ExpressionType::COMPARE_BETWEEN:
		return MatchBetweenExpression<TRANSFORM>(context, expr, stats, transform);
	case ExpressionType::COMPARE_NOT_BETWEEN:
		return MatchNotBetweenExpression<TRANSFORM>(context, expr, stats, transform);

	//! === STRING FUNCTIONS ===
	case ExpressionType::BOUND_FUNCTION: {
		auto &func_expr = expr.Cast<BoundFunctionExpression>();
		if (func_expr.function.name == "prefix" || func_expr.function.name == "starts_with") {
			return MatchStartsWithExpression<TRANSFORM>(context, func_expr, stats, transform);
		}
		if (func_expr.function.name == "suffix" || func_expr.function.name == "ends_with") {
			return MatchEndsWithExpression<TRANSFORM>(context, func_expr, stats, transform);
		}
		return true;
	}

	//! === MEMBERSHIP OPERATORS ===
	case ExpressionType::COMPARE_NOT_IN:
		return MatchNotInExpression<TRANSFORM>(context, expr, stats, transform);

	//! === NULL OPERATORS ===
	case ExpressionType::OPERATOR_IS_NULL:
		return stats.has_null == true;
	case ExpressionType::OPERATOR_IS_NOT_NULL:
		return stats.has_not_null == true;

	default:
		return true; //! Conservative: unknown expression type
	}
}

bool IcebergPredicate::MatchBoundsFromExpression(ClientContext &context, const Expression &expr,
                                                 const IcebergPredicateStats &stats,
                                                 const IcebergTransform &transform) {
	switch (transform.Type()) {
	case IcebergTransformType::IDENTITY:
		return MatchBoundsFromExpressionTemplated<IdentityTransform>(context, expr, stats, transform);
	case IcebergTransformType::BUCKET:
		return true;
	case IcebergTransformType::TRUNCATE:
		return true;
	case IcebergTransformType::YEAR:
		return MatchBoundsFromExpressionTemplated<YearTransform>(context, expr, stats, transform);
	case IcebergTransformType::MONTH:
		return MatchBoundsFromExpressionTemplated<MonthTransform>(context, expr, stats, transform);
	case IcebergTransformType::DAY:
		return MatchBoundsFromExpressionTemplated<DayTransform>(context, expr, stats, transform);
	case IcebergTransformType::HOUR:
		return MatchBoundsFromExpressionTemplated<HourTransform>(context, expr, stats, transform);
	case IcebergTransformType::VOID:
		return true;
	default:
		throw InvalidConfigurationException("Transform '%s' not implemented", transform.RawType());
	}
}

bool IcebergPredicate::MatchBounds(ClientContext &context, const TableFilter &filter,
                                   const IcebergPredicateStats &stats, const IcebergTransform &transform) {
	switch (transform.Type()) {
	case IcebergTransformType::IDENTITY:
		return MatchBoundsTemplated<IdentityTransform>(context, filter, stats, transform);
	case IcebergTransformType::BUCKET:
		return true;
	case IcebergTransformType::TRUNCATE:
		return true;
	case IcebergTransformType::YEAR:
		return MatchBoundsTemplated<YearTransform>(context, filter, stats, transform);
	case IcebergTransformType::MONTH:
		return MatchBoundsTemplated<MonthTransform>(context, filter, stats, transform);
	case IcebergTransformType::DAY:
		return MatchBoundsTemplated<DayTransform>(context, filter, stats, transform);
	case IcebergTransformType::HOUR:
		return MatchBoundsTemplated<HourTransform>(context, filter, stats, transform);
	case IcebergTransformType::VOID:
		return true;
	default:
		throw InvalidConfigurationException("Transform '%s' not implemented", transform.RawType());
	}
}

} // namespace duckdb
