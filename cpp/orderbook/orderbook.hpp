// Limit order book. Step 4.2.
//
// Price-level book with O(1) order lookup by reference and O(log n) level
// access. Built for replay: add / cancel / execute / replace arrive in the
// order ITCH emitted them and the book is the state after each one.
//
// The interesting part for this project is not the book itself, it is that
// every resting order records how much volume sat ahead of it at its level
// when it arrived. That is what makes queue-position simulation possible, and
// queue position is the difference between an honest passive fill model and a
// bps assumption.
#pragma once

#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

#include "types.hpp"

namespace aqc {

// Aggregate state of one price level.
struct Level {
    Shares shares = 0;
    std::uint32_t order_count = 0;
    // Monotonic count of shares ever executed at this level. An order's
    // position in the queue is (its level_volume_ahead - this), floored at 0.
    std::uint64_t cumulative_executed = 0;
};

class OrderBook {
public:
    OrderBook() = default;

    // ---- ITCH-driven mutations ------------------------------------------
    void add(OrderRef ref, Side side, Price price, Shares shares);

    // Partial or full execution of a resting order. Returns shares actually
    // removed, which is less than requested if the book has drifted.
    Shares execute(OrderRef ref, Shares shares);

    // Cancel reduces; delete removes entirely.
    Shares cancel(OrderRef ref, Shares shares);
    Shares remove(OrderRef ref);

    // ITCH replace is delete + add with a NEW reference, and critically the
    // replaced order goes to the BACK of the queue. Modelling it as an
    // in-place edit is the classic mistake - it would hand every replaced
    // order an undeserved queue position.
    void replace(OrderRef old_ref, OrderRef new_ref, Price price, Shares shares);

    void clear();

    // ---- queries ---------------------------------------------------------
    Quote top(Timestamp ts = 0) const;
    Price best_bid() const;
    Price best_ask() const;
    Shares size_at(Side side, Price price) const;
    std::size_t level_count(Side side) const;
    std::size_t order_count() const { return orders_.size(); }

    // Depth for market-impact walking: levels from best outward.
    std::vector<std::pair<Price, Shares>> depth(Side side, std::size_t levels) const;

    // Total shares available between the touch and a limit price inclusive.
    std::uint64_t liquidity_to(Side side, Price limit) const;

    // Shares still ahead of `ref` in its level's queue. 0 means next to fill.
    std::uint64_t queue_ahead(OrderRef ref) const;

    bool has_order(OrderRef ref) const { return orders_.count(ref) > 0; }
    const Order* find(OrderRef ref) const;

private:
    // Bids descending, asks ascending, so begin() is always the touch.
    std::map<Price, Level, std::greater<Price>> bids_;
    std::map<Price, Level, std::less<Price>> asks_;
    std::unordered_map<OrderRef, Order> orders_;

    template <typename BookSide>
    void apply_add(BookSide& book, Order& order);

    template <typename BookSide>
    Shares apply_reduce(BookSide& book, Order& order, Shares shares, bool executed);
};

}  // namespace aqc
