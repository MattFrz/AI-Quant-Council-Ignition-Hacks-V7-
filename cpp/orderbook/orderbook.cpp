#include "orderbook.hpp"

#include <algorithm>

namespace aqc {

template <typename BookSide>
void OrderBook::apply_add(BookSide& book, Order& order) {
    Level& level = book[order.price];
    // Everything currently resting plus everything already executed here is
    // what this order must wait behind.
    order.level_volume_ahead = level.cumulative_executed + level.shares;
    level.shares += order.shares;
    level.order_count += 1;
}

template <typename BookSide>
Shares OrderBook::apply_reduce(BookSide& book, Order& order, Shares shares, bool executed) {
    auto it = book.find(order.price);
    if (it == book.end()) return 0;

    // Never remove more than is actually resting. Feed gaps and out-of-order
    // replay both produce oversized reductions, and letting them underflow an
    // unsigned counter corrupts every level below.
    const Shares removed = std::min(shares, order.shares);
    order.shares -= removed;

    Level& level = it->second;
    level.shares = (level.shares >= removed) ? level.shares - removed : 0;
    if (executed) {
        level.cumulative_executed += removed;
    }

    if (order.shares == 0) {
        level.order_count = level.order_count > 0 ? level.order_count - 1 : 0;
    }
    if (level.shares == 0 && level.order_count == 0) {
        book.erase(it);
    }
    return removed;
}

void OrderBook::add(OrderRef ref, Side side, Price price, Shares shares) {
    if (shares == 0 || price == PRICE_INVALID) return;

    Order order;
    order.ref = ref;
    order.side = side;
    order.price = price;
    order.shares = shares;

    if (side == Side::Buy) {
        apply_add(bids_, order);
    } else {
        apply_add(asks_, order);
    }
    orders_[ref] = order;
}

Shares OrderBook::execute(OrderRef ref, Shares shares) {
    auto it = orders_.find(ref);
    if (it == orders_.end()) return 0;

    Order& order = it->second;
    const Shares removed = (order.side == Side::Buy)
        ? apply_reduce(bids_, order, shares, true)
        : apply_reduce(asks_, order, shares, true);

    if (order.shares == 0) orders_.erase(it);
    return removed;
}

Shares OrderBook::cancel(OrderRef ref, Shares shares) {
    auto it = orders_.find(ref);
    if (it == orders_.end()) return 0;

    Order& order = it->second;
    const Shares removed = (order.side == Side::Buy)
        ? apply_reduce(bids_, order, shares, false)
        : apply_reduce(asks_, order, shares, false);

    if (order.shares == 0) orders_.erase(it);
    return removed;
}

Shares OrderBook::remove(OrderRef ref) {
    auto it = orders_.find(ref);
    if (it == orders_.end()) return 0;
    return cancel(ref, it->second.shares);
}

void OrderBook::replace(OrderRef old_ref, OrderRef new_ref, Price price, Shares shares) {
    auto it = orders_.find(old_ref);
    if (it == orders_.end()) return;

    const Side side = it->second.side;
    remove(old_ref);
    // New reference, back of the queue at the new price. Per ITCH semantics.
    add(new_ref, side, price, shares);
}

void OrderBook::clear() {
    bids_.clear();
    asks_.clear();
    orders_.clear();
}

Price OrderBook::best_bid() const {
    return bids_.empty() ? PRICE_INVALID : bids_.begin()->first;
}

Price OrderBook::best_ask() const {
    return asks_.empty() ? PRICE_INVALID : asks_.begin()->first;
}

Quote OrderBook::top(Timestamp ts) const {
    Quote q;
    q.timestamp = ts;
    if (!bids_.empty()) {
        q.bid = bids_.begin()->first;
        q.bid_size = bids_.begin()->second.shares;
    }
    if (!asks_.empty()) {
        q.ask = asks_.begin()->first;
        q.ask_size = asks_.begin()->second.shares;
    }
    return q;
}

Shares OrderBook::size_at(Side side, Price price) const {
    if (side == Side::Buy) {
        auto it = bids_.find(price);
        return it == bids_.end() ? 0 : it->second.shares;
    }
    auto it = asks_.find(price);
    return it == asks_.end() ? 0 : it->second.shares;
}

std::size_t OrderBook::level_count(Side side) const {
    return side == Side::Buy ? bids_.size() : asks_.size();
}

std::vector<std::pair<Price, Shares>> OrderBook::depth(Side side, std::size_t levels) const {
    std::vector<std::pair<Price, Shares>> out;
    out.reserve(levels);
    if (side == Side::Buy) {
        for (const auto& [price, level] : bids_) {
            if (out.size() >= levels) break;
            out.emplace_back(price, level.shares);
        }
    } else {
        for (const auto& [price, level] : asks_) {
            if (out.size() >= levels) break;
            out.emplace_back(price, level.shares);
        }
    }
    return out;
}

std::uint64_t OrderBook::liquidity_to(Side side, Price limit) const {
    std::uint64_t total = 0;
    if (side == Side::Buy) {
        // Buying lifts asks: every ask at or below the limit is reachable.
        for (const auto& [price, level] : asks_) {
            if (price > limit) break;
            total += level.shares;
        }
    } else {
        for (const auto& [price, level] : bids_) {
            if (price < limit) break;
            total += level.shares;
        }
    }
    return total;
}

std::uint64_t OrderBook::queue_ahead(OrderRef ref) const {
    auto it = orders_.find(ref);
    if (it == orders_.end()) return 0;

    const Order& order = it->second;
    const auto& book_side = order.side;

    std::uint64_t executed = 0;
    if (book_side == Side::Buy) {
        auto lit = bids_.find(order.price);
        if (lit != bids_.end()) executed = lit->second.cumulative_executed;
    } else {
        auto lit = asks_.find(order.price);
        if (lit != asks_.end()) executed = lit->second.cumulative_executed;
    }

    // Everything that was ahead, minus everything that has since traded.
    return order.level_volume_ahead > executed
        ? order.level_volume_ahead - executed
        : 0;
}

const Order* OrderBook::find(OrderRef ref) const {
    auto it = orders_.find(ref);
    return it == orders_.end() ? nullptr : &it->second;
}

}  // namespace aqc
