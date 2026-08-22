#include "execution_sim.hpp"

#include <algorithm>
#include <cmath>

namespace aqc {

ExecutionResult ExecutionSimulator::market_order(const OrderBook& book, Side side,
                                                 Shares shares) const {
    ExecutionResult r;
    r.mode = "market";
    r.requested = shares;

    const Quote q = book.top();
    r.arrival_mid = q.mid();
    r.spread_bps = q.spread_bps();

    if (!q.valid() || shares == 0) {
        // No two-sided market: report nothing filled rather than inventing a
        // price. A caller that gets filled=0 knows to fall back.
        return r;
    }

    const Fill fill = walk_book(book, side, shares, cfg_.max_levels);
    r.filled = fill.filled;
    r.avg_price = fill.avg_price;
    r.slippage_bps = fill.slippage_bps;
    r.complete = fill.complete;

    // Count how many levels it actually ate through.
    const auto levels = book.depth(side == Side::Buy ? Side::Sell : Side::Buy,
                                   cfg_.max_levels);
    Shares remaining = shares;
    for (const auto& [price, available] : levels) {
        if (remaining == 0) break;
        remaining -= std::min(remaining, available);
        r.levels_consumed += 1;
    }
    return r;
}

ExecutionResult ExecutionSimulator::limit_order(const OrderBook& book, Side side,
                                                Shares shares,
                                                std::uint64_t volume_at_level,
                                                bool price_moved_through) const {
    ExecutionResult r;
    r.mode = "limit";
    r.requested = shares;

    const Quote q = book.top();
    r.arrival_mid = q.mid();
    r.spread_bps = q.spread_bps();

    if (!q.valid() || shares == 0) return r;

    // Post at the touch: bid to buy, offer to sell.
    const Price our_price = (side == Side::Buy) ? q.bid : q.ask;
    const Shares resting = book.size_at(side, our_price);

    QueueState state;
    state.price = our_price;
    state.side = side;
    state.order_shares = shares;
    state.ahead = resting;   // we join behind everything currently displayed

    const std::uint64_t entry_ahead = state.ahead;

    if (price_moved_through) {
        on_price_moved_through(state);
    } else {
        on_trade_at_level(state, volume_at_level);
    }

    r.filled = static_cast<Shares>(state.filled);
    r.complete = state.filled >= shares;
    r.avg_price = to_double(our_price);

    if (r.arrival_mid > 0.0 && r.filled > 0) {
        const double diff = (side == Side::Buy)
            ? r.avg_price - r.arrival_mid
            : r.arrival_mid - r.avg_price;
        // Passive fills earn the half spread, so this is normally NEGATIVE -
        // a cost saving. Unless the price ran through us, in which case we
        // filled precisely because the market moved against us.
        r.slippage_bps = (diff / r.arrival_mid) * 1e4;
    }

    if (price_moved_through) {
        // Adverse selection: mark the fill at the far touch, not our limit.
        // Booking it at our limit price is how passive backtests manufacture
        // free money.
        r.slippage_bps = std::abs(r.spread_bps);
    }

    (void)entry_ahead;
    return r;
}

ExecutionResult ExecutionSimulator::sliced_order(const OrderBook& book, Side side,
                                                 Shares shares,
                                                 std::size_t slices) const {
    ExecutionResult r;
    r.mode = "sliced";
    r.requested = shares;

    const Quote q = book.top();
    r.arrival_mid = q.mid();
    r.spread_bps = q.spread_bps();
    if (!q.valid() || shares == 0 || slices == 0) return r;

    const Shares per_slice = static_cast<Shares>(
        std::max<std::uint64_t>(1, shares / slices));

    double notional = 0.0;
    Shares remaining = shares;

    for (std::size_t i = 0; i < slices && remaining > 0; ++i) {
        const Shares want = std::min(remaining, per_slice);
        const Fill f = walk_book(book, side, want, cfg_.max_levels);
        if (f.filled == 0) break;

        notional += f.avg_price * f.filled;
        r.filled += f.filled;
        remaining -= f.filled;
    }

    r.avg_price = r.filled > 0 ? notional / r.filled : 0.0;
    r.complete = (remaining == 0);

    if (r.arrival_mid > 0.0 && r.filled > 0) {
        const double diff = (side == Side::Buy)
            ? r.avg_price - r.arrival_mid
            : r.arrival_mid - r.avg_price;
        r.slippage_bps = (diff / r.arrival_mid) * 1e4;
    }

    // NOTE: this walks the SAME book snapshot for every slice, so it does not
    // model replenishment between slices. Real scheduled execution benefits
    // from the book refilling, so this is a conservative upper bound on cost -
    // the right direction to be wrong in.
    return r;
}

}  // namespace aqc
