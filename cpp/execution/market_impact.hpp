// Market impact by walking the real book. Step 4.4.
//
// The Python model guesses impact from a square-root participation formula
// with an assumed volatility. This does not guess: it consumes actual resting
// depth level by level and reports what the order would have paid.
//
// That difference is the whole argument for this layer. A square-root model
// says a large order in a thin book costs "more"; walking the book says it
// costs 47 bps and clears six levels, and that number can be checked.
#pragma once

#include <algorithm>
#include <cmath>

#include "orderbook.hpp"
#include "types.hpp"

namespace aqc {

// Immediate cost of taking `shares` from the book, right now.
inline Fill walk_book(const OrderBook& book, Side side, Shares shares, std::size_t max_levels = 50) {
    Fill fill;
    const Quote q = book.top();
    const double arrival_mid = q.mid();

    // Buying consumes asks, selling consumes bids.
    const auto levels = book.depth(opposite(side) == Side::Buy ? Side::Buy : Side::Sell, max_levels);

    Shares remaining = shares;
    double notional = 0.0;

    for (const auto& [price, available] : levels) {
        if (remaining == 0) break;
        const Shares take = std::min(remaining, available);
        notional += static_cast<double>(take) * to_double(price);
        remaining -= take;
        fill.filled += take;
    }

    fill.unfilled = remaining;
    fill.complete = (remaining == 0);
    fill.avg_price = fill.filled > 0 ? notional / fill.filled : 0.0;

    if (arrival_mid > 0.0 && fill.filled > 0) {
        // Signed so it is always a cost: buys pay above the mid, sells below.
        const double diff = (side == Side::Buy)
            ? fill.avg_price - arrival_mid
            : arrival_mid - fill.avg_price;
        fill.slippage_bps = (diff / arrival_mid) * 1e4;
    }
    return fill;
}

// Depth-weighted book imbalance in [-1, 1]. Positive means bid-heavy.
// A genuine microstructure signal and free once the book exists.
inline double book_imbalance(const OrderBook& book, std::size_t levels = 5) {
    double bid_vol = 0.0, ask_vol = 0.0;
    for (const auto& [p, s] : book.depth(Side::Buy, levels))  bid_vol += s;
    for (const auto& [p, s] : book.depth(Side::Sell, levels)) ask_vol += s;
    const double total = bid_vol + ask_vol;
    return total > 0.0 ? (bid_vol - ask_vol) / total : 0.0;
}

// How many shares can be taken without moving price more than `max_bps`.
// The honest answer to "how big can this strategy actually trade?".
inline Shares capacity_within(const OrderBook& book, Side side, double max_bps,
                              std::size_t max_levels = 50) {
    const Quote q = book.top();
    const double mid = q.mid();
    if (mid <= 0.0) return 0;

    const double limit = (side == Side::Buy)
        ? mid * (1.0 + max_bps / 1e4)
        : mid * (1.0 - max_bps / 1e4);

    return static_cast<Shares>(
        std::min<std::uint64_t>(
            book.liquidity_to(side, from_double(limit)),
            std::numeric_limits<Shares>::max()));
}

}  // namespace aqc
