// Passive fill modelling via queue position. Step 4.4.
//
// This is the part a bps assumption cannot express. Post a limit order and you
// do not simply "get the spread" - you join the back of a queue and fill only
// once everything ahead of you trades. In a deep name that queue can be
// thousands of shares and most of the time you never fill at all; you get
// picked off when the price moves against you instead.
//
// Backtests that assume passive orders fill are the single most common way a
// market-making or limit-order strategy looks profitable and is not.
#pragma once

#include <algorithm>
#include <cstdint>

#include "types.hpp"

namespace aqc {

struct QueueState {
    Price price = PRICE_INVALID;
    Side side = Side::Buy;
    Shares order_shares = 0;
    std::uint64_t ahead = 0;       // shares in front of us
    std::uint64_t filled = 0;      // ours filled so far
    bool cancelled = false;

    bool done() const { return filled >= order_shares || cancelled; }
    std::uint64_t remaining() const {
        return order_shares > filled ? order_shares - filled : 0;
    }

    double fill_ratio() const {
        return order_shares > 0 ? static_cast<double>(filled) / order_shares : 0.0;
    }
};

// Apply `traded` shares executing at our price level.
//
// Strict FIFO: volume first consumes the queue ahead of us, and only the
// remainder touches our order. Splitting the trade pro-rata across the level
// would flatter every passive strategy, because it hands us fills we were
// never entitled to.
inline void on_trade_at_level(QueueState& q, std::uint64_t traded) {
    if (q.done()) return;

    if (traded <= q.ahead) {
        q.ahead -= traded;
        return;
    }

    const std::uint64_t spill = traded - q.ahead;
    q.ahead = 0;
    q.filled += std::min(spill, q.remaining());
}

// Cancellations ahead of us in the queue.
//
// Only cancels from orders BEHIND the front can be attributed loosely; without
// per-order visibility the honest approximation is that a cancel removes
// someone ahead with probability proportional to how much of the level is
// ahead of us. Being conservative here means assuming cancels do NOT help us,
// which is the direction that avoids inventing fills.
inline void on_cancel_at_level(QueueState& q, std::uint64_t cancelled,
                               bool assume_ahead = false) {
    if (q.done() || !assume_ahead) return;
    q.ahead = q.ahead > cancelled ? q.ahead - cancelled : 0;
}

// Adverse selection: the price left our level. A resting order that survives
// a move against it has been picked off - it fills exactly when it should not.
inline void on_price_moved_through(QueueState& q) {
    if (q.done()) return;
    // Everything ahead is gone because the level cleared, so we fill the rest,
    // at our limit price, into a market that has already moved past us.
    q.ahead = 0;
    q.filled = q.order_shares;
}

// Probability-free summary of how a passive order fared.
struct PassiveResult {
    std::uint64_t filled = 0;
    std::uint64_t unfilled = 0;
    double fill_ratio = 0.0;
    std::uint64_t queue_ahead_at_entry = 0;
    std::uint64_t volume_required = 0;   // volume that had to trade for a full fill
    bool adversely_selected = false;
};

inline PassiveResult summarise(const QueueState& q, std::uint64_t entry_ahead,
                               bool picked_off) {
    PassiveResult r;
    r.filled = q.filled;
    r.unfilled = q.remaining();
    r.fill_ratio = q.fill_ratio();
    r.queue_ahead_at_entry = entry_ahead;
    r.volume_required = entry_ahead + q.order_shares;
    r.adversely_selected = picked_off;
    return r;
}

}  // namespace aqc
