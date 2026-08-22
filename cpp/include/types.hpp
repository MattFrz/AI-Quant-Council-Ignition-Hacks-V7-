// Shared POD types crossing the Python/C++ boundary. Step 4.1.
//
// Deliberately plain structs: no virtuals, no std::string in hot paths, no
// allocation during message processing. The whole point of this layer is that
// replaying millions of ITCH messages stays cheap, and that goes away the
// moment the message path allocates.
#pragma once

#include <cstdint>
#include <limits>

namespace aqc {

// Nasdaq ITCH quotes prices as integers scaled by 10,000 (4 implied decimals).
// Keeping them integral avoids float comparison in the book, where an
// off-by-one-ulp price would silently split a price level in two.
using Price = std::int64_t;
using Shares = std::uint32_t;
using OrderRef = std::uint64_t;
using Timestamp = std::uint64_t;  // nanoseconds since midnight

inline constexpr int PRICE_SCALE = 10000;
inline constexpr Price PRICE_INVALID = std::numeric_limits<Price>::min();

inline double to_double(Price p) {
    return static_cast<double>(p) / PRICE_SCALE;
}

inline Price from_double(double p) {
    return static_cast<Price>(p * PRICE_SCALE + (p >= 0 ? 0.5 : -0.5));
}

enum class Side : std::uint8_t { Buy = 0, Sell = 1 };

inline Side opposite(Side s) {
    return s == Side::Buy ? Side::Sell : Side::Buy;
}

// One resting order in the book.
struct Order {
    OrderRef ref = 0;
    Price price = PRICE_INVALID;
    Shares shares = 0;
    Side side = Side::Buy;
    // Cumulative shares executed at this price level before this order was
    // added. Queue position is derived from this rather than by walking a
    // list, so it stays O(1) as the level grows.
    std::uint64_t level_volume_ahead = 0;
};

// Top of book snapshot.
struct Quote {
    Price bid = PRICE_INVALID;
    Price ask = PRICE_INVALID;
    Shares bid_size = 0;
    Shares ask_size = 0;
    Timestamp timestamp = 0;

    bool valid() const {
        return bid != PRICE_INVALID && ask != PRICE_INVALID && ask > bid;
    }

    double mid() const {
        return valid() ? (to_double(bid) + to_double(ask)) * 0.5 : 0.0;
    }

    double spread() const {
        return valid() ? to_double(ask) - to_double(bid) : 0.0;
    }

    // Spread in basis points of the mid. The number the Python cost model
    // currently guesses from an ADV bucket.
    double spread_bps() const {
        const double m = mid();
        return m > 0.0 ? (spread() / m) * 1e4 : 0.0;
    }
};

// Result of simulating one order against the reconstructed book.
struct Fill {
    Shares filled = 0;
    Shares unfilled = 0;
    double avg_price = 0.0;
    double slippage_bps = 0.0;   // vs the arrival mid, signed against you
    double queue_wait_shares = 0.0;  // shares that traded ahead of us
    bool complete = false;
};

}  // namespace aqc
