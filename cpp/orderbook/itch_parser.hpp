// Nasdaq TotalView-ITCH 5.0 parser. Step 4.3.
//
// Feeds an OrderBook directly from the binary feed. Two things make this
// fiddly and both are handled here:
//
//   1. ITCH is BIG-ENDIAN and the wire format is packed with no padding, so
//      every multi-byte field needs an explicit byte-swapped read. Casting a
//      struct over the buffer works on paper and produces garbage in practice.
//   2. Timestamps are 6 bytes, not 8. There is no native type for that.
//
// Messages arrive length-prefixed (2-byte big-endian length) when read from a
// .gz/.bin dump, so the reader handles both framed and raw streams.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "orderbook.hpp"
#include "types.hpp"

namespace aqc {

enum class ItchType : char {
    SystemEvent      = 'S',
    StockDirectory   = 'R',
    TradingAction    = 'H',
    AddOrder         = 'A',
    AddOrderMPID     = 'F',
    OrderExecuted    = 'E',
    OrderExecutedPrice = 'C',
    OrderCancel      = 'X',
    OrderDelete      = 'D',
    OrderReplace     = 'U',
    Trade            = 'P',
    CrossTrade       = 'Q',
    Unknown          = '?',
};

struct ItchStats {
    std::uint64_t messages = 0;
    std::uint64_t adds = 0;
    std::uint64_t executes = 0;
    std::uint64_t cancels = 0;
    std::uint64_t deletes = 0;
    std::uint64_t replaces = 0;
    std::uint64_t trades = 0;
    std::uint64_t skipped = 0;
    Timestamp first_ts = 0;
    Timestamp last_ts = 0;
};

// ---- big-endian readers -------------------------------------------------
// ITCH is network byte order. These are the only correct way to pull fields
// off the wire; a reinterpret_cast of a packed struct is not portable and is
// wrong on every little-endian machine, which is all of them.

inline std::uint16_t read_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) << 8 | p[1];
}

inline std::uint32_t read_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) << 24 |
           static_cast<std::uint32_t>(p[1]) << 16 |
           static_cast<std::uint32_t>(p[2]) << 8  |
           static_cast<std::uint32_t>(p[3]);
}

inline std::uint64_t read_u64(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
    return v;
}

// 6-byte timestamp: nanoseconds since midnight. No native type fits.
inline Timestamp read_ts48(const std::uint8_t* p) {
    Timestamp v = 0;
    for (int i = 0; i < 6; ++i) v = (v << 8) | p[i];
    return v;
}

inline std::string read_stock(const std::uint8_t* p) {
    // 8 bytes, space padded.
    std::string s(reinterpret_cast<const char*>(p), 8);
    const auto end = s.find_last_not_of(' ');
    return end == std::string::npos ? std::string() : s.substr(0, end + 1);
}

// Expected wire length per message type, excluding any length prefix.
std::size_t message_length(char type);

class ItchParser {
public:
    // `symbol` empty means process every stock. Filtering here rather than
    // downstream matters: a full ITCH day is tens of millions of messages and
    // building books for all 8,000 symbols to use one is wasteful.
    explicit ItchParser(OrderBook& book, std::string symbol = "");

    // Feed one raw message (no length prefix). Returns false if unrecognised.
    bool process(const std::uint8_t* data, std::size_t len);

    // Parse a whole buffer of length-prefixed messages.
    std::size_t process_buffer(const std::uint8_t* data, std::size_t len);

    // Parse a .bin / .itch file of length-prefixed messages.
    std::size_t process_file(const std::string& path, std::size_t max_messages = 0);

    const ItchStats& stats() const { return stats_; }
    Timestamp clock() const { return clock_; }
    const std::string& symbol() const { return symbol_; }

private:
    OrderBook& book_;
    std::string symbol_;
    ItchStats stats_;
    Timestamp clock_ = 0;

    // Order references seen for our symbol. ITCH execute/cancel/delete
    // messages carry no stock field, so the only way to know whether a
    // reference belongs to our symbol is to remember it from the add.
    // References are sparse 64-bit values, so this is a set, not a bitmap.
    std::unordered_set<OrderRef> tracked_;
    bool is_tracked(OrderRef ref) const {
        return symbol_.empty() || tracked_.count(ref) > 0;
    }
};

}  // namespace aqc
