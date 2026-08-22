#include "itch_parser.hpp"

#include <fstream>
#include <vector>

namespace aqc {

std::size_t message_length(char type) {
    // TotalView-ITCH 5.0 fixed message sizes, excluding the 2-byte frame.
    switch (type) {
        case 'S': return 12;
        case 'R': return 39;
        case 'H': return 25;
        case 'Y': return 20;
        case 'L': return 26;
        case 'V': return 35;
        case 'W': return 12;
        case 'K': return 28;
        case 'J': return 35;
        case 'h': return 21;
        case 'A': return 36;
        case 'F': return 40;
        case 'E': return 31;
        case 'C': return 36;
        case 'X': return 23;
        case 'D': return 19;
        case 'U': return 35;
        case 'P': return 44;
        case 'Q': return 40;
        case 'B': return 19;
        case 'I': return 50;
        case 'N': return 20;
        default:  return 0;
    }
}

ItchParser::ItchParser(OrderBook& book, std::string symbol)
    : book_(book), symbol_(std::move(symbol)) {}

bool ItchParser::process(const std::uint8_t* data, std::size_t len) {
    if (len < 11) return false;

    const char type = static_cast<char>(data[0]);

    // Layout common to every order message:
    //   [0]    type
    //   [1-2]  stock locate
    //   [3-4]  tracking number
    //   [5-10] timestamp (48 bit)
    clock_ = read_ts48(data + 5);
    if (stats_.first_ts == 0) stats_.first_ts = clock_;
    stats_.last_ts = clock_;
    stats_.messages += 1;

    switch (type) {
        case 'A':
        case 'F': {
            // ref(8) side(1) shares(4) stock(8) price(4)
            if (len < 36) return false;
            const OrderRef ref = read_u64(data + 11);
            const Side side = (data[19] == 'B') ? Side::Buy : Side::Sell;
            const Shares shares = read_u32(data + 20);
            const std::string stock = read_stock(data + 24);
            const Price price = static_cast<Price>(read_u32(data + 32));

            if (!symbol_.empty() && stock != symbol_) {
                stats_.skipped += 1;
                return true;
            }
            if (!symbol_.empty()) tracked_.insert(ref);

            book_.add(ref, side, price, shares);
            stats_.adds += 1;
            return true;
        }

        case 'E': {
            // ref(8) executed(4) match(8)
            if (len < 31) return false;
            const OrderRef ref = read_u64(data + 11);
            if (!is_tracked(ref)) { stats_.skipped += 1; return true; }
            book_.execute(ref, read_u32(data + 19));
            stats_.executes += 1;
            return true;
        }

        case 'C': {
            // ref(8) executed(4) match(8) printable(1) price(4)
            // Executed at a price other than the display price. Book effect is
            // identical to 'E'; the price only matters for trade prints.
            if (len < 36) return false;
            const OrderRef ref = read_u64(data + 11);
            if (!is_tracked(ref)) { stats_.skipped += 1; return true; }
            book_.execute(ref, read_u32(data + 19));
            stats_.executes += 1;
            return true;
        }

        case 'X': {
            // ref(8) cancelled(4)
            if (len < 23) return false;
            const OrderRef ref = read_u64(data + 11);
            if (!is_tracked(ref)) { stats_.skipped += 1; return true; }
            book_.cancel(ref, read_u32(data + 19));
            stats_.cancels += 1;
            return true;
        }

        case 'D': {
            // ref(8)
            if (len < 19) return false;
            const OrderRef ref = read_u64(data + 11);
            if (!is_tracked(ref)) { stats_.skipped += 1; return true; }
            book_.remove(ref);
            tracked_.erase(ref);
            stats_.deletes += 1;
            return true;
        }

        case 'U': {
            // old_ref(8) new_ref(8) shares(4) price(4)
            if (len < 35) return false;
            const OrderRef old_ref = read_u64(data + 11);
            const OrderRef new_ref = read_u64(data + 19);
            if (!is_tracked(old_ref)) { stats_.skipped += 1; return true; }

            const Shares shares = read_u32(data + 27);
            const Price price = static_cast<Price>(read_u32(data + 31));

            book_.replace(old_ref, new_ref, price, shares);
            if (!symbol_.empty()) {
                tracked_.erase(old_ref);
                tracked_.insert(new_ref);
            }
            stats_.replaces += 1;
            return true;
        }

        case 'P': {
            // Non-displayable trade. Does not touch the book - it never rested
            // there. Counted so volume statistics stay honest.
            stats_.trades += 1;
            return true;
        }

        default:
            stats_.skipped += 1;
            return true;
    }
}

std::size_t ItchParser::process_buffer(const std::uint8_t* data, std::size_t len) {
    std::size_t offset = 0;
    std::size_t count = 0;

    while (offset + 2 <= len) {
        const std::uint16_t msg_len = read_u16(data + offset);
        offset += 2;
        if (msg_len == 0 || offset + msg_len > len) break;
        process(data + offset, msg_len);
        offset += msg_len;
        ++count;
    }
    return count;
}

std::size_t ItchParser::process_file(const std::string& path, std::size_t max_messages) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return 0;

    std::size_t count = 0;
    std::vector<std::uint8_t> buf(128);

    while (in) {
        std::uint8_t frame[2];
        in.read(reinterpret_cast<char*>(frame), 2);
        if (in.gcount() != 2) break;

        const std::uint16_t msg_len = read_u16(frame);
        if (msg_len == 0) break;
        if (buf.size() < msg_len) buf.resize(msg_len);

        in.read(reinterpret_cast<char*>(buf.data()), msg_len);
        if (in.gcount() != static_cast<std::streamsize>(msg_len)) break;

        process(buf.data(), msg_len);
        ++count;
        if (max_messages && count >= max_messages) break;
    }
    return count;
}

}  // namespace aqc
