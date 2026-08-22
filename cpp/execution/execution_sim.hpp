// Execution simulator. Step 4.4.
//
// Ties the book, the impact walk and the queue model together into the one
// thing Python asks for: what would this order actually have cost?
//
// Two modes, and the difference between them is the point:
//   aggressive - cross the spread, pay impact, certain fill
//   passive    - join the queue, maybe fill, maybe get picked off
#pragma once

#include <string>
#include <vector>

#include "market_impact.hpp"
#include "orderbook.hpp"
#include "queue_position.hpp"
#include "types.hpp"

namespace aqc {

struct ExecutionConfig {
    // Cap on any single child order as a fraction of displayed depth.
    double max_participation = 0.05;
    // Levels to walk before giving up on filling the rest.
    std::size_t max_levels = 50;
    // Half-spread credit assumed when a passive order fills unmolested.
    bool credit_spread_on_passive = true;
};

struct ExecutionResult {
    Shares requested = 0;
    Shares filled = 0;
    double avg_price = 0.0;
    double arrival_mid = 0.0;
    double slippage_bps = 0.0;
    double spread_bps = 0.0;
    std::size_t levels_consumed = 0;
    bool complete = false;
    std::string mode;

    double fill_ratio() const {
        return requested > 0 ? static_cast<double>(filled) / requested : 0.0;
    }
};

class ExecutionSimulator {
public:
    explicit ExecutionSimulator(ExecutionConfig cfg = {}) : cfg_(cfg) {}

    // Cross the spread now. Always fills if there is depth; the cost is real.
    ExecutionResult market_order(const OrderBook& book, Side side, Shares shares) const;

    // Post at the touch and see what happens. `volume_at_level` is how much
    // subsequently traded at our price - from the ITCH replay, not assumed.
    ExecutionResult limit_order(const OrderBook& book, Side side, Shares shares,
                                std::uint64_t volume_at_level,
                                bool price_moved_through = false) const;

    // Slice a parent order into child orders no larger than the participation
    // cap, walking the book fresh for each. Approximates a scheduled execution
    // without needing a full simulation clock.
    ExecutionResult sliced_order(const OrderBook& book, Side side, Shares shares,
                                 std::size_t slices) const;

    const ExecutionConfig& config() const { return cfg_; }

private:
    ExecutionConfig cfg_;
};

}  // namespace aqc
