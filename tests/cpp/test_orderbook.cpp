// Order book and execution assertions. No test framework - one binary, exits
// non-zero on failure, so it runs anywhere without pulling in gtest.
//
//   cmake -S cpp -B cpp/build && cmake --build cpp/build && ./cpp/build/test_orderbook
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "execution_sim.hpp"
#include "itch_parser.hpp"
#include "market_impact.hpp"
#include "orderbook.hpp"

using namespace aqc;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        ++checks;                                                           \
        if (!(cond)) {                                                      \
            std::printf("  FAIL  %s\n        at %s:%d\n", msg, __FILE__, __LINE__); \
            ++failures;                                                     \
        }                                                                   \
    } while (0)

#define CLOSE(a, b, tol, msg) CHECK(std::fabs((a) - (b)) < (tol), msg)

static void test_price_conversion() {
    std::printf("price conversion\n");
    CHECK(from_double(100.25) == 1002500, "100.25 scales to 1002500");
    CLOSE(to_double(1002500), 100.25, 1e-9, "round trips");
    CHECK(from_double(0.0001) == 1, "sub-cent tick survives");
}

static void test_add_and_top() {
    std::printf("add and top of book\n");
    OrderBook b;
    b.add(1, Side::Buy, from_double(100.00), 500);
    b.add(2, Side::Sell, from_double(100.05), 300);

    const Quote q = b.top();
    CHECK(q.valid(), "two-sided book is valid");
    CLOSE(to_double(q.bid), 100.00, 1e-9, "best bid");
    CLOSE(to_double(q.ask), 100.05, 1e-9, "best ask");
    CHECK(q.bid_size == 500, "bid size");
    CLOSE(q.mid(), 100.025, 1e-9, "mid");
    CLOSE(q.spread(), 0.05, 1e-9, "spread");
}

static void test_price_priority() {
    std::printf("price priority\n");
    OrderBook b;
    b.add(1, Side::Buy, from_double(99.00), 100);
    b.add(2, Side::Buy, from_double(100.00), 100);
    b.add(3, Side::Buy, from_double(98.00), 100);
    CLOSE(to_double(b.best_bid()), 100.00, 1e-9, "highest bid is best");

    b.add(4, Side::Sell, from_double(101.00), 100);
    b.add(5, Side::Sell, from_double(100.50), 100);
    CLOSE(to_double(b.best_ask()), 100.50, 1e-9, "lowest ask is best");
}

static void test_execute_and_cancel() {
    std::printf("execute and cancel\n");
    OrderBook b;
    b.add(1, Side::Buy, from_double(100.00), 500);

    CHECK(b.execute(1, 200) == 200, "partial execute returns removed");
    CHECK(b.size_at(Side::Buy, from_double(100.00)) == 300, "level reduced");

    CHECK(b.cancel(1, 100) == 100, "cancel reduces");
    CHECK(b.size_at(Side::Buy, from_double(100.00)) == 200, "level reduced again");

    b.remove(1);
    CHECK(b.level_count(Side::Buy) == 0, "empty level is erased");
    CHECK(b.order_count() == 0, "order is gone");
}

static void test_oversized_reduction_does_not_underflow() {
    std::printf("oversized reduction\n");
    // Feed gaps produce reductions bigger than the resting size. Unsigned
    // underflow here would corrupt every level below.
    OrderBook b;
    b.add(1, Side::Buy, from_double(100.00), 100);
    CHECK(b.execute(1, 999999) == 100, "clamps to available");
    CHECK(b.level_count(Side::Buy) == 0, "level cleanly removed");
}

static void test_replace_goes_to_back_of_queue() {
    std::printf("replace loses queue position\n");
    OrderBook b;
    b.add(1, Side::Buy, from_double(100.00), 100);   // ahead
    b.add(2, Side::Buy, from_double(100.00), 100);   // ours

    CHECK(b.queue_ahead(2) == 100, "100 shares ahead at entry");

    // Replacing our order must send it behind order 1 again, not preserve
    // position. Treating replace as an in-place edit is the classic bug.
    b.replace(2, 3, from_double(100.00), 150);
    CHECK(!b.has_order(2), "old ref gone");
    CHECK(b.has_order(3), "new ref present");
    CHECK(b.queue_ahead(3) == 100, "still behind order 1");
}

static void test_queue_ahead_decreases_with_executions() {
    std::printf("queue position\n");
    OrderBook b;
    b.add(1, Side::Buy, from_double(100.00), 300);   // ahead of us
    b.add(2, Side::Buy, from_double(100.00), 100);   // ours

    CHECK(b.queue_ahead(2) == 300, "starts behind 300");
    b.execute(1, 100);
    CHECK(b.queue_ahead(2) == 200, "drops as the queue trades");
    b.execute(1, 200);
    CHECK(b.queue_ahead(2) == 0, "front of queue");
}

static void test_market_order_walks_levels() {
    std::printf("market order impact\n");
    OrderBook b;
    b.add(1, Side::Buy,  from_double(99.99), 1000);
    b.add(2, Side::Sell, from_double(100.00), 100);
    b.add(3, Side::Sell, from_double(100.05), 100);
    b.add(4, Side::Sell, from_double(100.10), 100);

    ExecutionSimulator sim;

    const auto small = sim.market_order(b, Side::Buy, 50);
    CHECK(small.complete, "small order fills at the touch");
    CLOSE(small.avg_price, 100.00, 1e-9, "no impact beyond level one");

    const auto large = sim.market_order(b, Side::Buy, 250);
    CHECK(large.complete, "large order fills across levels");
    CHECK(large.avg_price > small.avg_price, "bigger order pays more");
    CHECK(large.slippage_bps > small.slippage_bps, "impact grows with size");
    CHECK(large.levels_consumed >= 3, "consumed at least three levels");
}

static void test_market_order_beyond_depth_is_incomplete() {
    std::printf("insufficient depth\n");
    OrderBook b;
    b.add(1, Side::Buy,  from_double(99.99), 100);
    b.add(2, Side::Sell, from_double(100.00), 100);

    ExecutionSimulator sim;
    const auto r = sim.market_order(b, Side::Buy, 5000);
    CHECK(!r.complete, "cannot fill more than exists");
    CHECK(r.filled == 100, "fills only what is displayed");
}

static void test_slippage_always_costs() {
    std::printf("slippage direction\n");
    OrderBook b;
    b.add(1, Side::Buy,  from_double(99.90), 500);
    b.add(2, Side::Sell, from_double(100.10), 500);

    ExecutionSimulator sim;
    const auto buy  = sim.market_order(b, Side::Buy, 100);
    const auto sell = sim.market_order(b, Side::Sell, 100);

    CHECK(buy.slippage_bps > 0, "buying costs");
    CHECK(sell.slippage_bps > 0, "selling costs too");
    CHECK(buy.avg_price > buy.arrival_mid, "buy fills above mid");
    CHECK(sell.avg_price < sell.arrival_mid, "sell fills below mid");
}

static void test_passive_fill_requires_queue_to_trade() {
    std::printf("passive fills\n");
    OrderBook b;
    b.add(1, Side::Buy,  from_double(100.00), 1000);  // queue ahead of us
    b.add(2, Side::Sell, from_double(100.05), 500);

    ExecutionSimulator sim;

    // 500 shares trade: not enough to clear the 1000 ahead of us.
    const auto starved = sim.limit_order(b, Side::Buy, 200, 500);
    CHECK(starved.filled == 0, "no fill while queue is ahead");

    // 1200 trade: clears the queue, 200 spills to us.
    const auto filled = sim.limit_order(b, Side::Buy, 200, 1200);
    CHECK(filled.filled == 200, "fills once the queue clears");
    CHECK(filled.complete, "fully filled");
}

static void test_adverse_selection_is_charged() {
    std::printf("adverse selection\n");
    OrderBook b;
    b.add(1, Side::Buy,  from_double(100.00), 100);
    b.add(2, Side::Sell, from_double(100.10), 100);

    ExecutionSimulator sim;
    const auto picked_off = sim.limit_order(b, Side::Buy, 100, 0, true);

    CHECK(picked_off.filled == 100, "price ran through us, so we filled");
    CHECK(picked_off.slippage_bps > 0,
          "getting picked off is a COST, not a spread credit");
}

static void test_book_imbalance() {
    std::printf("book imbalance\n");
    OrderBook b;
    b.add(1, Side::Buy,  from_double(100.00), 900);
    b.add(2, Side::Sell, from_double(100.05), 100);
    CHECK(book_imbalance(b, 5) > 0.5, "bid-heavy book reads positive");
}

static void test_itch_endianness() {
    std::printf("ITCH big-endian readers\n");
    const std::uint8_t u32[] = {0x00, 0x00, 0x01, 0x00};
    CHECK(read_u32(u32) == 256, "u32 is big-endian");

    const std::uint8_t u16[] = {0x01, 0x00};
    CHECK(read_u16(u16) == 256, "u16 is big-endian");

    const std::uint8_t ts[] = {0x00, 0x00, 0x00, 0x00, 0x01, 0x00};
    CHECK(read_ts48(ts) == 256, "48-bit timestamp is big-endian");

    const std::uint8_t stock[] = {'A','A','P','L',' ',' ',' ',' '};
    CHECK(read_stock(stock) == "AAPL", "stock symbol is space-trimmed");
}

static void test_itch_add_message() {
    std::printf("ITCH add message\n");
    // Hand-build an 'A' message: type, locate, tracking, ts48, ref, side,
    // shares, stock, price.
    std::uint8_t msg[36] = {0};
    msg[0] = 'A';
    for (int i = 0; i < 6; ++i) msg[5 + i] = 0;
    msg[10] = 100;                       // timestamp low byte
    for (int i = 0; i < 7; ++i) msg[11 + i] = 0;
    msg[18] = 42;                        // order ref = 42
    msg[19] = 'B';                       // buy
    msg[20] = 0; msg[21] = 0; msg[22] = 0x01; msg[23] = 0x2C;  // 300 shares
    const char* sym = "AAPL    ";
    for (int i = 0; i < 8; ++i) msg[24 + i] = static_cast<std::uint8_t>(sym[i]);
    // price 100.0000 -> 1000000 -> 0x000F4240
    msg[32] = 0x00; msg[33] = 0x0F; msg[34] = 0x42; msg[35] = 0x40;

    OrderBook b;
    ItchParser p(b, "AAPL");
    CHECK(p.process(msg, 36), "message accepted");
    CHECK(p.stats().adds == 1, "counted as an add");
    CHECK(b.order_count() == 1, "order is in the book");
    CLOSE(to_double(b.best_bid()), 100.00, 1e-9, "price decoded correctly");
    CHECK(b.size_at(Side::Buy, from_double(100.00)) == 300, "shares decoded");
}

static void test_itch_symbol_filter() {
    std::printf("ITCH symbol filter\n");
    std::uint8_t msg[36] = {0};
    msg[0] = 'A';
    msg[18] = 7;
    msg[19] = 'B';
    msg[23] = 100;
    const char* sym = "MSFT    ";
    for (int i = 0; i < 8; ++i) msg[24 + i] = static_cast<std::uint8_t>(sym[i]);
    msg[35] = 100;

    OrderBook b;
    ItchParser p(b, "AAPL");
    p.process(msg, 36);
    CHECK(b.order_count() == 0, "other symbols are skipped");
    CHECK(p.stats().skipped == 1, "and counted as skipped");
}

int main() {
    std::printf("\n=== order book / execution tests ===\n\n");

    test_price_conversion();
    test_add_and_top();
    test_price_priority();
    test_execute_and_cancel();
    test_oversized_reduction_does_not_underflow();
    test_replace_goes_to_back_of_queue();
    test_queue_ahead_decreases_with_executions();
    test_market_order_walks_levels();
    test_market_order_beyond_depth_is_incomplete();
    test_slippage_always_costs();
    test_passive_fill_requires_queue_to_trade();
    test_adverse_selection_is_charged();
    test_book_imbalance();
    test_itch_endianness();
    test_itch_add_message();
    test_itch_symbol_filter();

    std::printf("\n%d checks, %d failures\n\n", checks, failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
