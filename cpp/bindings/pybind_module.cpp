// pybind11 surface. Step 4.6.
//
// Deliberately narrow. Python does not need the whole book API - it needs to
// replay a feed and ask what an order would cost. Every extra binding is
// another thing to keep working while the demo is on fire.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "execution_sim.hpp"
#include "itch_parser.hpp"
#include "market_impact.hpp"
#include "orderbook.hpp"
#include "types.hpp"

namespace py = pybind11;
using namespace aqc;

PYBIND11_MODULE(aqc_exec, m) {
    m.doc() = "Order book, ITCH replay and execution simulation (Lane A, Phase 4)";

    py::enum_<Side>(m, "Side")
        .value("BUY", Side::Buy)
        .value("SELL", Side::Sell);

    py::class_<Quote>(m, "Quote")
        .def_property_readonly("bid", [](const Quote& q) { return to_double(q.bid); })
        .def_property_readonly("ask", [](const Quote& q) { return to_double(q.ask); })
        .def_readonly("bid_size", &Quote::bid_size)
        .def_readonly("ask_size", &Quote::ask_size)
        .def_readonly("timestamp", &Quote::timestamp)
        .def("mid", &Quote::mid)
        .def("spread", &Quote::spread)
        .def("spread_bps", &Quote::spread_bps)
        .def("valid", &Quote::valid)
        .def("__repr__", [](const Quote& q) {
            return "<Quote " + std::to_string(to_double(q.bid)) + " x " +
                   std::to_string(to_double(q.ask)) + ">";
        });

    py::class_<ExecutionResult>(m, "ExecutionResult")
        .def_readonly("requested", &ExecutionResult::requested)
        .def_readonly("filled", &ExecutionResult::filled)
        .def_readonly("avg_price", &ExecutionResult::avg_price)
        .def_readonly("arrival_mid", &ExecutionResult::arrival_mid)
        .def_readonly("slippage_bps", &ExecutionResult::slippage_bps)
        .def_readonly("spread_bps", &ExecutionResult::spread_bps)
        .def_readonly("levels_consumed", &ExecutionResult::levels_consumed)
        .def_readonly("complete", &ExecutionResult::complete)
        .def_readonly("mode", &ExecutionResult::mode)
        .def("fill_ratio", &ExecutionResult::fill_ratio)
        .def("__repr__", [](const ExecutionResult& r) {
            return "<ExecutionResult " + r.mode + " filled=" +
                   std::to_string(r.filled) + "/" + std::to_string(r.requested) +
                   " slippage=" + std::to_string(r.slippage_bps) + "bps>";
        });

    py::class_<ItchStats>(m, "ItchStats")
        .def_readonly("messages", &ItchStats::messages)
        .def_readonly("adds", &ItchStats::adds)
        .def_readonly("executes", &ItchStats::executes)
        .def_readonly("cancels", &ItchStats::cancels)
        .def_readonly("deletes", &ItchStats::deletes)
        .def_readonly("replaces", &ItchStats::replaces)
        .def_readonly("trades", &ItchStats::trades)
        .def_readonly("skipped", &ItchStats::skipped)
        .def_readonly("first_ts", &ItchStats::first_ts)
        .def_readonly("last_ts", &ItchStats::last_ts);

    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def("add", [](OrderBook& b, OrderRef ref, Side side, double price, Shares shares) {
            b.add(ref, side, from_double(price), shares);
        }, py::arg("ref"), py::arg("side"), py::arg("price"), py::arg("shares"))
        .def("execute", &OrderBook::execute, py::arg("ref"), py::arg("shares"))
        .def("cancel", &OrderBook::cancel, py::arg("ref"), py::arg("shares"))
        .def("remove", &OrderBook::remove, py::arg("ref"))
        .def("clear", &OrderBook::clear)
        .def("top", &OrderBook::top, py::arg("timestamp") = 0)
        .def("size_at", [](const OrderBook& b, Side s, double price) {
            return b.size_at(s, from_double(price));
        })
        .def("level_count", &OrderBook::level_count)
        .def("order_count", &OrderBook::order_count)
        .def("queue_ahead", &OrderBook::queue_ahead, py::arg("ref"))
        .def("depth", [](const OrderBook& b, Side side, std::size_t levels) {
            std::vector<std::pair<double, Shares>> out;
            for (const auto& [p, s] : b.depth(side, levels)) {
                out.emplace_back(to_double(p), s);
            }
            return out;
        }, py::arg("side"), py::arg("levels") = 10)
        .def("imbalance", [](const OrderBook& b, std::size_t levels) {
            return book_imbalance(b, levels);
        }, py::arg("levels") = 5);

    py::class_<ItchParser>(m, "ItchParser")
        .def(py::init<OrderBook&, std::string>(),
             py::arg("book"), py::arg("symbol") = "",
             py::keep_alive<1, 2>())
        .def("process_file", &ItchParser::process_file,
             py::arg("path"), py::arg("max_messages") = 0,
             "Replay a length-prefixed ITCH file. Returns messages processed.")
        .def("process_buffer", [](ItchParser& p, py::bytes data) {
            std::string s = data;
            return p.process_buffer(
                reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
        }, py::arg("data"))
        .def_property_readonly("stats", &ItchParser::stats)
        .def_property_readonly("clock", &ItchParser::clock);

    py::class_<ExecutionConfig>(m, "ExecutionConfig")
        .def(py::init<>())
        .def_readwrite("max_participation", &ExecutionConfig::max_participation)
        .def_readwrite("max_levels", &ExecutionConfig::max_levels);

    py::class_<ExecutionSimulator>(m, "ExecutionSimulator")
        .def(py::init<ExecutionConfig>(), py::arg("config") = ExecutionConfig{})
        .def("market_order", &ExecutionSimulator::market_order,
             py::arg("book"), py::arg("side"), py::arg("shares"),
             "Cross the spread. Walks real depth, returns realised cost.")
        .def("limit_order", &ExecutionSimulator::limit_order,
             py::arg("book"), py::arg("side"), py::arg("shares"),
             py::arg("volume_at_level"), py::arg("price_moved_through") = false,
             "Post at the touch. Fills only once the queue ahead trades.")
        .def("sliced_order", &ExecutionSimulator::sliced_order,
             py::arg("book"), py::arg("side"), py::arg("shares"), py::arg("slices"));

    m.def("capacity_within", [](const OrderBook& b, Side side, double max_bps) {
        return capacity_within(b, side, max_bps);
    }, py::arg("book"), py::arg("side"), py::arg("max_bps"),
       "Shares tradeable without moving price more than max_bps.");

    m.attr("__version__") = "0.1.0";
}
