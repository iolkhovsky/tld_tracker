#include "cmdline_parser.h"

std::unordered_map<std::string, std::string> parse_stream(std::istream& is) {
    std::unordered_map<std::string, std::string> out;
    std::string buf;
    while (is >> buf) {
        std::size_t prefix_pos = buf.find("--");
        if (prefix_pos == 0) {
            std::size_t equal_pos = buf.find("=");
            if (equal_pos != std::string::npos) {
                std::string key = std::string(std::next(buf.begin(), 2), std::next(buf.begin(), equal_pos));
                std::string val = std::string(std::next(buf.begin(), equal_pos+1), buf.end());
                out[key] = val;
            } else {
                std::string key = std::string(std::next(buf.begin(), 2), buf.end());
                out[key] = "unknown";
            }
        }
    }
    return out;
}

std::unordered_map<std::string, std::string> parse(int argc, char** argv) {
    std::stringstream ss;
    for (int i = 0; i < argc; i++) {
        if (i)
            ss << " ";
        ss << argv[i];
    }
    return parse_stream(ss);
}
