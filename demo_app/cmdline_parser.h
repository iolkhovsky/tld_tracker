#pragma once

#include <string>
#include <unordered_map>
#include <iostream>
#include <sstream>

std::unordered_map<std::string, std::string> parse_stream(std::istream& is);
std::unordered_map<std::string, std::string> parse(int argc, char** argv);

class CmdLineParser {

};
