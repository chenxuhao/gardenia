#pragma once
#include <iostream>
#include <string.h>
#include <cstdlib>
#include <sstream>
#include <cerrno>

void gPrintStr(const std::string& s) {
  std::cout << s << "\n";
}

void gErrorStr(const std::string& s) {
  std::cerr << s << "\n";
}

//! Prints a sequence of things
template <typename... Args>
void gPrint(Args&&... args) {
  std::ostringstream os;
  (os << ... << args);
  gPrintStr(os.str());
}

//! Prints error message
template <typename... Args>
void gError(Args&&... args) {
  std::ostringstream os;
  (os << ... << args);
  gErrorStr(os.str());
}

#define GARDENIA_DIE(...)                                              \
  do {                                                                 \
    gError(__FILE__, ":", __LINE__, ": ", ##__VA_ARGS__);              \
    abort();                                                           \
  } while (0)


