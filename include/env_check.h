#pragma once

#include <cassert>
#include <string>

template <typename T>
struct ConvByType {};

template <>
struct ConvByType<int> {
  static void go(const char* varVal, int& ret) {
    assert(varVal);
    ret = std::atoi(varVal);
  }
};

template <>
struct ConvByType<double> {
  static void go(const char* varVal, double& ret) {
    assert(varVal);
    ret = std::atof(varVal);
  }
};

template <>
struct ConvByType<std::string> {
  static void go(const char* varVal, std::string& ret) {
    assert(varVal);
    ret = varVal;
  }
};

template <typename T>
bool genericGetEnv(const char* varName, T& ret) {
  char* varVal = getenv(varName);
  if (varVal) {
    ConvByType<T>::go(varVal, ret);
    return true;
  } else {
    return false;
  }
}

//! Return true if the Enviroment variable is set
bool EnvCheck(const char* varName);
bool EnvCheck(const std::string& varName);

/**
 * Return true if Enviroment variable is set, and extract its value into
 * 'retVal' parameter
 * @param varName: name of the variable
 * @param retVal: lvalue to store the value of environment variable
 * @return true if environment variable set, false otherwise
 */
template <typename T>
bool EnvCheck(const char* varName, T& retVal) {
  return genericGetEnv(varName, retVal);
}

template <typename T>
bool EnvCheck(const std::string& varName, T& retVal) {
  return EnvCheck(varName.c_str(), retVal);
}

