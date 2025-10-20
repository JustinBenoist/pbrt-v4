#!/usr/bin/env bash
# Replace any cmake_minimum_required version < 3.5 with 3.5
find src/ext -name CMakeLists.txt -exec \
  sed -i 's/cmake_minimum_required(VERSION [0-9.]\+)/cmake_minimum_required(VERSION 3.5)/g' {} +