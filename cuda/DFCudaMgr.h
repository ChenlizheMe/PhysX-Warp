// ----------------------------------------------------------------------------
// Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
//
// All rights reserved.
// ----------------------------------------------------------------------------
#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <algorithm>

#include "DFCudaCodes.h"

#define RENDERING_STREAM 0
#define CALCULATE_STREAM 1
#define GEOMETRY_STREAM 2
#define PHYSICS_STREAM 3
#define CUSTOM_STREAM 4

namespace dexsim {
namespace cudamgr {
class ICudaManager {
public:
    virtual ICudaFunctionManager* GetCuda() const = 0;
};

// External C function for initializing the Warp Manager
// \param mgr Output parameter that will receive the initialized manager
// instance
extern "C" void cudaInit(ICudaManager** mgr);

}  // namespace cudamgr
}  // namespace dexsim