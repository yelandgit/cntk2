//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPUMatrix.cpp : full implementation of all matrix functions on the CPU side
//
#pragma once

#include "RNGHandle.h"

#ifndef CPUONLY
#include <curand.h>
#endif // !CPUONLY

namespace Microsoft { namespace MSR { namespace CNTK {

class GPURNGHandle : public RNGHandle
{
#ifndef CPUONLY
public:
    GPURNGHandle(device_t deviceId, uint64_t seed, uint64_t offset = 0);
    virtual ~GPURNGHandle();
    curandGenerator_t Generator() { return m_generator; }
private:
    curandGenerator_t	m_generator;
#else
public:
    GPURNGHandle(device_t deviceId, uint64_t seed, uint64_t offset = 0) : RNGHandle(deviceId) {}
    virtual ~GPURNGHandle() {}
#endif
};

} } }
