//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// RNGHandle.cpp : an abstraction around a random number generator
//
#include "stdafx.h"
#include "RNGHandle.h"
#include "CPURNGHandle.h"
#include "GPURNGHandle.h"

namespace Microsoft { namespace MSR { namespace CNTK {

std::shared_ptr<RNGHandle> RNGHandle::Create(device_t deviceId, uint64_t seed, uint64_t offset)
{
	if (deviceId == CPUDEVICE)
		return std::make_shared<CPURNGHandle>(deviceId, seed, offset);
	return std::make_shared<GPURNGHandle>(deviceId, seed, offset);
}

} } }
