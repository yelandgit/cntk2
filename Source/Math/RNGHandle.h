//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// RNGHandle.h: An abstraction around a random number generator
//
#pragma once

#include "BaseMatrix.h"
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

class MATH_API RNGHandle
{
	device_t	m_deviceId;

protected:
	RNGHandle(device_t deviceId) : m_deviceId(deviceId) {}
public:
	virtual ~RNGHandle() {}

	device_t GetDeviceId() const { return m_deviceId; }

	static std::shared_ptr<RNGHandle> Create(device_t deviceId, uint64_t seed, uint64_t offset=0);
};
typedef std::shared_ptr<RNGHandle> RNGHandlePtr;

} } }
