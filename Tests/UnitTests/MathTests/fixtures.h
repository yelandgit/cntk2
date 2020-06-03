//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

class RandomSeedFixture
{
    static unsigned long s_counter;
public:
    RandomSeedFixture();
    unsigned long GetSeed() { return ++s_counter; }
};

class DeterministicCPUAlgorithmsFixture
{
public:
    DeterministicCPUAlgorithmsFixture();
};