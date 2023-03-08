# Copyright 2023 Garrett Kinman
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

using Flux
using Parameters: @with_kw

# TODO: add crossover functions
"""
    Genetic(popsize = 100, ϵ = 1)

Genetic algorithm-based optimizer with a given population size, crossover rate, mutation rate, and ϵ survivors of each generation.
"""
@with_kw mutable struct Genetic <: Flux.Optimise.AbstractOptimiser
    popsize::Int32 = 100
    crossover_rate::Float32 = 1 / 5
    mutation_rate::Float32 = 1 / 100
    ϵ::Int32 = 10
end

function apply!(o::Genetic, x, Δ)
    # return a Δ for each element of x (which is a subset of model params)
    # the calling funtion will call x .-= apply!(opt, x, x̄r), where x̄r is just the gradients corresponding to x
    # treat Δ as all zeros to start with, as we will zero-initialize them to feed in,
    # because this package was designed with gradient-based methods in mind, and thus requires you feed in gradients (even if not real)

    # TODO
    # randomly generate a population of Δ candidates (of size o.popsize)
    # perform a feed-through the network for each candidate Δ
    # select the surviving Δ based on minimized loss

end

# TODO: add adaptive genetic optimizer, which modifies the crossover and mutation rates over time